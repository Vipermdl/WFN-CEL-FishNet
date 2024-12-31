import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.fishnet import get_fishdata
from utils import AverageMeter, accuracy
from sklearn.metrics import confusion_matrix, f1_score


logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class WarmupMultiStepLR(MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_steps = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.last_epoch < self.warmup_steps:
            return [
                lr * self.warmup_steps**0.5 * min(step_num**-0.5, step_num * self.warmup_steps**-1.5) for lr in self.base_lrs
            ]
        else:
            return super().get_lr()


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def inference(current_logit, qhat, tau=0.5):
    debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
    return debiased_prob

def initial_qhat(class_num=1000):
    # initialize qhat of predictions (probability)
    qhat = (torch.ones([1, class_num], dtype=torch.float)/class_num).cuda()
    return qhat

def update_qhat(probs, qhat, momentum, qhat_mask=None):
    if qhat_mask is not None:
        mean_prob = probs.detach()*qhat_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    qhat = momentum * qhat + (1 - momentum) * mean_prob
    return qhat



def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--ratio-labeled', type=float, default=0.1,
                        help='number of labeled data')
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['wideresnet', 'resnext', 'resnet'],
                        help='network name')
    parser.add_argument('--total-steps', default=2**17.64, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=12, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup-epoch', default=0.05, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=3e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=4, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=10, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=0.4, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'resnet50':
            import models.resnet as models
            model = models.build_resnet(
                name='Dual_ResNet50',
                num_classes=args.num_classes
            )
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    meta_df = pd.read_csv(os.path.join('data', 'train_full_meta_new.csv'))
    args.num_classes = len(list(set(meta_df['Family'].values)))

    import pdb; pdb.set_trace()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = get_fishdata(
        args, root='data', meta_df=meta_df)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)

    warmup_step = args.warmup_epoch * args.eval_step

    milestones = [x * args.eval_step for x in [30, 60, 100, 150]]    

    scheduler = WarmupMultiStepLR(optimizer, milestones=milestones, gamma=0.5, warmup_steps=warmup_step)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, meta_df)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, meta_df):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    
    labels_common = list(set(np.asarray(meta_df.loc[meta_df['fam_info']==0]['Family_cls'])))
    labels_medium = list(set(np.asarray(meta_df.loc[meta_df['fam_info']==1]['Family_cls'])))
    labels_rare = list(set(np.asarray(meta_df.loc[meta_df['fam_info']==2]['Family_cls'])))
    labels_all = list(set(np.asarray(meta_df.loc[meta_df['fam_info']>=0]['Family_cls'])))

    qhat = initial_qhat(class_num=args.num_classes)

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()            

            data_time.update(time.time() - end)
            
            batch_size = inputs_x['LF'].shape[0]
            inputs_LF = interleave(torch.cat((inputs_x['LF'], inputs_u_w['LF'], inputs_u_s['LF'])), 2*args.mu+1).to(args.device)
            inputs_HF = interleave(torch.cat((inputs_x['HF'], inputs_u_w['HF'], inputs_u_s['HF'])), 2*args.mu+1).to(args.device)

            targets_x = targets_x.to(args.device)
            logits = model(inputs_LF, inputs_HF)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            pseudo_label = inference(logits_u_w.detach(), qhat, tau=args.T)
            max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            qhat = update_qhat(torch.softmax(logits_u_w.detach(), dim=-1), qhat, momentum=0.99, qhat_mask=None)
            delta_logits = torch.log(qhat)
            logits_u_s = logits_u_s + args.T * delta_logits #(+,-)
            Lu = (F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none', weight=None) * mask).mean()

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # original loss
            # pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            # mask = max_probs.ge(args.threshold).float()
            # Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            
            loss = Lx + args.lambda_u * (epoch / args.epochs) * Lu  

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg 
                    )) 
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            loss, test_acc, avg_cls_acc_1, avg_cls_acc_2, avg_cls_acc_3, avg_cls_acc_4 = test(args, test_loader, test_model, epoch, labels_common, labels_medium, labels_rare, labels_all)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.loss', loss, epoch)
            args.writer.add_scalar('test/common_acc', avg_cls_acc_1, epoch)
            args.writer.add_scalar('test/medium_acc', avg_cls_acc_2, epoch)
            args.writer.add_scalar('test/rare_acc', avg_cls_acc_3, epoch)
            args.writer.add_scalar('test/all_acc', avg_cls_acc_4, epoch)

            is_best = avg_cls_acc_4 > best_acc
            best_acc = max(avg_cls_acc_4, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': avg_cls_acc_4,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'qhat': qhat
            }, is_best, args.out)

            test_accs.append(avg_cls_acc_4)
            logger.info('Best all acc: {:.2f}'.format(best_acc))
            logger.info('Mean all acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch, labels_common, labels_medium, labels_rare, labels_all):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        all_preds = []
        all_gts = []
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs_LF = inputs['LF'].to(args.device)
            inputs_HF = inputs['HF'].to(args.device)
            
            targets = targets.to(args.device)

            outputs = model(inputs_LF, inputs_HF)
            
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            outputs = torch.max(outputs,-1)[1]
            all_preds.append(outputs.data.cpu().numpy())
            all_gts.append(targets.data.cpu().numpy())

            losses.update(loss.item(), inputs['LF'].shape[0])
            top1.update(prec1.item(), inputs['LF'].shape[0])
            top5.update(prec5.item(), inputs['LF'].shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_gts)

        matrix = confusion_matrix(y_true, y_pred)
        cls_acc = matrix.diagonal()/ matrix.sum(axis=1)
        cls_f1 = f1_score(y_true, y_pred, average=None)

        avg_cls_acc_1 = np.mean(cls_acc[labels_common])
        avg_cls_acc_2 = np.mean(cls_acc[labels_medium])
        avg_cls_acc_3 = np.mean(cls_acc[labels_rare])
        avg_cls_acc_4 = np.mean(cls_acc[labels_all])
        
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    logger.info("common acc: {:.2f}".format(avg_cls_acc_1))
    logger.info("medium acc: {:.2f}".format(avg_cls_acc_2))
    logger.info("rare acc: {:.2f}".format(avg_cls_acc_3))
    logger.info("all acc: {:.2f}".format(avg_cls_acc_4))
    
    return losses.avg, top1.avg, avg_cls_acc_1, avg_cls_acc_2, avg_cls_acc_3, avg_cls_acc_4


if __name__ == '__main__':
    main()
