from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os, math
import pywt
from .randaugment import RandAugmentMC, Resize, RandomHorizontalFlip, RandomCrop
from .randaugment import ToTensor, Normalize, CenterCrop



class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, meta_df, transform=None, index=None, wavelet_type='db2'):
        self.data_frame = pd.read_csv(csv_file)
        # select the ratio to train
        self.root_dir = root_dir
        self.transform = transform
        self.image_col = "image"
        self.folder_col = "Folder"
        self.meta_df = meta_df
        self.targets = self.get_targets(csv_file)
        self.data = self.get_data(csv_file)
        self.wavelet_type = wavelet_type
    
    def get_targets(self, csv_file):
        if os.path.exists(csv_file[:-4]+'_targets.npy'):
            targets = np.load(csv_file[:-4]+'_targets.npy')
            return targets
        else:
            targets = []
            for idx in range(len(self.data_frame)):
                cls_name = self.data_frame.iloc[idx]["Family"]
                targets.append(self.meta_df.loc[self.meta_df['Family']==cls_name]['Family_cls'].values[0])
            np.save(csv_file[:-4]+'_targets.npy', np.array(targets))
            return np.array(targets)

    def get_data(self, csv_file):
        if os.path.exists(csv_file[:-4]+'_data.npy'):
            data = np.load(csv_file[:-4]+'_data.npy')
            return data
        else:
            data = []
            for idx in range(len(self.data_frame)):
                img_name = self.data_frame.iloc[idx][self.image_col]
                img_name = img_name.split('/')[-1]
                folder = self.data_frame.iloc[idx][self.folder_col]
                img_path = os.path.join(folder, img_name)
                data.append(os.path.join(self.root_dir, img_path))
            np.save(csv_file[:-4]+'_data.npy', np.array(data))
            return np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx], self.targets[idx]
        image = Image.open(path)

        image = np.array(image)
        LL, (LH, HL, HH) = pywt.dwt2(image, self.wavelet_type)
        LL = (LL - LL.min()) / (LL.max() - LL.min()) * 255
        LF = Image.fromarray(LL.astype(np.uint8))
        
        LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255
        HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255
        HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255
        HF = HH + HL + LH
        HF = (HF-HF.min()) / (HF.max()-HF.min()) * 255
        HF = Image.fromarray(HF.astype(np.uint8))

        inputs = {'LF': LF, 'HF': HF}
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, label

def x_u_split(args, labels):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    # unlabeled_idx = range(len(labels))

    for i in range(args.num_classes):
        tot_idx = np.where(labels == i)[0]
        label_per_class =  math.ceil(args.ratio_labeled * len(tot_idx))
        idx = np.random.choice(tot_idx, label_per_class, False)
        labeled_idx.extend(idx)
        unlabeled_idx.extend(np.array(list(set(tot_idx).difference(set(idx)))))

    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


class FishSSL(ImageDataset):
    def __init__(self, indexs, csv_file, root_dir, meta_df, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, 
                         meta_df=meta_df, transform=transform)
        if indexs is not None:
            self.data = np.array(self.data)[indexs]
            self.targets = np.array(self.targets)[indexs]


class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            Resize(256),
            RandomHorizontalFlip(),
            RandomCrop(size=224,
                    padding=int(32*0.125),
                    padding_mode='reflect')])
        self.strong = transforms.Compose([
            Resize(256),
            RandomHorizontalFlip(),
            RandomCrop(size=224,
                    padding=int(32*0.125),
                    padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        
        self.normalize = transforms.Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


def get_fishdata(args, root, meta_df):
    transform = transforms.Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])
    base_dataset = ImageDataset(
        csv_file = os.path.join(root, 'train.csv'),
        root_dir = os.path.join(root, 'FishNet_Image_Library'),
        meta_df = meta_df,
        transform = None
    )

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)
    
    train_labeled_dataset = FishSSL(
        indexs=train_labeled_idxs, 
        csv_file= os.path.join(root, 'train.csv'),
        root_dir= os.path.join(root, 'FishNet_Image_Library'), 
        meta_df=meta_df, transform=transform
    )
    train_unlabeled_dataset = FishSSL(
        indexs = train_unlabeled_idxs, 
        csv_file= os.path.join(root, 'train.csv'),
        root_dir= os.path.join(root, 'FishNet_Image_Library'), 
        meta_df=meta_df, transform=TransformFixMatch()
    )

    test_dataset = ImageDataset(
        csv_file= os.path.join(root, 'test.csv'),
        root_dir= os.path.join(root, 'FishNet_Image_Library'), 
        meta_df=meta_df, transform=transform
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

