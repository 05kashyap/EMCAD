import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch

def _find_file(root, base, exts):
    for ext in exts:
        p = os.path.join(root, base + ext)
        print(p)
        if os.path.isfile(p):
            return p
    return None

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations, file_list=None):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)

        # Ensure roots end with a separator to match list-concatenation style used here
        if not image_root.endswith(os.sep):
            image_root = image_root + os.sep
        if not gt_root.endswith(os.sep):
            gt_root = gt_root + os.sep

        img_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        msk_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

        if file_list is not None and os.path.isfile(file_list):
            imgs, gts = [], []
            with open(file_list, 'r') as f:
                for line in f:
                    toks = line.strip().split()
                    if not toks:
                        continue
                    if len(toks) == 1:
                        token = toks[0]
                        base, ext = os.path.splitext(token)
                        if ext:  # has extension, treat as relative file name
                            img_path = os.path.join(image_root, token)
                            msk_path = os.path.join(gt_root, token)
                        else:    # stem only: resolve by trying known extensions
                            img_path = _find_file(image_root, token, img_exts)
                            msk_path = _find_file(gt_root,   token, msk_exts)
                    else:
                        img_rel, msk_rel = toks[0], toks[1]
                        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
                        msk_path = msk_rel if os.path.isabs(msk_rel) else os.path.join(gt_root, msk_rel)

                    if img_path and msk_path and os.path.isfile(img_path) and os.path.isfile(msk_path):
                        imgs.append(img_path)
                        gts.append(msk_path)
            self.images = imgs
            self.gts = gts
        else:
            # Fallback: pair by basename (stem) to ensure correct alignment
            img_files = [os.path.join(image_root, f) for f in os.listdir(image_root)
                         if f.lower().endswith(tuple(img_exts))]
            msk_files = [os.path.join(gt_root, f) for f in os.listdir(gt_root)
                         if f.lower().endswith(tuple(msk_exts))]
            def stem(p): 
                b = os.path.basename(p)
                return os.path.splitext(b)[0]
            msk_map = {stem(p): p for p in msk_files}
            self.images, self.gts = [], []
            for p in img_files:
                s = stem(p)
                if s in msk_map:
                    self.images.append(p)
                    self.gts.append(msk_map[s])

        self.size = len(self.images)
        if self.size == 0:
            raise RuntimeError(f'PolypDataset is empty. images={image_root}, masks={gt_root}, list={file_list}')

        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        seed = np.random.randint(2147483647)
        random.seed(seed); torch.manual_seed(seed)
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed); torch.manual_seed(seed)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def convert2polar(self, img, gt):
    
    	center = polar_transformations.centroid(gt)
    	img = polar_transformations.to_polar(img, center)
    	gt = polar_transformations.to_polar(gt, center)
    	
    	return img, gt
            #center_max_shift = 0.05 * LesionDataset.height
            #center = np.array(center)
            #center = (
               #center[0] + np.random.uniform(-center_max_shift, center_max_shift),
               #center[1] + np.random.uniform(-center_max_shift, center_max_shift))
    ## to PyTorch expected format
    #input = input.transpose(2, 0, 1)
    #label = np.expand_dims(label, axis=-1)
    #label = label.transpose(2, 0, 1)

    #input_tensor = torch.from_numpy(input)
    
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=False, num_workers=4, pin_memory=True, augmentation=False): #shuffle=True

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.gts = [gt_root + f for f in os.listdir(gt_root)
                    if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
