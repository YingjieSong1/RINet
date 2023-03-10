
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch
import os, cv2
from scipy import io

    
def resize_fixation(image, row, col):
    resized_fixation = np.zeros((row, col))
    ratio_row = row / image.shape[0]
    ratio_col = col / image.shape[1]

    coords = np.argwhere(image)
    for coord in coords:
        coord_r = int(np.round(coord[0]*ratio_row))
        coord_c = int(np.round(coord[1]*ratio_col))
        if coord_r == row:
            coord_r -= 1
        if coord_c == col:
            coord_c -= 1
        resized_fixation[coord_r, coord_c] = 1

    return resized_fixation

def get_complexitys(path,id):
    f = open(os.path.join(path,f'{id}.txt'), "r")
    c=int(f.read())
    c=np.array(c)
    c=np.expand_dims(c,axis=[0])
    return c

class Sal_Dataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids,complexitys_dir=None,arg=None, exten=['.jpg','.png','.mat'],val=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.exten = exten
        self.val=val
        self.img_size=arg.img_size
        self.arg=arg
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.complexitys_path=complexitys_dir
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.exten[0])
        gt_path = os.path.join(self.gt_dir, img_id + self.exten[1])
        fix_path = os.path.join(self.fix_dir, img_id + self.exten[2])
        
        img_origin = Image.open(img_path).convert('RGB')
        sz = img_origin.size

        img = self.img_transform(img_origin)

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, [self.img_size[1],self.img_size[0]])
        if np.max(gt) > 1.0:
            gt = gt / 255.0

        if self.exten[2]=='.mat':
            fixations = io.loadmat(fix_path)
            if 'im' in fixations.keys():
                fixations = fixations['im']
            elif 'fixLocs' in fixations.keys():
                fixations = fixations['fixLocs']
        else:
            fixations = np.array(Image.open(fix_path).convert('L'))

        fixations=resize_fixation(fixations,self.img_size[0],self.img_size[1])
        fixations = fixations.astype('float')
        fixations = (fixations > 0.5).astype('float')

        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        if self.val==False :
            complexitys=get_complexitys(self.complexitys_path,img_id)
            return img, torch.FloatTensor(gt), torch.FloatTensor(fixations),torch.FloatTensor(complexitys)
        elif self.val==True:
            return img, torch.FloatTensor(gt), torch.FloatTensor(fixations),img_id,sz,img_path,gt_path
        
        
    def __len__(self):
        return len(self.img_ids)    


class TestLoader(DataLoader):
    def __init__(self, img_dir, img_ids,arg):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.img_size=arg.img_size
        self.img_transform = transforms.Compose([
    
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])    
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert('RGB')
        sz = img.size
        img = self.img_transform(img)
        return img, img_id, sz
    
    def __len__(self):
        return len(self.img_ids)

