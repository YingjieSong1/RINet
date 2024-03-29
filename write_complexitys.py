from glob import glob
import argparse
import numpy as np
import os
from skimage.feature import peak_local_max
from PIL import Image
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', type=str)
args = parser.parse_args()

map_path_val=glob.glob(os.path.join(args.dataset_dir,'maps/val/*.png'))
map_path_train=glob.glob(os.path.join(args.dataset_dir,'maps/train/*.png'))
map_paths=map_path_val+map_path_train
txt_out_path=os.path.join(args.dataset_dir,'complexitys')

os.makedirs(txt_out_path,exist_ok=True)
 
complexitys=[]
complexitys_record=[]
for map_path in tqdm(map_paths):
    map_name=os.path.basename(map_path).split('.')[0]
    im=np.array(Image.open(map_path).convert('L'))
    im = im.astype('float')

    coordinates = peak_local_max(im, min_distance=20)
    complexitys.append(coordinates.shape[0])
    complexitys_record.append([map_name,coordinates.shape[0]])

complexitys_unique= np.unique(complexitys)
complexitys_num=[]
for i in complexitys_unique:
    complexitys_num.append(int(sum(complexitys==i)))
print(f'complexitys unique={complexitys_unique}, num={complexitys_num}')

for map_name,complexity in tqdm(complexitys_record):
    map_name=map_name.replace('_fixMap','')
    with open(os.path.join(txt_out_path,f'{map_name}.txt'),"w") as f:
        f.write(f'{complexity}')
