import argparse
import os
import torch
from dataloader import TestLoader
from utils import test_model

parser = argparse.ArgumentParser()

parser.add_argument('--img_size',default=(288,384), type=int)
parser.add_argument('--dataset_name',default='salicon', type=str)
parser.add_argument('--model_val_path',default="./best_model_on_salicon.pt", type=str)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--results_dir',default="./result/", type=str)
parser.add_argument('--test_img_dir', type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(args.model_val_path)

from module.model_densenet_fusion_3 import Dense_Model       
model = Dense_Model(train_enc=False, load_weight=False,type='RI_spacial_channel')
weights=torch.load(args.model_val_path)
weights_dict = {}
for k, v in weights.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v

model.load_state_dict(weights_dict)
model = model.to(device)
os.makedirs(args.results_dir,exist_ok=True)
   
test_img_ids = os.listdir(args.test_img_dir)
test_dataset = TestLoader(args.test_img_dir, test_img_ids,arg=args)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
test_model(model, test_loader, device, args)

    







