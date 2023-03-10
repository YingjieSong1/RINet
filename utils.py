import cv2, os
import torch
from torchvision import utils
from tqdm import tqdm
from PIL import Image
from dataloader import Sal_Dataset
from loss import *

def create_model(args):
    print(f'enc_model:{args.enc_model} \ntrain_enc:{bool(args.train_enc)} ')
    if args.enc_model == "densenet_fusion_3":
        from module.model_densenet_fusion_3 import Dense_Model       
        model = Dense_Model(train_enc=bool(args.train_enc), load_weight=True,type='RI_spacial_channel')
    else:
        raise AssertionError('Please check args.enc_model!')

    if args.load_weight!=None:
        model.load_state_dict(torch.load(args.load_weight))
        print(f'load weight from: {args.load_weight}')
    
    import shutil
    if os.path.exists(args.model_val_path):
        shutil.rmtree(args.model_val_path)
    if os.path.exists(args.log_val_path):
        shutil.rmtree(args.log_val_path)
    os.makedirs(args.model_val_path,exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model.to(device)
    return model,device



def create_optimizer_scheduler(model,args,length):
    params = list(filter(lambda p: p.requires_grad, model.parameters())) 

    if args.optim=="Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.optim=="Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=args.lr)
    if args.optim=="SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    return optimizer,scheduler

def dataset_process(args):
    if args.dataset_name=='salicon':
        dataset_dir=args.dataset_dir
        exten=['.jpg','.png','.mat']
    else:
        raise AssertionError('Please check the dataset for training!')
    print(f'dataset name: {args.dataset_name}')

    train_img_dir = os.path.join(dataset_dir,'images','train')
    train_gt_dir = os.path.join(dataset_dir , 'maps','train')
    train_fix_dir = os.path.join(dataset_dir , 'fixations','train')

    val_img_dir =os.path.join(dataset_dir , 'images','val')
    val_gt_dir = os.path.join(dataset_dir , 'maps','val')
    val_fix_dir = os.path.join(dataset_dir , 'fixations','val')

    complexitys_dir=os.path.join(dataset_dir , 'complexitys')

    train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
    val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]

    train_dataset = Sal_Dataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids,complexitys_dir,arg=args,exten=exten)
    val_dataset =   Sal_Dataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids,complexitys_dir,arg=args,exten=exten)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
    return train_loader,val_loader



def img_save(tensor, fp, nrow=8, padding=2,
               normalize=True, range=None, scale_each=False, pad_value=0, format=None):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ''' Add 0.5 after unnormalizing to [0, 255] to round to nearest integer '''
    
    ndarr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()[:,:,0]
    im = Image.fromarray(ndarr)
    exten = fp.split('.')[-1]
    if exten=="png":
        im.save(fp, format=format, compress_level=0)
    else:
        im.save(fp, format=format, quality=100) #for jpg

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def test_model(model, loader, device, args):
    with torch.no_grad():
        model.eval()
        results_dir=os.path.join(args.results_dir,args.dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        
        for (img, img_id, sz) in tqdm(loader):
            img = img.to(device)
            
            pred_map = model(img)
            
            img=img.cpu().squeeze(0).clone().permute(1,2,0).numpy()
            pred_map = pred_map.cpu().squeeze(0).clone().numpy()
            pred_map = cv2.resize(pred_map, (int(sz[0]), int(sz[1])))
            
            pred_map = torch.FloatTensor(pred_map)
            img_save(pred_map, os.path.join(results_dir, img_id[0].replace('.jpg','.png')), normalize=True)



