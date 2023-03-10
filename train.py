import argparse
import os
import torch
import sys
import time
import numpy as np
from loss import *
from loss_weight import loss_func_weight
from utils import *
from tensorboardX import SummaryWriter
import random

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loss_weight(complexity,no_epochs,global_steps,len_loader):
    gama=5.0
    alpha=10.0
    p=min(global_steps/(no_epochs*len_loader),1)
    tmp=2.0/(1+np.exp(-1*gama*p))
    weight=1+(complexity/alpha)*(tmp-1)
    return weight[:,0]


def train(model, optimizer, loader, epoch, device, args):
    model.train()
    global tic_train_begin 
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations,complexity) in enumerate(tqdm(loader)):
        global global_steps
        global_steps=global_steps+1

        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        
        optimizer.zero_grad()

        pred_map= model(img)
        assert pred_map.size() == gt.size()
        if args.CRFloss:
            weight=loss_weight(complexity,args.no_epochs,global_steps,len(loader)).to(device)
            loss= loss_func_weight(pred_map, gt, fixations,args,weight)
        else:
            loss= loss_func(pred_map, gt, fixations,args)

        loss.backward()
        total_loss += loss.item()
        cur_loss += loss.item()
        
        optimizer.step()
        global log_writer
        log_writer.add_scalar("loss/train_loss", loss.item(), global_step=global_steps)
        log_writer.add_scalar("loss/lr", optimizer.param_groups[0]['lr'], global_step=global_steps)
    

        if idx%args.log_interval==0:
            tqdm.write('[{:3d}, {:5d},{:6d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, global_steps,cur_loss/args.log_interval, (time.time()-tic_train_begin)/60))
            cur_loss = 0.0

    tqdm.write('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    # sys.stdout.flush()
    return total_loss/len(loader)


def validate(model, loader, device, args):
    model.eval()
    val_loss=AverageMeter()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    
    for img, gt, fixations,complexity in tqdm(loader):

        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        pred_map= model(img)
        loss = loss_func(pred_map, gt, fixations, args)

        assert pred_map.size() == gt.size()
        
        cc_loss.update(cc(pred_map, gt))    
        kldiv_loss.update(kldiv(pred_map, gt))    
        nss_loss.update(nss(pred_map, fixations))    
        val_loss.update(loss)

    global log_writer,global_steps
    log_writer.add_scalar("accuracy/val_cc", cc_loss.avg, global_step=global_steps)
    log_writer.add_scalar("accuracy/val_nss", nss_loss.avg, global_step=global_steps)
    log_writer.add_scalar("accuracy/val_kldiv", kldiv_loss.avg, global_step=global_steps)
    log_writer.add_scalar("loss/val_loss", val_loss.avg, global_step=global_steps)

    acc_str=f'val: CC : {cc_loss.avg}, KLDIV : { kldiv_loss.avg}, NSS : {nss_loss.avg}'
    tqdm.write(acc_str)
    sys.stdout.flush()
    
    return val_loss.avg,acc_str


##############################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_size',default= (288,384))
    parser.add_argument('--dataset_name',default='salicon', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--load_weight',default=None, type=str)
    parser.add_argument('--no_epochs',default=10, type=int)
    parser.add_argument('--lr',default=1e-4, type=float)
    parser.add_argument('--batch_size',default=8, type=int)
    parser.add_argument('--step_size',default=4, type=int)

    parser.add_argument('--kldiv',default=True, type=bool)
    parser.add_argument('--cc',default=True, type=bool)
    parser.add_argument('--nss',default=False, type=bool)
    parser.add_argument('--enc_model',default="densenet_fusion_3", type=str)
    parser.add_argument('--optim',default="Adam", type=str)

    parser.add_argument('--kldiv_coeff',default=1.0, type=float)
    parser.add_argument('--cc_coeff',default=-1.0, type=float)
    parser.add_argument('--nss_coeff',default=0.0, type=float)
    parser.add_argument('--train_enc',default=1, type=int)
    parser.add_argument('--CRFloss',default=True, type=int)

    parser.add_argument('--log_interval',default=30, type=int)
    parser.add_argument('--no_workers',default=4, type=int)
    parser.add_argument('--model_val_path',default="./result/RINet/model_saved/")
    parser.add_argument('--log_val_path',default="./result/RINet/log/")

    global_seed=random.randint(0,500)
    fix_seed(global_seed)
    print(f'global_seed:{global_seed}')

    args,unknown = parser.parse_known_args()
    print(f'model_val_path:{args.model_val_path}')
    print(f'CRFloss:{args.CRFloss}')
    train_loader,val_loader= dataset_process(args)
    model,device=create_model(args)
    optimizer,scheduler=create_optimizer_scheduler(model,args,length=len(train_loader))

    global log_writer
    log_writer = SummaryWriter(args.log_val_path)
    global global_steps
    global_steps=-1
    best_model_path=os.path.join(args.model_val_path,"best_model.pt")
    tic_train_begin = time.time()
    for epoch in range(0, args.no_epochs):

        if epoch != 0:
                model_dict = model.state_dict()
                pretrained_dict = torch.load(best_model_path)
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

        train_loss = train(model, optimizer, train_loader, epoch, device, args)
        
        with torch.no_grad():
            loss,acc_str = validate(model, val_loader, device, args)
            
            if epoch == 0 :
                best_loss = loss
            if best_loss >= loss:
                best_loss = loss
                print('[{:2d},  save, {}]'.format(epoch, best_model_path))
                with open(os.path.join(args.model_val_path,"best_model.txt"),"a") as f:
                        f.write(f'epoch:{epoch}, global_steps:{global_steps}, {acc_str},global_seed:{global_seed},train loss:{train_loss}\n')
                
                torch.save(model.state_dict(), best_model_path)
          
            torch.save(model.state_dict(), os.path.join(args.model_val_path,f'model_{epoch}.pt'))
    
        scheduler.step()