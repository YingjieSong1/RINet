import torch
import cv2


def loss_func(pred_map, gt, fixations,args):
    loss = torch.FloatTensor([0.0]).cuda()
 
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += args.cc_coeff * cc(pred_map, gt)
    if args.nss:
        loss += args.nss_coeff * nss(pred_map, fixations)

    return loss

def kldiv(s_map, gt):
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    assert expand_gt.size() == gt.size()

    s_map = s_map/(expand_s_map*1.0)
    gt = gt / (expand_gt*1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 2.2204e-16
    result = gt * torch.log(eps + gt/(s_map + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))



def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    
    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map)/(max_s_map-min_s_map*1.0)
    return norm_s_map



def cc(s_map, gt):
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    return torch.mean(ab / (torch.sqrt(aa*bb)))

def nss(s_map, gt):
    if s_map.size() != gt.size():
        s_map = s_map.cpu().squeeze(0).numpy()
        s_map = torch.FloatTensor(cv2.resize(s_map, (gt.size(2), gt.size(1)))).unsqueeze(0)
        s_map = s_map.cuda()
        gt = gt.cuda()
    # print(s_map.size(), gt.size())
    assert s_map.size()==gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    eps = 2.2204e-16
    s_map = (s_map - mean_s_map) / (std_s_map + eps)

    s_map = torch.sum((s_map * gt).view(batch_size, -1), 1)
    count = torch.sum(gt.view(batch_size, -1), 1)
    return torch.mean(s_map / count)

