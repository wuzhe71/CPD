import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc

from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_path = 'path/dataset/'

if opt.is_ResNet:
    model = CPD_ResNet()
    model.load_state_dict(torch.load('CPD-R.pth'))
else:
    model = CPD_VGG()
    model.load_state_dict(torch.load('CPD.pth'))

model.cuda()
model.eval()

test_datasets = ['PASCAL', 'ECSSD', 'DUT-OMRON', 'DUTS-TEST', 'HKUIS']

for dataset in test_datasets:
    if opt.is_ResNet:
        save_path = './results/ResNet50/' + dataset + '/'
    else:
        save_path = './results/VGG16/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/images/'
    gt_root = dataset_path + dataset + '/gts/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        _, res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)
