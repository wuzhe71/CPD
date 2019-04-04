import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc

from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data import test_dataset
from metric import cal_mae, cal_maxF

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')

opt = parser.parse_args()

if opt.is_ResNet:
    model = CPD_ResNet()
    model.load_state_dict(torch.load('CPD-R.pth'))
    # pre_model = torch.load('./model/CPD-R')
    # all_params = {}
    # for key in pre_model.keys():
    #     if key in model.state_dict().keys():
    #         all_params[key] = pre_model[key]
    #     elif 'Incep' in key:
    #         name = 'rfb' + key.split('Incep')[1]
    #         if name in model.state_dict().keys():
    #             all_params[name] = pre_model[key]
    #     elif 'SA' in key:
    #         name = 'HA' + key.split('SA')[1]
    #         if name in model.state_dict().keys():
    #             all_params[name] = pre_model[key]
    # assert len(all_params) == len(model.state_dict().keys())
    # model.load_state_dict(all_params)
else:
    model = CPD_VGG()
    model.load_state_dict(torch.load('CPD.pth'))
    # pre_model = torch.load('./model/CPD')
    # all_params = {}
    # for key in pre_model.keys():
    #     if key in model.state_dict().keys():
    #         all_params[key] = pre_model[key]
    #     elif 'Incep' in key:
    #         name = 'rfb' + key.split('Incep')[1]
    #         if name in model.state_dict().keys():
    #             all_params[name] = pre_model[key]
    #     elif 'SA' in key:
    #         name = 'HA' + key.split('SA')[1]
    #         if name in model.state_dict().keys():
    #             all_params[name] = pre_model[key]
    # assert len(all_params) == len(model.state_dict().keys())
    # model.load_state_dict(all_params)


model.cuda()
model.eval()

test_datasets = ['PASCAL', 'ECSSD', 'DUT-OMRON', 'DUTS-TEST', 'HKUIS', 'THUR15K', 'SOD']

for dataset in test_datasets:
    if opt.is_ResNet:
        save_path = './results/ResNet50/' + dataset + '/'
        # torch.save(model.state_dict(), 'CPD-R.pth')
    else:
        save_path = './results/VGG16/' + dataset + '/'
        torch.save(model.state_dict(), 'CPD.pth')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = '/backup/materials/Dataset/SalientObject/dataset/' + dataset + '/images/'
    gt_root = '/backup/materials/Dataset/SalientObject/dataset/' + dataset + '/gts/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    mae, maxF = cal_mae(), cal_maxF(test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        _, res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        mae.update(res, gt)
        maxF.update(res, gt)

        misc.imsave(save_path+name, res)
    print('dataset: {} MAE: {:.4f} maxF: {:.4f}'.format(dataset, mae.show(), maxF.show()))
