#!/usr/bin/python3
#coding=utf-8

import sys
#sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from warnings import filterwarnings
filterwarnings('ignore')
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from lib import dataset
import time
from multiprocessing import Process
from saliency_toolbox import calculate_measures
import warnings
warnings.filterwarnings('ignore')

TAG = "FMFNet"
SAVE_PATH = TAG


class Test(object):
    def __init__(self, Dataset, datapath, Network, model_path):
        self.datapath = datapath.split("/")[-1]
        self.datapath2 = datapath
        print(datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(
            self.data,
            batch_size=bs,
            shuffle=False,
            num_workers=4)#,
            #use_shared_memory=False)
        # network
        self.net = Network
        self.net.eval()
        for p in self.net.parameters():
            p.stop_gradient = True
        self.net.load_dict(paddle.load(model_path))

    def read_img(self, path):
        gt_img = self.norm_img(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        gt_img = (gt_img >= 0.5).astype(np.float32)
        return gt_img

    def norm_img(self, im):
        return cv2.normalize(im.astype('float'),
                             None,
                             0.0, 1.0,
                             cv2.NORM_MINMAX)

    def accuracy(self):
        loss_list = []
        with paddle.no_grad():
            mae = 0
            cost_time = 0
            step = 0
            for image, mask, (H, W), maskpath in self.loader:

                out= self.net(image)
                pred = F.sigmoid(out[0])
                k_pred = pred
                for num in range(len(H)):
                    mae_pred = k_pred[num].unsqueeze(0)
                    mae_pred = F.interpolate(
                        mae_pred,
                        size=(
                            H[num].numpy()[0],
                            W[num].numpy()[0]),
                        mode='bilinear',
                        align_corners=True)
                    path = self.datapath2 + '/mask/' + maskpath[num]
                    mae_mask = paddle.to_tensor(self.read_img(path))
                    mae += (mae_pred[0][0] - mae_mask).abs().mean()
                    step += 1

            return mae.numpy()[0] / step

    def read_img(self, path):
        gt_img = self.norm_img(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        gt_img = (gt_img >= 0.5).astype(np.float32)
        return gt_img

    def norm_img(self, im):
        return cv2.normalize(im.astype('float'),
                             None,
                             0.0, 1.0,
                             cv2.NORM_MINMAX)

    def show_save_sod(self, e, save_path=None):
        for p in self.net.parameters():
            p.stop_gradient = True
        s = time.time()
        with paddle.no_grad():
            mae, fscores, cnt, number = 0, [], 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            stepss = 0
            for num, (image, mask, (H, W), maskpath) in enumerate(self.loader):
                stepss += 1
                start_time = time.time()
                out = self.net(image)
                pred = F.sigmoid(out[0])

                k_pred = pred
                for num in range(len(H)):
                    mae_pred = k_pred[num].unsqueeze(0)
                    path = e + '/mask/' + maskpath[num] + '.png'
                    mae_mask = paddle.to_tensor(self.read_img(path)).unsqueeze(0).unsqueeze(0)
                    mae_pred = F.interpolate(mae_pred, size=mae_mask.shape[2:], mode='bilinear', align_corners=True)
                    mae += (mae_pred - mae_mask).abs().mean()

                    if save_path:
                        save_paths = os.path.join(save_path, self.cfg.datapath.split('/')[-1])
                        if not os.path.exists(save_paths):
                            os.makedirs(save_paths)
                        mae_pred = mae_pred[0].transpose((1, 2, 0)) * 255
                        cv2.imwrite(save_paths + '/' + maskpath[num] + '.png', mae_pred.cpu().numpy())

                    cnt += 1
                end_time = time.time()
                cost_time += end_time - start_time
        e = time.time()
       # print(e - s)
        msg = '%s MAE=%.6f' % (
        self.datapath, mae / cnt)
        print(msg)
        return "%.4f" % float(mae / cnt)

    def fps(self, e):

        self.loader = DataLoader(
            self.data,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            use_shared_memory=False)

        pics = 0
        with paddle.no_grad():
            start_time = time.time()
            for num, (image, mask, (H, W), maskpath) in enumerate(self.loader, start=1):
                out = self.net(image)
                pred = F.sigmoid(out[0])
                pics += 1
            end_time = time.time()
        print(e, "fps:", len(self.loader)/(end_time-start_time))


def test_socre(path, save_root=None):

    if save_root:
        if not os.path.exists(save_root):
            os.mkdir(save_root)
    sms_dir = [
        path + '/DUTS-TE/',
        path + '/ECSSD/',
        path + 'HKU-IS/',
        path + '/DUT-OMRON/',
        path + '/PASCALS/'
    ]
    gts_dir = [
        'work/DUTS-TE/mask'
        'work/ECSSD/mask',
        'work/HKU-IS',
        'work/DUT-OMRON/mask',
        'work/PASCALS/mask',
    ]
    measures = ['Max-F', 'MAE', 'E-measure', 'S-measure', 'Wgt-F']
    for i in range(len(gts_dir)):
        if save_root:
            save = save_root + '/' + sms_dir[i].split('/')[-2]
            print(save)
        else:
            save=None
        res = calculate_measures(gts_dir[i], sms_dir[i], measures, save=save)
        print(path, gts_dir[i].split('/')[-3], 'MAE:', res['MAE'], 'Fm:', res['Mean-F'], 'E-measure:',
              res['E-measure'], 'S-measure', res['S-measure'], 'Wgt-F', res['Wgt-F'])


def mutil_test_score(path, save_root=None):
    if save_root:
        if not os.path.exists(save_root):
            os.makedirs(save_root)
    sms_dir = [
        path + '/DUTS-TE/',
        path + '/ECSSD/',
        path + '/HKU-IS/',
        path + '/DUT-OMRON/',
        path + '/PASCALS/'
    ]
    gts_dir = [
        '/home/aistudio/work/DUTS-TE/mask/',
        '/home/aistudio/work/ECSSD/mask/',
        '/home/aistudio/work/HKU-IS/mask/',
        '/home/aistudio/work/DUT-OMRON/mask/',
        '/home/aistudio/work/PASCALS/mask/'
    ]
    measures = ['Max-F', 'MAE', 'E-measure', 'S-measure', 'Wgt-F']
    if save_root is not None:
        saves = [save_root + '/' + sms_dir[i].split('/')[-2] for i in range(len(sms_dir))]
    else:
        saves = [None] * len(sms_dir)

    processes = [Process(target=sing_score, args=(path, gts_dir[i], sms_dir[i], measures, saves[i]), ) for i in range(len(sms_dir))]
    [p.start() for p in processes]


def sing_score(path, gt, pre, measures, save=None):
    print(gt)
    print(pre)
    res = calculate_measures(gt, pre, measures, save=save)
    print(path, gt.split('/')[-3], 'MAE:', res['MAE'], 'Fm:', res['Mean-F'], 'E-measure:',
          res['E-measure'], 'S-measure', res['S-measure'], 'Wgt-F', res['Wgt-F'])


def select_best_model(model, model_list, save=None):

    DATASETS = [
        '/home/aistudio/work/PASCALS',
        '/home/aistudio/work/ECSSD',
        '/home/aistudio/work/DUT-OMRON',
        '/home/aistudio/work/HKU-IS',
        '/home/aistudio/work/DUTS-TE',
    ]
    all_res = {}
    for model_path in model_list:
        print(model_path)
        results = {}
        for e in DATASETS:
            t = Test(dataset, e, model, model_path)
            res = t.show_save_sod(e=e, save_path=save)
            results[e.split('/')[-1]] = res
        print(results)
        print('\n')


if __name__=='__main__':
    import os
    from net import R2Net
    bs = 128

    model_list1 = [
        "R2Net.pdparams"
    ]
    select_best_model(R2Net(1), model_list1, save='R2Net_Maps')
    mutil_test_score('R2Net_Maps', 'R2Net_Scores')








