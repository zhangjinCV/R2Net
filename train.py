import datetime
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from lib import dataset
import numpy as np
import cv2
import argparse
import os
import random


def config():
    parser = argparse.ArgumentParser(description='train params')
    parser.add_argument('--Min_LR', default=0.000032, help='min lr')
    parser.add_argument('--Max_LR', default=0.032)
    parser.add_argument('--top_epoch', default=40)
    parser.add_argument('--epoch', default=80)
    parser.add_argument('--mode_path', default=False, help='where your pretrained model')
    parser.add_argument('--train_bs', default=64)
    parser.add_argument('--show_step', default=20)
    parser.add_argument('--train_dataset', default=r'work/DUTS-TR')
    parser.add_argument('--save_path', default='weight/R2Net_Weight')
    parser.add_argument('--save_iter', default=1, help=r'every iter to save model')
    cag = parser.parse_args()
    return cag


cag = config()


def lr_decay(steps):
    mum_step = cag.top_epoch * (10553 / cag.train_bs + 1)
    min_lr = cag.Min_LR
    max_lr = cag.Max_LR
    total_steps = cag.epoch * (10553 / cag.train_bs + 1)
    if steps < mum_step:
        lr = min_lr + abs(max_lr - min_lr) / (mum_step) * steps
    else:
        lr = max_lr - abs(max_lr - min_lr) / (total_steps - mum_step + 1) * (steps - mum_step)
    return lr


def structure_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = F.sigmoid(pred)

    inter = (pred * mask).sum(axis=(2, 3))
    union = (pred + mask).sum(axis=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    iou = iou.mean()
    return bce + iou


def boundary_loss(pred, mask):
    pred = F.sigmoid(pred)
    n, c = pred.shape[0], pred.shape[1]
    mask_boundary = paddle.nn.functional.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    mask_boundary -= 1 - mask

    pred_boundary = paddle.nn.functional.max_pool2d(1 - pred, kernel_size=3, stride=1, padding=1)
    pred_boundary -= 1 - pred

    mask_boundary = paddle.nn.functional.max_pool2d(mask_boundary, kernel_size=5, stride=1, padding=2)
    pred_boundary = paddle.nn.functional.max_pool2d(pred_boundary, kernel_size=5, stride=1, padding=2)

    mask_boundary = paddle.reshape(mask_boundary, shape=(n, c, -1))
    pred_boundary = paddle.reshape(pred_boundary, shape=(n, c, -1))
    P = paddle.sum(pred_boundary * mask_boundary, axis=2) / (paddle.sum(pred_boundary, axis=2) + 1e-7)
    R = paddle.sum(pred_boundary * mask_boundary, axis=2) / (paddle.sum(mask_boundary, axis=2) + 1e-7)
    B = 2 * P * R / (P + R + 1e-7)
    loss = paddle.mean(1 - B)
    return loss


def train(Dataset, Network, savepath):
    # dataset
    cag.min_mae = 10
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    cfg = Dataset.Config(
        snapshot=cag.mode_path, datapath=cag.train_dataset, savepath=savepath,
        mode='train', batch=cag.train_bs, lr=cag.Max_LR, momen=0.9, decay=5e-4, epoch=cag.epoch
    )

    data = Dataset.Data(cfg)
    loader = DataLoader(
        data,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=8,
        collate_fn=data.collate,
        use_shared_memory=False,
    )

    # network
    net = Network(cfg)
    net.train()

    # params
    total_params = sum(p.numel() for p in net.parameters())
    print('total params : ', total_params)

    # optimizer
    optimizer = paddle.optimizer.Momentum(parameters=net.parameters(), learning_rate=cag.Max_LR, momentum=cfg.momen,
                                          weight_decay=cfg.decay)
    global_step = 0
    # training
    all_losses = []
    all_lr = []
    all_metric = []
    for epoch in range(0, cfg.epoch):
        start = datetime.datetime.now()
        loss_list = []
        for batch_idx, (image, mask) in enumerate(loader, start=1):
            lr = lr_decay(global_step)
            optimizer.clear_grad()
            optimizer.set_lr(lr)
            all_lr.append(optimizer.get_lr())

            global_step += 1
            out2, out3, out4, out5 = net(image)
            loss1 = structure_loss(out2, mask) + boundary_loss(out2, mask) # structure_pooling_loss
            loss2 = structure_loss(out3, mask) + boundary_loss(out3, mask)
            loss3 = structure_loss(out4, mask) + boundary_loss(out4, mask)
            loss4 = structure_loss(out5, mask) + boundary_loss(out5, mask)
            loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8
            loss.backward()
            loss_list.append(loss.numpy()[0])
            all_losses.append(loss.numpy()[0])
            optimizer.step()

            if batch_idx % cag.show_step == 0:
                msg = '%s | step:%d/%d/%d (%.2f%%) | lr=%.6f |  loss=%.6f | loss1=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f| loss6=%.6f| %s ' % (
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx, epoch + 1, cfg.epoch,
                    batch_idx / (10553 / cag.train_bs) * 100, optimizer.get_lr(), loss.item()
                    , loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss1.item(), loss1.item(), image.shape)
                print(msg)

        if epoch > cag.epoch / 3 * 2:
            paddle.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1) + '.pdparams')

        end = datetime.datetime.now()
        spend = int((end - start).seconds)
        mins = spend // 60
        secon = spend % 60
        loss_list = '%.5f' % np.mean(loss_list)
        print(f'this epoch spend {mins} m {secon} s and the average loss is {loss_list}', '\n')


if __name__ == '__main__':
    from net import R2Net

    train(dataset, R2Net, 'weight')
