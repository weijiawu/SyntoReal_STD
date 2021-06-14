import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset.SynthText import SynthText
from dataset.ICDAR15 import ICDAR15
from network.model import EAST
from network.loss import Loss
from network.loss_target import Loss_target
import os
from shapely.geometry import *
import time
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from lib.detect import detect
from evaluate.script import getresult
import argparse
import os
from lib.pseduo_label import generate_pseduo


parser = argparse.ArgumentParser(description='EAST reimplementation')

# Model path
parser.add_argument('--resume', default="./SynthText.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--target_pseudo_negative', default="./pseudo_label_negative/", type=str,
                    help='the pseduo label of target domain')
parser.add_argument('--target_pseudo_positive', default="./pseudo_label_positive/", type=str,
                    help='the positive sample of pseduo label in target domain')
parser.add_argument('--target_image', default="./Image/", type=str,
                    help='the image of target domain')
parser.add_argument('--path_save', default="./Model_save/", type=str,
                    help='save model')


# Training strategy
parser.add_argument('--epoch_iter', default=8000, type = int,
                    help='the max epoch iter')
parser.add_argument('--batch_size', default=6, type = int,
                    help='batch size of training')
# parser.add_argument('--cdua', default=True, type=str2bool,
#                     help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')

args = parser.parse_args()


def train(epoch,  model, optimizer, train_loader_source,train_loader_target):
    model.train()
    scheduler.step()
    epoch_loss = 0
    epoch_time = time.time()

    for i, (source,target) in enumerate(zip(train_loader_source,train_loader_target)):
        start_time = time.time()
        img, gt_score, gt_geo, ignored_map = source
        img_target, gt_score_target, gt_geo_target, ignored_map_target = target

        # target domain training
        img_target, gt_score_target, gt_geo_target, ignored_map_target = \
        img_target.to(device), gt_score_target.to(device), gt_geo_target.to(device), ignored_map_target.to(device)

        (pred_score_target, pred_geo_target),img_class_t = model(img_target)

        geo_loss_target, classify_loss_target,doamin_loss_target = \
        criterion_target(gt_score_target, pred_score_target, gt_geo_target, pred_geo_target, ignored_map_target,img_class_t)


        # source domain training
        img, gt_score, gt_geo, ignored_map  = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)

        (pred_score, pred_geo), img_class_s = model(img)

        geo_loss_source,classify_loss_source , doamin_loss_source  = \
        criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map, img_class_s)
        #
        # if epoch<5:
        #     k=0
        # else:
        #     k=epoch%5 * 0.1
        loss =  (geo_loss_source + classify_loss_source + 0.02*classify_loss_target) + (doamin_loss_target + doamin_loss_source)*0.1
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
            epoch + 1, args.epoch_iter, i + 1, int(1000 / args.batch_size), time.time() - start_time, loss.item()))


    print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(1000 / args.batch_size),time.time() - epoch_time))
    print(time.asctime(time.localtime(time.time())))




if __name__ == '__main__':
    train_img_path = os.path.abspath('./SynthText/')
    train_gt_path = os.path.abspath('./SynthText/gt.mat')

    # source domain
    trainset_ = SynthText(train_img_path, train_gt_path)
    train_loader_source = data.DataLoader(trainset_, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, drop_last=True)

    # target domain
    trainset = ICDAR15(args.target_image, args.target_pseudo_positive,args.target_pseudo_negative)
    train_loader_target = data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, drop_last=True)


    criterion = Loss()
    criterion_target = Loss_target()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_iter // 2], gamma=0.1)

    # generating pseudo-label
    print("loading pretrained model from ",args.resume)
    model.load_state_dict(torch.load(args.resume))
    generate_pseduo(model, args.target_image, args.target_pseudo_positive, args.target_pseudo_negative, device)

    for epoch in range(args.epoch_iter):

        train( epoch, model, optimizer, train_loader_source,train_loader_target)
        # 进行target domain的eval，看看指标。




