import os
import scipy.io
import numpy as np
import logging
import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
#from skimage.measure import compare_psnr, compare_ssim

from data import SonyDataset
from models import UNet


def train(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    # data
    print('===> Loading datasets')
    print(args.input_dir)
    trainset = SonyDataset(args.input_dir, args.gt_dir, args.ps)
    print(len(trainset))
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8, shuffle=True)

    # model
    print('===> Building model')
    model = UNet(3, 3).to(device).train()

    # resume training
    starting_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
        starting_epoch = int(args.resume[-7:-3])
        print('resume at %d epoch' % starting_epoch)

    # loss function
    color_loss = nn.L1Loss()

    print("===> Setting optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    # training
    print("===> Start training")

    for epoch in range(starting_epoch + 1, starting_epoch + args.num_epoch):
        losses = []
        scheduler.step()
        for i, databatch in enumerate(train_loader):

            inputs, gt = databatch
            inputs, gt = inputs.to(device), gt.to(device)

            # back prop
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = color_loss(outputs, gt_patch)
            loss.backward()
            optimizer.step()

            # print statistics
            losses.append(loss.item())
            if (i+1) % args.log_interval == 0:
                print("===> {}\tEpoch[{}]({}/{}): Loss: {:.6f}".format(
                    time.ctime(), epoch, i + 1, len(train_loader), loss.item()))

            if epoch % args.save_freq == 0:
                if not os.path.isdir(os.path.join(args.result_dir, '%04d' % epoch)):
                    os.makedirs(os.path.join(args.result_dir, '%04d' % epoch))

                    gt_patch = gt_patch.cpu().detach().numpy()
                    outputs = outputs.cpu().detach().numpy()

                    temp = np.concatenate((gt_patch[0, :, :, :], outputs[0, :, :, :]), axis=2)
                    scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                        args.result_dir + '%04d/train_%d.jpg' % (epoch, i))

        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f}".format(
                time.ctime(), epoch, np.mean(losses)))

        # save models
        if epoch % args.model_save_freq == 0:
            torch.save(model.state_dict(), args.checkpoint_dir + './model_%d.pth' % epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training")
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--input_dir', type=str, default='/data/hsun/SIDSony/Sony/short/')
    parser.add_argument('--gt_dir', type=str, default='/data/hsun/SIDSony/Sony/long/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--model_save_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, help='continue training')
    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        filename=os.path.join(args.result_dir, 'log.txt'),
                        filemode='w')

    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)
    logging.info("using device %s" % str(args.gpu))
    train(args)
