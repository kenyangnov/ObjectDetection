#-*- coding:utf-8 -*-

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from config import cfg
from EXTD_64 import build_extd
from layers.modules.multibox_loss import MultiBoxLoss
from factory import dataset_factory, detection_collate
from logger import Logger

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# argparse：解析args
parser = argparse.ArgumentParser(
    description='EXTD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset',
                    default='voc',
                    choices=['voc'],
                    help='Train target')
parser.add_argument('--basenet',
                    # default='vgg16_reducedfc.pth',
                    default='mobileFacenet.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size',
                    default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=True, type=str2bool,
                    help='Use mutil Gpu training')
parser.add_argument('--eval_verbose',
                    default=True, type=str2bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='./weights/{}/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained', default='./weights/mobileFacenet_maxpool_v5_.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

# 设置网络版本，帮助确定文件保存路径
parser.add_argument('net_version')

args = parser.parse_args()

#计算flops
def compute_flops(model, image_size):
  import torch.nn as nn
  flops = 0.
  input_size = image_size
  for m in model.modules():
    if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
      input_size = input_size / 2.
    if isinstance(m, nn.Conv2d):
      if m.groups == 1:
        flop = (input_size[0] / m.stride[0] * input_size[1] / m.stride[1]) * m.kernel_size[0] ** 2 * m.in_channels * m.out_channels
      else:
        flop = (input_size[0] / m.stride[0] * input_size[1] / m.stride[1]) * m.kernel_size[0] ** 2 * ((m.in_channels/m.groups) * (m.out_channels/m.groups) * m.groups)
      flops += flop
      if m.stride[0] == 2: input_size = input_size / 2.

  return flops / 1000000000., flops / 1000000



if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("设置cuda")

save_folder = args.save_folder.format(args.net_version)
if not os.path.exists(save_folder):
    print("创建保存目录")
    os.makedirs(save_folder)


#数据集加载
print('加载数据集...')
#生成训练集和验证集
train_dataset, val_dataset = dataset_factory(args.dataset)
train_loader = data.DataLoader(train_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,
                            collate_fn=detection_collate,
                            pin_memory=True)
val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                            num_workers=args.num_workers,
                            shuffle=False,
                            collate_fn=detection_collate,
                            pin_memory=True)

min_loss = np.inf

start_epoch = 0
extd_net = build_extd('train', cfg.NUM_CLASSES)
net = extd_net
#print(net)

#打印flops
gflops, mflops = compute_flops(net, np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE]))
print('分类模型参数量: %d, flops: %.2f GFLOPS, %.2f MFLOPS, 图片尺寸: %d' % \
      (sum([p.data.nelement() for p in net.parameters()]), gflops, mflops,cfg.INPUT_SIZE))


#加载检查点
if args.resume:
    print('加载检查点, 加载 {}...'.format(args.resume))
    start_epoch = net.load_weights(args.resume)

else:
    try:
        _weights = torch.load(args.pretrained)
        print('加载预训练 base network....')
        net.base.load_state_dict(_weights['state_dict'], strict=False)
    except:
        print('初始化 base network....')
        net.base.apply(net.weights_init)

if args.cuda:
    net = net.cuda()
    cudnn.benckmark = True

if not args.resume:
    print('初始化权重...')
    #s3fd_net.extras.apply(s3fd_net.weights_init) # used only for s3fd
    extd_net.loc.apply(extd_net.weights_init)
    extd_net.conf.apply(extd_net.weights_init)
    #s3fd_net.head.apply(s3fd_net.weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

# MultiBoxLoss中已经做了负样本挖掘
criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)


print('参数列表:\n')
print(args)

# tensorboard使用
tensor_board_dir = os.path.join('./logs/{}/'.format(args.net_version))
if not os.path.isdir(tensor_board_dir):
    os.mkdir(tensor_board_dir)
logger = Logger(tensor_board_dir)

def train():
    step_index = 0
    iteration = 0
    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.cuda:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda())
                            for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c # stress more on loss_l
            loss_add = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss_add.item()

            #计算 tensorboard log 信息
            if iteration % 10 == 0:
                logger.scalar_summary('train_loss_classification', loss_c.item(), iteration, scope='c')
                logger.scalar_summary('train_loss_regression', loss_l.item(), iteration, scope='r')
                logger.scalar_summary('train_total_loss', loss_add.item(), iteration, scope='t')

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                print("[epoch:{}][iter:{}][lr:{:.5f}] loss_class {:.8f} - loss_reg {:.8f} - total {:.8f}".format(
                    epoch, iteration, args.lr, loss_c.item(), loss_l.item(), tloss
                ))

            iteration += 1

        val(epoch)
        if iteration == cfg.MAX_STEPS:
            break


# 每一轮epoch结束后进行验证
def val(epoch):
    net.eval()
    loc_loss = 0
    conf_loss = 0
    step = 0

    with torch.no_grad():
        t1 = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.cuda:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [ann.cuda() for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            out = net(images)
            loss_l, loss_c = criterion(out, targets)

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            step += 1

        tloss = (loc_loss + 3 * conf_loss) / step
        t2 = time.time()
        print('Timer: %.4f' % (t2 - t1))
        print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

        global min_loss
	#保存最优结果
        if tloss < min_loss:
            print('保存最优结果,epoch', epoch)
            file = 'extd_{}.pth'.format(args.dataset)
            torch.save(extd_net.state_dict(), os.path.join(
                save_folder, file))
            min_loss = tloss

        states = {
            'epoch': epoch,
            'weight': extd_net.state_dict(),
        }
	#保存检查点（覆盖）
        print('保存检查点,epoch',epoch)
        file = 'extd_{}_checkpoint.pth'.format(args.dataset)
        torch.save(states, os.path.join(save_folder, file))

        #每5个epoch保存一次（不覆盖）
        if(epoch%5==0):
            print('保存检查点,epoch',epoch)
            file = 'epoch_{}.pth'.format(epoch)
            torch.save(states, os.path.join(save_folder, file))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()

