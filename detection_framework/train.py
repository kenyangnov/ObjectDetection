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
parser.add_argument('--save_folder',
                    default='./weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained', default='./weights/mobileFacenet_maxpool_v5_.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

args = parser.parse_args()

print('参数设置如下:\n', args)

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


if not os.path.exists(args.save_folder):
    print("创建权重保存目录")
    os.makedirs(args.save_folder)


#数据集加载(目前仅支持VOC数据集)
print('加载数据集...')
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

#创建网络
extd_net = build_extd('train', cfg.NUM_CLASSES)
net = extd_net
if args.cuda:
    net = net.cuda()
    cudnn.benckmark = True
#print(net)

#打印flops
gflops, mflops = compute_flops(net, np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE]))
print('分类模型参数量: %d, flops: %.2f GFLOPS, %.2f MFLOPS, 图片尺寸: %d' % \
      (sum([p.data.nelement() for p in net.parameters()]), gflops, mflops,cfg.INPUT_SIZE))


#初始化权重
if args.resume:
    #有检查点文件
    print('加载检查点, 加载 {}...'.format(args.resume))
    start_epoch = net.load_weights(args.resume)

else:
    #无检查点文件
    try:
        #有预训练主干网络
        _weights = torch.load(args.pretrained)
        print('加载预训练 base network....')
        net.base.load_state_dict(_weights['state_dict'], strict=False)
    except:
        #无预训练主干网络
        print('初始化 base network....')
        net.base.apply(net.weights_init)

if not args.resume:
    print('初始化权重...')
    #s3fd_net.extras.apply(s3fd_net.weights_init) # used only for s3fd
    extd_net.loc.apply(extd_net.weights_init)
    extd_net.conf.apply(extd_net.weights_init)
    #s3fd_net.head.apply(s3fd_net.weights_init)

#设置优化器和损失函数
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)




# tensorboard使用
tensor_board_dir = os.path.join('./logs', 'tensorboard')
if not os.path.isdir(tensor_board_dir):
    os.mkdir(tensor_board_dir)
logger = Logger(tensor_board_dir)

min_loss = np.inf
start_epoch = 0

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


# 每一轮epoch训练结束后进行验证
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

                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            out = net(images)
            loss_l, loss_c = criterion(out, targets)

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            step += 1

        tloss = (loc_loss + conf_loss) / step
        t2 = time.time()
        print('Timer: %.4f' % (t2 - t1))
        print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

        global min_loss
	    #保存最优结果
        if tloss < min_loss:
            print('e当前最优结果为epoch', epoch,'，已保存。')
            file = 'extd_{}_best.pth'.format(args.dataset)
            torch.save(extd_net.state_dict(), os.path.join(
                args.save_folder, file))
            min_loss = tloss

        states = {
            'epoch': epoch,
            'weight': extd_net.state_dict(),
        }
	    #保存检查点
        print('保存epoch',epoch,'，检查点。')
        file = 'extd_{}_checkpoint.pth'.format(args.dataset)
        torch.save(states, os.path.join(
            args.save_folder, file))


# 调整学习率
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

