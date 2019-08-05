from densebox.DenseBoxDataset import DenseBoxDataset
from densebox.DenseBoxDataset import DenseBoxDatasetOnline
from densebox.DenseBox import DenseBox


import os
import torch
import torchvision
from torch import nn
import numpy as np

import time

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

root = "/media/wl/000675B10007A33A/DatasetRepo/haierpatch/"

#调整学习率
def adjust_LR(optimizer,
              epoch):
    """

    :param optimizer:
    :param epoch:
    :return:
    """
    lr = 1e-9
    if epoch < 5:
        lr = 1e-9
    elif epoch >= 5 and epoch < 10:
        lr = 2e-9
    elif epoch >= 10 and epoch < 15:
        lr = 4e-9
    else:
        lr = 1e-9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def mask_by_sel(loss_mask,
                pos_indices,
                neg_indices):
    """
    cpu side calculation
    :param loss_mask:
    :param pos_indices: N×4dim
    :param neg_indices:
    :return:
    """

    assert loss_mask.size() == torch.Size([loss_mask.size(0), 1, 120, 120])

    # print('=> before fill loss mask:%d non_zeros.' % torch.nonzero(loss_mask).size(0))

    for pos_idx in pos_indices:
        loss_mask[pos_idx[0], pos_idx[1], pos_idx[2], pos_idx[3]] = 1.0
    
    for row in range(neg_indices.size(0)):
        for col in range(neg_indices.size(1)):
            idx = int(neg_indices[row][col])

            if idx < 0 or idx >= 14400: #120*120
                # print('=> idx: ', idx)
                continue

            y = idx // 120
            x = idx % 120

            try:
                # row相当于batch维
                loss_mask[row, 0, y, x] = 1.0
            except Exception as e:
                print(row, y, x)

    # print('=> before fill loss mask:%d non_zeros.' % torch.nonzero(loss_mask).size(0))


#这里的offline的含义是：数据(如bbox,label等数据)转换为张量,在训练前已经处理完成
#训练前所有的数据张量都已加载到内存
def train_offline(root,
                  num_epoch=30,
                  lambda_loc=3.0,
                  base_lr=1e-5,
                  resume=None):
    """
    generating vertices offline(pre-load to main memory)
    :param num_epoch:
    :param lambda_loc:
    :param resume:
    :return:
    """
    train_set = DenseBoxDataset(root)
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    

    # network
    vgg19_pretrain = torchvision.models.vgg19()
    vgg19_pretrain.load_state_dict(torch.load('vgg19.pth'))
    net = DenseBox(vgg19=vgg19_pretrain).to(device)
    print('=> net:\n', net)

    # ---------------- whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)

    # ---------------- loss functions
    # element-wise L2 loss
    loss_func = nn.MSELoss(reduce=False).to(device)

    # optimization function
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=base_lr,
                                momentum=9e-1,
                                weight_decay=5e-8)  # 5e-4

    # net start to train in train mode
    print('\nTraining...')
    net.train()

    for epoch_i in range(num_epoch):
        # adjust learning before this epoch
        lr = adjust_LR(optimizer=optimizer,
                       epoch=epoch_i)
        print('=> learning rate: ', lr)

        for batch_i, (data, label_map, loss_mask) in enumerate(train_loader):
            # ------------- put data to device
            data, label_map = data.to(device), label_map.to(device)

            # loss_mask = loss_mask.unsqueeze(1)

            # ------------- clear gradients
            optimizer.zero_grad()

            # ------------- forward pass
            score_out, loc_out = net.forward(data)

            # ------------- loss calculation with hard negative mining
            score_map_gt = label_map[:, 0]
            score_map_gt = score_map_gt.unsqueeze(1)
            loc_map_gt = label_map[:, 1:]

            positive_indices = torch.nonzero(score_map_gt)
            positive_num = positive_indices.size(0)

            # to keep the ratio of positive and negative sample to 1
            negative_num = int(float(positive_num) / float(data.size(0)) + 0.5)
            score_loss = loss_func(score_out, score_map_gt)

            # loc loss should be masked by label scores and to be summed
            loc_loss = loss_func(loc_out, loc_map_gt)

            # negative smapling... debug
            ones_mask = torch.ones([data.size(0), 1, 120, 120],
                                   dtype=torch.float32).to(device)
            neg_mask = ones_mask - score_map_gt
            negative_score_loss = score_loss * neg_mask
            # print('=> neg pix numner: ', torch.nonzero(negative_score_loss).size(0))

            half_neg_num = int(negative_num * 0.5 + 0.5)
            negative_score_loss = negative_score_loss.view(data.size(0), -1)
            hard_negs, hard_neg_indices = torch.topk(input=negative_score_loss,
                                                     k=half_neg_num,
                                                     dim=1)

            rand_neg_indices = torch.zeros([data.size(0), half_neg_num],
                                           dtype=torch.long).to(device)
            for i in range(data.size(0)):
                indices = np.random.choice(14400,  # 120 * 120
                                           half_neg_num,
                                           replace=False)
                indices = torch.Tensor(indices)
                rand_neg_indices[i] = indices

            # concatenate negative sample ids
            neg_indices = torch.cat((hard_neg_indices, rand_neg_indices), dim=1)

            neg_indices = neg_indices.cpu()
            positive_indices = positive_indices.cpu()

            # fill the loss mask
            mask_by_sel(loss_mask=loss_mask,
                        pos_indices=positive_indices,
                        neg_indices=neg_indices)

            # ------------- calculate final loss
            loss_mask = loss_mask.to(device)

            mask_score_loss = loss_mask * score_loss
            mask_loc_loss = loss_mask * score_map_gt * loc_loss

            loss = torch.sum(mask_score_loss) \
                   + torch.sum(lambda_loc * mask_loc_loss)

            # ------------- back propagation
            loss.backward()
            optimizer.step()

            # ------------- print loss
            iter_count = epoch_i * len(train_loader) + batch_i
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>3d}/{:>3d}'
                      ', total_iter {:>5d} '
                      '| loss {:>5.3f}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              loss.item()))

        # ------------ save checkpoint
        torch.save(net.state_dict(), root+'denseboxtemp.pth')
        print('<= {} saved.\n'.format(root+'densebox.pth'))

    torch.save(net.state_dict(), root+'densebox.pth')
    print('<= {} saved.\n'.format(root+'densebox.pth'))


def collate_fn_customer(batch):
    """
    这个函数的作用是将读取到的batch中的多组数据,融合成整体
    也就是增加一个batch维度
    """
    images = []
    bboxes = []
    for i, data in enumerate(batch):
        # data[0]为img维度
        images.append(data[0])
        # data[1]为bbox维度
        
        bboxes.append(data[1])
    
    #images类型转换:list==>torch.tensor
    images = torch.stack(images)
    batch = (images, bboxes)
    return batch
    

# 初始化score_map
def init_score_map(bboxes, ratio=0.3):
    """
    初始化score_map
    ratio为保留的中心区域的比率
    """
    score_map = torch.zeros([1, 120, 120], dtype=torch.float32)
    #初始化score_map
    for bbox in bboxes:
        #首先转换到120*120坐标空间
        leftup_x = bbox[0] / 4.0
        leftup_y = bbox[1] / 4.0
        rightdown_x = bbox[2] / 4.0
        rightdown_y = bbox[3] / 4.0

        bbox_center_x = float(leftup_x + rightdown_x) * 0.5
        bbox_center_y = float(leftup_y + rightdown_y) * 0.5
        bbox_w = rightdown_x - leftup_x
        bbox_h = rightdown_y - leftup_y

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        score_map[:, org_y: end_y + 1, org_x: end_x + 1] = 1.0
    return score_map

def init_dist_map(bboxes, ratio = 0.3):
    dist_map = torch.zeros([4, 120, 120], dtype=torch.float32)
    dxt_map, dyt_map = dist_map[0], dist_map[1]
    dxb_map, dyb_map = dist_map[2], dist_map[3]
    for bbox in bboxes:
        #首先转换到120*120坐标空间
        leftup_x = bbox[0] / 4.0
        leftup_y = bbox[1] / 4.0
        rightdown_x = bbox[2] / 4.0
        rightdown_y = bbox[3] / 4.0

        bbox_w = rightdown_x - leftup_x
        bbox_h = rightdown_y - leftup_y

        for y in range(dxt_map.size(0)):  # dim H
            for x in range(dxt_map.size(1)):  # dim W
                dist_xt = (float(x) - leftup_x) / bbox_w
                dist_yt = (float(y) - leftup_y) / bbox_h
                dist_xb = (float(x) - rightdown_x) / bbox_w
                dist_yb = (float(y) - rightdown_y) / bbox_h
                # 行和列分别是y和x
                dxt_map[y, x] = dist_xt
                dyt_map[y, x] = dist_yt
                dxb_map[y, x] = dist_xb
                dyb_map[y, x] = dist_yb
    return dist_map

def init_mask_map(bboxes, ratio=0.3):
    mask_map = torch.zeros([1, 120, 120], dtype=torch.float32)
    for bbox in bboxes:
        #首先转换到输出特征图的120*120坐标空间
        leftup_x = bbox[0] / 4.0
        leftup_y = bbox[1] / 4.0
        rightdown_x = bbox[2] / 4.0
        rightdown_y = bbox[3] / 4.0

        bbox_center_x = float(leftup_x + rightdown_x) * 0.5
        bbox_center_y = float(leftup_y + rightdown_y) * 0.5
        bbox_w = rightdown_x - leftup_x
        bbox_h = rightdown_y - leftup_y

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        mask_map[:, org_y: end_y + 1, org_x: end_x + 1] = 1.0
    return mask_map

def train_online(root,
                 num_epoch=1,
                 lambda_cls=1.0,
                 lambda_loc=3.0,
                 base_lr=1e-5,
                 resume=None):
    train_set =  DenseBoxDatasetOnline(root)
    batch_size = 4

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn = collate_fn_customer
                                               )


    # network
    vgg19_pretrain = torchvision.models.vgg19()
    vgg19_pretrain.load_state_dict(torch.load('vgg19.pth'))
    net = DenseBox(vgg19=vgg19_pretrain).to(device)
    print('=> net:\n', net)

    # element-wise L2 loss function
    loss_func = nn.MSELoss(reduce=False).to(device)

    # optimization function
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=base_lr,
                                momentum=9e-1,
                                weight_decay=5e-8)  # 5e-4 or 5e-8
    # net start to train in train mode
    print('\nTraining...')
    net.train()

    print('=> base learning rate: ', base_lr)

    for epoch_i in range(num_epoch):
        # 每一轮训练
        epoch_loss = []

        for batch_idx, batch in enumerate(train_loader):
            #初始化图像,同时放到GPU上
            img = batch[0]
            img = img.to(device)

            bboxes_batch = batch[1]
            #初始化GT_maps,同时将它们放到GPU上
            score_maps = []
            dist_maps = []
            mask_maps = []
            for bboxes_img in bboxes_batch:
                score_map = init_score_map(bboxes=bboxes_img, ratio=1)
                score_maps.append(score_map)
                dist_map = init_dist_map(bboxes=bboxes_img, ratio=1)
                dist_maps.append(dist_map)
                mask_map = score_map.clone()
                mask_maps.append(mask_map)
            cls_maps_gt = torch.stack(score_maps).to(device)
            loc_maps_gt = torch.stack(dist_maps).to(device)
            mask_maps = torch.stack(mask_maps)

            #清空梯度
            optimizer.zero_grad()

            #前向传播
            score_out, loc_out = net.forward(img)

            #分类损失
            cls_loss = loss_func(score_out, cls_maps_gt)
            
            #定位损失
            bbox_loc_loss = loss_func(loc_out, loc_maps_gt)

            """
            负样本挖掘
            """
            pos_indices = torch.nonzero(cls_maps_gt)
            positive_num = pos_indices.size(0)
            #保证正负样本比为1
            #注:img.size(0)即batch_size,这里使用batch_size会在最后不足一个完整batch时报错
            neg_num = int(float(positive_num) / float(img.size(0)) + 0.5)
            ones_mask = torch.ones([img.size(0), 1, 120, 120],
                                   dtype=torch.float32).to(device)
            neg_mask = ones_mask - cls_maps_gt
            neg_cls_loss = cls_loss * neg_mask

            #一半负样本挖掘获得,一半从负样本中随机采样
            half_neg_num = int(neg_num * 0.5 + 0.5)
            neg_cls_loss = neg_cls_loss.view(img.size(0), -1)
            hard_negs, hard_neg_indices = torch.topk(input=neg_cls_loss,
                                                     k=half_neg_num,
                                                     dim=1)
            rand_neg_indices = torch.zeros([img.size(0), half_neg_num],
                                           dtype=torch.long).to(device)
            
            #可改进：这里的随机采样,并不是从负样本中采样,而是全体样本中随机
            for i in range(img.size(0)):
                indices = np.random.choice(14400, #120*120
                                           half_neg_num,
                                           replace=False)
                indices = torch.Tensor(indices)
                rand_neg_indices[i] = indices
            #汇总负样本indices
            neg_indices = torch.cat((hard_neg_indices,
                                     rand_neg_indices),
                                    dim=1)
            neg_indices = neg_indices.cpu()
            pos_indices = pos_indices.cpu()
            
            #更新mask_map,用于确定哪些样本拿来计算损失
            mask_by_sel(loss_mask=mask_maps,
                        pos_indices=pos_indices,
                        neg_indices=neg_indices)
            mask_maps.to(device)

            #计算最终的损失
            mask_cls_loss = mask_maps * cls_loss    #分类损失
            mask_bbox_loc_loss = mask_maps * cls_maps_gt * bbox_loc_loss    #定位损失
            full_loss = lambda_cls * (torch.sum(mask_cls_loss)
                        + lambda_loc * torch.sum(mask_bbox_loc_loss))
            
            # 记录当前epoch当前batch损失
            epoch_loss.append(full_loss.item())

            # 反向传播
            full_loss.backward()
            optimizer.step()

            # 损失的日志信息
            iter_count = epoch_i * len(train_loader) + batch_i
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>3d}/{:>3d}'
                      ', total_iter {:>5d} '
                      '| loss {:>5.3f}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              full_loss.item()))
            
            # 输出目前为止,当前epoch的平均损失
            print('=> epoch %d average loss: %.3f'
                % (epoch_i + 1, sum(epoch_loss) / len(epoch_loss)))

            # 保存checkpoint文件
            torch.save(net.state_dict(), root+'checkpoints/temp.pth')
            print('<= {} saved.'.format(root+'checkpoints/temp.pth'))
            
            # 调整下一轮的学习率
            lr = adjust_LR(optimizer=optimizer,
                        epoch=epoch_i)
            print('=> applying learning rate: ', lr)
 
            break
        break
    torch.save(net.state_dict(), root+'checkpoints/final.pth')
    print('<= {} saved.\n'.format(root+'checkpoints/final.pth'))



train_online(root)

"""
vgg19_pretrain = torchvision.models.vgg19()
vgg19_pretrain.load_state_dict(torch.load('vgg19.pth'))
net = DenseBox(vgg19=vgg19_pretrain).to(device)
net.cuda()
x = np.random.randint(0,255, (4, 3, 480, 480))
x = torch.from_numpy(x).float().to(device)
y = net.forward(x)
print(y[0].size(),y[1].size())
"""


