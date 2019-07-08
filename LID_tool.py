# -*- coding:utf-8 -*-
# --------------------------------------------------------- #
#  python  : 3.6 version                                    #
#  cuda    : Toolkit 9.1                                    #
#  pytorch : 0.4.1                                          #
# --------------------------------------------------------- #

import os
import time
import codecs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s[%(levelname)s] ---- %(message)s',
                    )

import torch
import torch.utils.data as Data

from dataloader import get_samples, get_data, TorchDataSet
from network import LanNet

## ======================================
# 配置文件和参数
# 数据列表
train_list = "./dataset/train.txt"
dev_list   = "./dataset/dev.txt"
test_list   = "./dataset/test.txt"

# 基本配置参数
use_cuda = False 
if use_cuda:
    device = torch.device("cuda:0")

# 保存模型地址
model_dir = "./models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
    
# 网络参数
dimension = 40
language_nums = 6
learning_rate = 0.1
batch_size = 64
chunk_num = 10
train_iteration = 10
display_fre = 10
half = 4

# 构建数据迭代器
train_dataset = TorchDataSet(train_list, batch_size, chunk_num, dimension)
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
test_dataset = TorchDataSet(test_list, batch_size, chunk_num, dimension)
logging.info('finish reading all train data')

# 设计网络优化器
train_module = LanNet(input_dim=dimension, hidden_dim=32, bn_dim=30, output_dim=language_nums)
print(train_module)

optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)

# 将模型放入GPU中
if use_cuda:
    train_module = train_module.to(device)

# 模型训练
for epoch in range(train_iteration):
    if epoch >= half:
        learning_rate /= 2.
        optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)

    train_dataset.reset()
    train_module.train()
    epoch_tic = time.time()
    train_loss = 0.
    train_acc = 0.

    sum_batch_size = 0
    curr_batch_size = 0
    curr_batch_acc = 0
    tic = time.time()
    for step, (batch_x, batch_y) in enumerate(train_dataset): 
        batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
        batch_frames = batch_y[:,1].contiguous().view(-1, 1)

        max_batch_frames = int(max(batch_frames).item())
        batch_train_data = batch_x[:, :max_batch_frames, :]

        step_batch_size = batch_target.size(0)
        batch_mask = torch.zeros(step_batch_size, max_batch_frames)
        for ii in range(step_batch_size):
            frames = int(batch_frames[ii].item())
            batch_mask[ii, :frames] = 1.

        # 将数据放入GPU中
        if use_cuda:
            batch_train_data = batch_train_data.to(device)
            batch_mask       = batch_mask.to(device)
            batch_target     = batch_target.to(device)

        acc, samples, loss = train_module(batch_train_data, batch_mask, batch_target)
        
        backward_loss = loss
        optimizer.zero_grad()
        backward_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc
        curr_batch_acc += acc
        sum_batch_size += samples
        curr_batch_size += samples
        if (step+1) % display_fre == 0:
            toc = time.time()
            step_time = toc-tic
            tic = time.time()
            logging.info('Epoch:%d, Batch:%d, acc:%.6f, loss:%.6f, cost time: %.6fs', epoch, step+1, curr_batch_acc/curr_batch_size, loss.item(), step_time)
            curr_batch_acc = 0.
            curr_batch_size = 0
 
    # 模型存储
    modelfile = '%s/model%d.model'%(model_dir, epoch)
    torch.save(train_module.state_dict(), modelfile)
    epoch_toc = time.time()
    epoch_time = epoch_toc-epoch_tic
    logging.info('Epoch:%d, train-acc:%.6f, train-loss:%.6f, cost time: %.6fs', epoch, train_acc/sum_batch_size, train_loss/sum_batch_size, epoch_time)

    # 模型验证
    train_module.eval()
    epoch_tic = time.time()
    dev_loss = 0.
    dev_acc = 0.
    dev_batch_num = 0 

    for step, (batch_x, batch_y) in enumerate(dev_dataset): 
        batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
        batch_frames = batch_y[:,1].contiguous().view(-1, 1)

        max_batch_frames = int(max(batch_frames).item())
        batch_dev_data = batch_x[:, :max_batch_frames, :]

        step_batch_size = batch_target.size(0)
        batch_mask = torch.zeros(step_batch_size, max_batch_frames)
        for ii in range(step_batch_size):
            frames = int(batch_frames[ii].item())
            batch_mask[ii, :frames] = 1.

        # 将数据放入GPU中
        if use_cuda:
            batch_dev_data   = batch_dev_data.to(device)
            batch_mask       = batch_mask.to(device)
            batch_target     = batch_target.to(device)
            
        with torch.no_grad():
            acc, samples, loss = train_module(batch_dev_data, batch_mask, batch_target)
        
        dev_loss += loss.item()
        dev_acc += acc
        dev_batch_num += samples
    
    epoch_toc = time.time()
    epoch_time = epoch_toc-epoch_tic
    logging.info('Epoch:%d, dev-acc:%.6f, dev-loss:%.6f, cost time: %.6fs\n', epoch, dev_acc/dev_batch_num, dev_loss/dev_batch_num, epoch_time)

#模型测试
train_module.eval()
epoch_tic = time.time()
test_acc = 0.
test_batch_num = 0 

for step, (batch_x, batch_y) in enumerate(test_dataset): 
    batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
    batch_frames = batch_y[:,1].contiguous().view(-1, 1)

    max_batch_frames = int(max(batch_frames).item())
    batch_dev_data = batch_x[:, :max_batch_frames, :]

    step_batch_size = batch_target.size(0)
    batch_mask = torch.zeros(step_batch_size, max_batch_frames)
    for ii in range(step_batch_size):
        frames = int(batch_frames[ii].item())
        batch_mask[ii, :frames] = 1.

    # 将数据放入GPU中
    if use_cuda:
        batch_dev_data   = batch_dev_data.to(device)
        batch_mask       = batch_mask.to(device)
        batch_target     = batch_target.to(device)
        
    with torch.no_grad():
        acc, samples, loss = train_module(batch_dev_data, batch_mask, batch_target)
    
    test_acc += acc
    test_batch_num += samples

epoch_toc = time.time()
epoch_time = epoch_toc-epoch_tic
logging.info('Epoch:%d, test-acc:%.6f, cost time: %.6fs\n', train_iteration, test_acc/test_batch_num,  epoch_time)

