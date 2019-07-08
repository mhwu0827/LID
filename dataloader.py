# -*- coding:utf-8 -*-

import codecs
import copy
import random

import torch

from lib.HTKfile import HTKfile

def get_samples(list):
    samples = 0
    max_frames = 0
    with codecs.open(list, 'r', 'utf-8') as file_list:
        for line in file_list:
            line = line.strip()  # 去除结尾换行符
            if not line:  # remove the blank line
                continue
            splited_line = line.split()
            htk_feature = splited_line[0]

            htk_file = HTKfile(htk_feature)
            feature_frames = htk_file.get_frame_num()

            max_frames = max(max_frames, feature_frames)
            samples += 1
    file_list.close()
    return samples, max_frames


def get_data(list, samples, max_frames, dimension):
    data = torch.zeros(samples, max_frames, dimension)
    target_frames = torch.zeros(samples, 2)
    name_list = []
    # 存储数据
    line_num = 0
    with codecs.open(list, 'r', 'utf-8') as file_list:
        for line in file_list:
            line = line.strip()  # 去除结尾换行符
            if not line:  # remove the blank line
                continue
            splited_line = line.split()
            htk_feature = splited_line[0]
            target_label = int(str(splited_line[1]))

            htk_file = HTKfile(htk_feature)
            feature_data = htk_file.read_data()
            file_name = htk_file.get_file_name()
            feature_frames = htk_file.get_frame_num()
            
            curr_feature = torch.Tensor(feature_data)
            means = curr_feature.mean(dim=0, keepdim=True)
            curr_feature_norm = curr_feature - means.expand_as(curr_feature)
            data[line_num,:feature_frames,:] = curr_feature_norm
            target_frames[line_num] = torch.Tensor([target_label, feature_frames])
            name_list.append(file_name)

            line_num += 1
    file_list.close()

    return data, target_frames, name_list

class TorchDataSet(object):
    def __init__(self, file_list, batch_size, chunk_num, dimension):
        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num*self._batch_size
        self._dimension = dimension
        self._file_point = codecs.open(file_list, 'r', 'utf-8')
        self._dataset = self._file_point.readlines()
        self._file_point.close()
        random.shuffle(self._dataset)

    def reset(self):
        random.shuffle(self._dataset)
    
    def __iter__(self):
        data_size = len(self._dataset)
        batch_data = []
        target_frames = []
        name_list = []
        max_frames = 0
        for ii in range(data_size):
            line = self._dataset[ii].strip()
            splited_line = line.split()
            htk_feature = splited_line[0]
            target_label = int(str(splited_line[1]))

            htk_file = HTKfile(htk_feature)
            feature_data = htk_file.read_data()
            file_name = htk_file.get_file_name()
            feature_frames = htk_file.get_frame_num()

            if feature_frames > max_frames:
                max_frames = feature_frames
            
            curr_feature = torch.Tensor(feature_data)
            means = curr_feature.mean(dim=0, keepdim=True)
            curr_feature_norm = curr_feature - means.expand_as(curr_feature)
            batch_data.append(curr_feature_norm)
            target_frames.append(torch.Tensor([target_label, feature_frames]))
            name_list.append(file_name)

            if (ii+1) % self._chunck_size == 0:
                chunk_size = len(batch_data)
                idx = 0
                data = torch.zeros(self._batch_size, max_frames, self._dimension)
                target = torch.zeros(self._batch_size, 2)
                for jj in range(chunk_size):
                    curr_data = batch_data[jj]
                    curr_tgt = target_frames[jj]
                    curr_frame = curr_data.size(0)

                    data[idx,:curr_frame,:] = curr_data[:,:]
                    target[idx,:] = curr_tgt[:]
                    idx += 1

                    if idx % self._batch_size == 0:
                        idx = 0
                        yield data, target
                
                max_frames = 0
                batch_data = []
                target_frames = []
                name_list = []
            
            else:
                pass
            

        chunk_size = len(batch_data)
        if chunk_size > self._batch_size: 
            idx = 0
            data = torch.zeros(self._batch_size, max_frames, self._dimension)
            target = torch.zeros(self._batch_size, 2)
            for jj in range(chunk_size):
                curr_data = batch_data[jj]
                curr_tgt = target_frames[jj]
                curr_frame = curr_data.size(0)

                data[idx,:curr_frame,:] = curr_data[:,:]
                target[idx,:] = curr_tgt[:]
                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data, target

