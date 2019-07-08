# -*- coding:utf-8 -*-

"""a module to read the HTK format file"""

import numpy as np
import struct
import re

# HTK文件结构：
# 帧数：4字节（第0-第3字节）
# 采样周期：4字节（第4-第7字节）
# 每一帧的字节数：2字节（第8-第9字节）
# 参数类型：2字节（第10-第11字节）
# 数据：N字节（第12字节开始-文件结尾）


class HTKfile(object):

    #
    def __init__(self, path):
        self.__start_frame = 0
        self.__end_frame = 0
        self.__new_path = ''

        if path[-1] == ']':  # 判断输入路径末尾有没有指定帧序号的部分 eg: [2, 56]
            temp_value = re.split(r'[\[,\s\]]', path)
            self.__new_path = temp_value[0]
            self.__start_frame = int(temp_value[-3])
            self.__end_frame = int(temp_value[-2])
        else:
            self.__new_path = path

        self.__input = open(self.__new_path, 'rb')
        #  HTK的数据存储方式是大端存储，需要进行大端到小端的转换
        self.__frame_num = struct.unpack('>I', self.__input.read(4))[0]           # 帧数
        self.__sample_period = struct.unpack('>I', self.__input.read(4))[0]       # 采样周期
        self.__bytes_of_one_frame = struct.unpack('>H', self.__input.read(2))[0]  # 每一帧的字节数
        self.__feature_dim = self.__bytes_of_one_frame // 4                        # dimension of feature
        self.__sample_kind = struct.unpack('>h', self.__input.read(2))[0]         # 参数类型
        temp_value_2 = re.split(r'[/.]', path)
        self.__file_name = temp_value_2[-2]

        if self.__end_frame == 0 or self.__end_frame > self.__frame_num:
            self.__end_frame = self.__frame_num  # 如果尾帧数据在之前未被更改，则其值为帧总数

    def read_data(self):
        curr_data = struct.unpack('>'+'f'*self.__frame_num*self.__feature_dim, self.__input.read(self.__frame_num*self.__bytes_of_one_frame))
        data = np.array(curr_data, dtype= 'float32')
        data = data.reshape(self.__frame_num, self.__feature_dim)

        return data[self.__start_frame: self.__end_frame]

    # 下方是在外界获取变量值的函数
    def get_start_frame(self):
        return self.__start_frame

    def get_end_frame(self):
        return self.__end_frame

    def get_frame_num(self):
        return (self.__end_frame - self.__start_frame)

    def get_sample_period(self):
        return self.__sample_period

    def get_bytes_of_one_frame(self):
        return self.__bytes_of_one_frame

    def get_file_name(self):
        return self.__file_name

    def get_feature_dim(self):
        return self.__feature_dim

    def get_state_label(self):
        return self.__state_label
