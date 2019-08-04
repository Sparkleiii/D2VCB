from torch.utils.data import Dataset
import torch
import load_data as ld
import numpy as np


k = 3
stride = 1
print("kmer", k)
print("stride", stride)


class MyDataSet(Dataset):
    def __init__(self, input, label):
        self.input_seq = input
        self.output = label

    def __getitem__(self, index):
        input_seq_origin = self.input_seq[index]  # 按照索引迭代读取内容
        # Complementary_sequence = ld.getComplementary(input_seq_origin)
        input_seq_origin = input_seq_origin + input_seq_origin[::-1] + Complementary_sequence + Complementary_sequence[::-1]
        input_seq = np.array(ld.k_mer_stride(input_seq_origin, k, stride)).T
        input_seq = torch.from_numpy(input_seq).type(torch.FloatTensor).cuda()
        output_seq = self.output[index]
        output_seq = torch.Tensor([output_seq]).cuda()
        return input_seq, output_seq  # 直接输出输入序列和输出序列

    def __len__(self):
        return len(self.input_seq)  # 返回的是样本集的大小，样本的个数
