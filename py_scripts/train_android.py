########################################################################
# Author(s):    Shubh Gupta, Ashwin Kanhere
# Date:         21 September 2021
# Desc:         Train a DNN to output position corrections for Android
#               measurements
########################################################################
# 导入标准库
import sys, os, csv, datetime
from typing import Dict
# 设置目录路径
parent_directory = os.path.split(os.getcwd())[0] # returns the current working directory of a process
src_directory = os.path.join(parent_directory, 'src')
data_directory = os.path.join(parent_directory, 'data') # deep_gnss/data
ephemeris_data_directory = os.path.join(data_directory, 'ephemeris')
sys.path.insert(0, src_directory)
# 导入第三方库和自定义模块
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
# 导入自定义模块
import gnss_lib.coordinates as coord
import gnss_lib.read_nmea as nmea
import gnss_lib.utils as utils 
import gnss_lib.solve_pos as solve_pos
from correction_network.android_dataset import Android_GNSS_Dataset
from correction_network.networks import Net_Snapshot, DeepSetModel

# 对批处理数据进行整理和填充
def collate_feat(batch):
    # 按照features的长度从大到小排序
    sorted_batch = sorted(batch, key=lambda x: x['features'].shape[0], reverse=True)
    # 提取feature
    features = [x['features'] for x in sorted_batch]
    # 填充，使每个序列长度相同
    features_padded = torch.nn.utils.rnn.pad_sequence(features) #pad_sequence stacks a list of Tensors along a new dimension, and pads them to equal length.
    # 获取填充后张量维度
    L, N, dim = features_padded.size()
    #全零掩码矩阵，N为序列数量，L为填充后最大长度
    pad_mask = np.zeros((N, L))
    #填充掩码矩阵，位置为填充值时掩码矩阵对应位置为true
    for i, x in enumerate(features):
        pad_mask[i, len(x):] = 1
    pad_mask = torch.Tensor(pad_mask).bool()

    # 提取true_correction和guess
    correction = torch.Tensor([x['true_correction'] for x in sorted_batch])
    guess = torch.Tensor([x['guess'] for x in sorted_batch])

    #构造返回值
    retval = {
            'features': features_padded,
            'true_correction': correction,
            'guess': guess
        }
    return retval, pad_mask


def test_eval(val_loader, net, loss_func):
    # VALIDATION EVALUATION
    stats_val = []  # 存储每个批次的误差
    loss_val = 0  # 累加每个批次的损失值
    generator = iter(val_loader)
    # tqdm库显示进度条，遍历100批次数据
    for i in tqdm(range(100), desc='test', leave=False):
        try:
            sample_batched = next(generator)
        except StopIteration:
            generator = iter(val_loader)
            # 从生成器中获取一个批次的数据，如果数据耗尽，则重新创建生成器并获取新的批次数据
            sample_batched = next(generator)
            # 从批次数据中获取样本和填充掩码
        _sample_batched, pad_mask = sample_batched
    #         feat_pack = torch.nn.utils.rnn.pack_padded_sequence(_sample_batched['features'], x_lengths)
        # 转换为浮点数并转移到GPU
        x = _sample_batched['features'].float().cuda()
        y = _sample_batched['true_correction'].float().cuda()
        pad_mask = pad_mask.cuda()
        # 模型预测和损失计算
        pred_correction = net(x, pad_mask=pad_mask)  # 使用模型进行预测，得到预测修正值
        loss = loss_func(pred_correction, y)  # 计算预测值和真实值之间的损失
        loss_val += loss
        # 计算误差并转移到CPU，存储在stats_val中
        stats_val.append((y-pred_correction).cpu().detach().numpy())
        # np.mean计算平均绝对误差，loss_val/len(stats_val)计算平均损失
    return np.mean(np.abs(np.array(stats_val)), axis=0), loss_val/len(stats_val)

# 利用hydra调用config文件里的conf
@hydra.main(config_path="../config", config_name="train_android_conf")
def main(config: DictConfig) -> None:
    # 存储数据值配置
    data_config = {
    "root": data_directory,
    "raw_data_dir" : config.raw_data_dir,
    "data_dir": config.data_dir,
    # "initialization_dir" : "initialization_data",
    # "info_path": "data_info.csv",
    "max_open_files": config.max_open_files,
    "guess_range": [config.pos_range_xy, config.pos_range_xy, config.pos_range_z, config.clk_range, config.vel_range_xy, config.vel_range_xy, config.vel_range_z, config.clkd_range],
    "history": config.history,
    "seed": config.seed,
    "chunk_size": config.chunk_size,
    "max_sats": config.max_sats,
    "bias_fname": config.bias_fname,
    }
    
    print('Initializing dataset')
    # 初始化数据集，使用Android_GNSS_Dataset类读取和处理数据
    dataset = Android_GNSS_Dataset(data_config)

    # 划分数据集，按frac划分train和val数据集；添加加载器，使用collate_feat函数处理批次数据
    train_set, val_set = torch.utils.data.random_split(dataset, [int(config.frac*len(dataset)), len(dataset) - int(config.frac*len(dataset))])
    dataloader = DataLoader(train_set, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers, collate_fn=collate_feat)
    val_loader = DataLoader(val_set, batch_size=1, 
                            shuffle=False, num_workers=0, collate_fn=collate_feat)
    # 初始化模型，根据model_name选择初始化模型
    print('Initializing network: ', config.model_name)
    if config.model_name == "set_transformer":
        net = Net_Snapshot(train_set[0]['features'].size()[1], 1, len(train_set[0]['true_correction']))     # define the network
    elif config.model_name == "deepsets":
        net = DeepSetModel(train_set[0]['features'].size()[1], len(train_set[0]['true_correction']))
    else:
        raise ValueError('This model is not supported yet!')

    # resume不为0，从指定路径夹子模型权重恢复训练
    if not config.resume==0:
        net.load_state_dict(torch.load(os.path.join(data_directory, 'weights', config.resume)))
        print("Resumed: ", config.resume)

    # 设置优化器和损失函数
    net.cuda()  # 转移到GPU上

    optimizer = torch.optim.Adam(net.parameters(), config.learning_rate)  # 使用Adam优化器，设置学习率
    loss_func = torch.nn.MSELoss()  # 均方误差损失函数
    count = 0
    fname = "android_" + config.prefix + "_"+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if config.writer:
        writer = SummaryWriter(os.path.join(data_directory, 'runs', fname))

    # 训练和验证循环
    min_acc = 1000000
    for epoch in range(config.N_train_epochs):
        # TRAIN Phase
        net.train()
        for i, sample_batched in enumerate(dataloader):
            _sample_batched, pad_mask = sample_batched
            
            x = _sample_batched['features'].float().cuda()
            y = _sample_batched['true_correction'].float().cuda()
            pad_mask = pad_mask.cuda()
            pred_correction = net(x, pad_mask=pad_mask)
            loss = loss_func(pred_correction, y)
            if config.writer:
                writer.add_scalar("Loss/train", loss, count)
                
            count += 1    
            
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients后向传播
            optimizer.step()        # apply gradients
        # TEST Phase
        net.eval()
        mean_acc, test_loss = test_eval(val_loader, net, loss_func)
        if config.writer:
            writer.add_scalar("Loss/test", test_loss, epoch)
        for j in range(len(mean_acc[0])):
            if config.writer:
                writer.add_scalar("Metrics/Acc_"+str(j), mean_acc[0, j], epoch)
        if np.sum(mean_acc) < min_acc:
            min_acc = np.sum(mean_acc)
            torch.save(net.state_dict(), os.path.join(data_directory, 'weights', fname))
        print('Training done for ', epoch, 'Loss is',test_loss.cpu().detach().numpy())

if __name__=="__main__":
    main()
