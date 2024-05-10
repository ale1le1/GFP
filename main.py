import os
import torch
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
parser = argparse.ArgumentParser(description="")
parser.add_argument('--train', type=bool, default=True)    #default:默认值
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--window_size', type=int, default=12) #滑动窗口
parser.add_argument('--horizon', type=int, default=1)#单步
parser.add_argument('--train_length', type=float, default=12)#训练数据长度
parser.add_argument('--valid_length', type=float, default=2)#验证数据长度
parser.add_argument('--test_length', type=float, default=1)#测试数据长度
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score')#归一化方法
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)


args = parser.parse_args() #提取参数信息
print(f'Training configs: {args}')
data_file = os.path.join('dataset', args.dataset + '.csv')
result_train_file = os.path.join('output', args.dataset, 'train') #训练结果目录
result_test_file = os.path.join('output', args.dataset, 'test')   #测试结果目录
if not os.path.exists(result_train_file): #如果文件夹未存在，测新建文件夹
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
data = pd.read_csv(data_file).values #读取数据集信息
data = data[:5951, 1:]
#print(data.shape)
# split data
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]#[0:训练集的末端]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]
torch.manual_seed(0)#随机数种子

if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _,normalize_statistic = train(train_data, valid_data, args, result_train_file)#当函数有两个及两个以上的返回值时可以用_,来选择返回其中某个值
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')