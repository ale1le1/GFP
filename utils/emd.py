from PyEMD import EMD
from utils import emd2d
import pylab as plt
import numpy as np
import pandas as pd
print('1212')
import numpy  as np
s = np.random.random(100)
emd = EMD()
emd2 = emd2d.emd2d()
emd2 = emd2d.emd2d()
# IMF = emd.emd(s,max_imf=2)
# print(IMF.shape)
data =  pd.read_csv("../dataset/ETTm1.csv",usecols=[1,2,3,4,5,6])
print(data.shape)
npdata = data.values
y = npdata[:,1][0:1000]
IMF = emd.emd(y,max_imf=3)
print(IMF.shape)
# IMF2=emd2.emd(npdata)
# print(IMF2.shape)
# y = npdata[:,1]
# y1 = IMF2[0][:,1]
# y2 = IMF2[1][:,1]
# x =np.arange(0,69680,1)
# plt.figure(figsize=(250, 3))
# plt.plot(x, y, label='Y')  # 绘制 y1 数据
# plt.plot(x,y1,color='red',label='Y1')
# plt.plot(x,y2,label='Y2')
N = IMF.shape[0]+1
x =np.arange(0,1000,1)
# Plot results
plt.figure(figsize=(100, 30))
plt.subplot(N,1,1)
plt.plot(x, y, 'r')
plt.title("Input signal")
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(x, imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")

plt.tight_layout()
plt.savefig('../JPG/test_1925_6.jpg')
print('输出完成')