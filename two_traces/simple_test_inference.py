# %% 

import matplotlib.pyplot as plt 
import sys
# sys.path.append("../")
from LSTM_Net import LSTM_Net 
import torch 
import torch.nn as nn 
import numpy as np 
import math 
from scipy.signal import ricker
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import m8r as sf
from skimage.transform import resize
import matplotlib as mlp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

#%% 

### 
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

opath = './Fig/'


scale='minmax'


hsz=50    # hidden size
LR = 0.007 # learning rate
num_layer = 2
model = LSTM_Net(input_size=1, hidden_layer_size=hsz, output_size=1,batch_sz=1,num_layer=num_layer) # model with (feature input,hidden_size,feature output, batch size)
model.load_state_dict(torch.load('./lstm_model.pth'))
model.cuda()
model.eval()
# %% 



# # Apply the learned network on deeper part
wav = ricker(50,5)
wav2= ricker(50,3)
wav3= ricker(100,6)
wav4= ricker(200,6)
x = np.concatenate((wav,0.5*wav2,-0.3*wav3,0.8*wav,-0.5*wav4),axis=0)

y = np.concatenate((x,0.3*wav4,0.6*wav3),axis=0)
x = np.concatenate((x,0.3*0.2*wav4,0.6*wav3),axis=0)

x = np.pad(x, (50,100), 'constant', constant_values=(0, 0))
y = np.pad(y, (60,90), 'constant', constant_values=(0, 0)) *0.75
# y = np.concatenate((y,0.3*wav),axis=0)
# y = np.pad(y, (0,150), 'constant', constant_values=(0, 0))

# plt.figure(figsize=(10,3))
# plt.plot(x,label='base')
# plt.plot(y,label='mon')
# plt.legend(fontsize=13)


x = x.reshape((-1,1))
y = y.reshape((-1,1))

# x = x.reshape((-1,1))[350:]
# y = y.reshape((-1,1))[350:]

if scale == 'minmax':
        scalerx= MinMaxScaler((-1,1))
        scalerx.fit(x)
        x_sc = scalerx.transform(x)
        scalery= MinMaxScaler((-1,1))
        scalery.fit(y)
        y_sc = scalery.transform(y)

elif scale== 'standard': 
        scalerx= StandardScaler()
        scalerx.fit(x)
        x_sc = scalerx.transform(x)
        scalery= StandardScaler()
        scalery.fit(y)
        y_sc = scalery.transform(y)


x_sc = torch.from_numpy(x_sc).float()
y_sc = torch.from_numpy(y_sc).float()
model.eval()

inp = x_sc.view(1,x_sc.shape[0],-1).cuda()
y_pred = model(inp)
y_pred = y_pred.cpu().detach().numpy()

y_pred = y_pred.reshape((-1,1))
y_pred = scalery.inverse_transform(y_pred)


plt.figure(figsize=(10,4))
plt.plot(y_pred,color='red',linewidth=2,label='Predicted Monitor')
plt.plot(y[:],color='black',label='Monitor')
plt.xlabel('Time samples',fontsize=12,fontweight='heavy')
plt.ylabel('Amplitude',fontsize=12,fontweight='heavy')
plt.legend(fontsize=10,shadow=True)
plt.ylim(-.5,.5)
plt.savefig(opath+'trace_matching_predicted')
plt.show()
# plt.savefig(opath+'trace_matching_2',format='pdf')

plt.figure(figsize=(10,4))
plt.plot(y_pred[:]-y[:],label='Difference after processing',color='black')
plt.xlabel('Time samples',fontsize=12,fontweight='heavy')
plt.ylabel('Amplitude',fontsize=12,fontweight='heavy')
plt.legend(fontsize=10,shadow=True)
plt.ylim(-.5,.5)
plt.savefig(opath+'Difference_after')
plt.show()

diff_pred = y_pred[:]-y[:] 

# plt.savefig(opath+'trace_difference_2')
# %%
wav = ricker(50,5)
wav2= ricker(50,3)
wav3= ricker(100,6)
wav4= ricker(200,6)
x = np.concatenate((wav,0.5*wav2,-0.3*wav3,0.8*wav,-0.5*wav4),axis=0)

y = np.concatenate((x,0.3*wav4,0.6*wav3),axis=0)
x = np.concatenate((x,0.3*0.2*wav4,0.6*wav3),axis=0)



x = np.pad(x, (60,90), 'constant', constant_values=(0, 0))
y = np.pad(y, (60,90), 'constant', constant_values=(0, 0)) 


# plt.figure(figsize=(10,4))
# plt.plot(x-y,label='True reservoir variation',color='black')
# plt.xlabel('Time samples',fontsize=12,fontweight='heavy')
# plt.ylabel('Amplitude',fontsize=12,fontweight='heavy')
# plt.legend(fontsize=10,shadow=True)
# plt.ylim(-.5,.5)
# plt.show()

diff_true = x-y 

# %% 

plt.figure(figsize=(10,4))
plt.plot(diff_true,label='True reservoir variation',color='black',linewidth=2)
plt.plot(diff_pred,label='Difference after processing',color='red',alpha=0.6)
plt.xlabel('Time samples',fontsize=12,fontweight='heavy')
plt.ylabel('Amplitude',fontsize=12,fontweight='heavy')
plt.legend(fontsize=10,shadow=True)
plt.ylim(-.5,.5)
plt.savefig(opath+'Reservoir diff')
plt.show()
# %%
