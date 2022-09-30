# from TL2 import TLold as TL
import matplotlib.pyplot as plt
from prep_data import Dataset 
from torch.utils.data import DataLoader
from LSTM_Net import LSTM_Net
import torch.nn as nn 
import torch 
from torch.autograd import Variable
import numpy as np
import math 
import gc
import time 
from skimage.transform import resize
from matplotlib.ticker import FormatStrFormatter





  
# def plot_history(trainingloss,testingloss,name,show=False):
#     trainingloss = np.array(trainingloss)
#     testingloss =  np.array(testingloss)
#     Epc = np.arange(1,trainingloss.shape[0]+1)
#     plt.figure(figsize=(8,6))
#     plt.semilogy(Epc,trainingloss,color='b',label='Training')
#     plt.semilogy(Epc,testingloss,color='r',label='Validation')
    
#     plt.xlabel('Epochs', fontsize=24, fontweight='bold')
#     plt.ylabel('Loss',fontsize=24, fontweight='bold')
#     plt.legend()
#     plt.tick_params(axis='both',which='minor',labelsize=20)
#     plt.xticks(fontsize=20,fontweight='semibold')
#     plt.yticks(fontsize=20,fontweight='semibold')
#     plt.legend(prop={'size': 20, 'weight':'bold'})    
#     plt.savefig(name, bbox_inches='tight')
#     if show:
#         plt.show()
#     plt.close()

  
def plot_history(trainingloss,testingloss,netname):

    Epc = np.arange(1,trainingloss.shape[0]+1)
    fig, ax = plt.subplots(figsize=(8,4))

    ax.semilogy(trainingloss,color='b',label='Training')
    ax.semilogy(testingloss,color='r',label='Validation')

    ax.grid(which='both')
    ax.set_xlabel('Epochs', fontsize=16, fontweight='bold')
    ax.set_ylabel('Loss ',fontsize=16, fontweight='bold')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

    ax.tick_params(axis='both',which='minor',labelsize=16)
    plt.xticks(fontsize=16,fontweight='semibold')
    plt.yticks(fontsize=16,fontweight='semibold')
    ax.legend(prop={'size': 16, 'weight':'bold'})

    name=netname+'.png'
    fig.savefig(name, bbox_inches='tight',format='eps',dpi=1200)



def plot_r2(trainingloss,testingloss,netname):

    Epc = np.arange(1,trainingloss.shape[0]+1)
    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(trainingloss,color='b',label='Training')
    ax.plot(testingloss,color='r',label='Validation')

    ax.grid(which='both')
    ax.set_xlabel('Epochs', fontsize=16, fontweight='bold')
    ax.set_ylabel('R2-score ',fontsize=16, fontweight='bold')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.tick_params(axis='both',which='minor',labelsize=16)
    plt.xticks(fontsize=16,fontweight='semibold')
    plt.yticks(fontsize=16,fontweight='semibold')
    ax.legend(prop={'size': 16, 'weight':'bold'})


    name='./output/figure/R2_'+netname+'.png'
    plt.savefig(name, bbox_inches='tight')


def r2_score(target, prediction):
    """Calculates the r2 score of the model
    
    Args-
        target- Actual values of the target variable
        prediction- Predicted values, calculated using the model
        
    Returns- 
        r2- r-squared score of the model
    """
    r2 = 1- torch.sum((target-prediction)**2) / torch.sum((target-target.mean())**2)
    return r2.cpu()




         

def muting(data,mxoff,parm,inverse=False):
        ns = parm['ns']
        ng = parm['ng']
        ds = parm['ds']
        dg = parm['dg']
        os = parm['os']
        og = parm['og']
        nt = parm['nt']
        s = np.arange(os,os+(ns)*ds,ds)
        r = np.zeros((ns,ng))
        oup = np.zeros((ns,ng,nt))

        for i in range(ns):
            if s[i]-mxoff < 0:
                idx1=0
            else:
                idx1= int((s[i]-mxoff)//dg) 

            if s[i]+mxoff > ng*dg+og:
                idx2 = int((ng*dg+og)//dg)
            else:
                idx2 = int((s[i]+mxoff)//dg)

            # print(idx1,idx2)    
            r[i,idx1:idx2] = np.ones(idx2-idx1) 
            
            
        indices = np.nonzero(r)
        if not inverse: oup[indices[0],indices[1],:] = data[indices[0],indices[1],:]
        if inverse:
            print('inverse muting') 
            oup[indices[0],indices[1],:] = data # note data here is 2d         
        return oup
        
        
def load_3drsf_data(filename):             
        f = sf.Input(filename)
        nt= f.int("n1")
        ng= f.int("n2")
        ns= f.int("n3")
        dt = f.float("d1")
        dg = f.float("d2")
        ds = f.float("d3")
        ot = f.float("o1")
        og = f.float("o2")
        os = f.float("o3")
        # note in reading rsf to numpy the diload_rsf_datamension are reverse 
        data = np.zeros((ns,ng,nt),dtype=np.float32)
        f.read(data)
        print('Shape of loaded data: {}'.format(np.shape(data)))
        parm = {'nt':nt, 'ng':ng, 'ns':ns, 
                'dt':dt, 'dg':dg, 'ds':ds,
                'ot':ot, 'og':og, 'os':os}
        return data,parm  
    
    
