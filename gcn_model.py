import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn import manifold

def normalize(A , symmetric=True):
	# A = A+I
	A = A + torch.eye(A.size(0))
	# 所有节点的度
	d = A.sum(1)
	if symmetric:
		#D = D^-1/2
		D = torch.diag(torch.pow(d , -0.5))
		return D.mm(A).mm(D)
	else :
		# D=D^-1
		D =torch.diag(torch.pow(d,-1))
		return D.mm(A)

def p_condition():
    p_co=pd.read_csv('E:/xiaozhe/onsets-and-frames-master/onsets_and_frames/p_co.csv')
    p_co = p_co.values
    p_co=p_co+p_co.T    
#    A_condition = preprocessing.normalize(p_co, axis=1, norm='l1')#行归一化
    A_condition = preprocessing.normalize(p_co, norm='max')
    for i in range (p_co.shape[0]):
        for j in range (p_co.shape[1]):
            if A_condition[i,j]>=0.6:#取一定阈值进行二值化
                A_condition[i,j]=1
            else:
                A_condition[i,j]=0
    return A_condition
    
def show_graph(data, label_num):
    # X=np.identity(label_num)
    X=data
    y=np.argmax(np.eye(label_num), axis=1)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=601)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}.  Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
     
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    
    marker= ['.',',', 'o','v','^','<','>','8','s','p','*','+']
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], s=60, color=plt.cm.tab20(y[i]%12), marker=marker[y[i]%12])
        plt.text(X_norm[i, 0]+0.015, X_norm[i, 1]-0.01, str(y[i]), color=plt.cm.tab20(y[i]%12), fontdict={'size': 10}, alpha= 1)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('matrix_A.pdf')
    plt.show()

class GCN(nn.Module):
    '''
    Z = AXW
    '''
    def __init__(self , dim_in , dim_out):
        super(GCN,self).__init__()
        
        self.A = normalize(torch.FloatTensor(p_condition()),True).to('cuda')
        self.x = torch.eye(self.A.size(0)).to('cuda')
        
        self.fc1 = nn.Linear(dim_in ,dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_out//2, bias=False)
        self.fc3 = nn.Linear(dim_out//2, dim_out, bias=False)
    def forward(self, intput):
        '''
        计算三层gcn
        '''
        X = F.relu(self.fc1(self.A))
        X = F.relu(self.fc2(self.A.mm(X)))
        X = self.fc3(self.A.mm(X))
        show_A = self.A.cpu().numpy()
        show_graph(show_A, show_A.shape[0])
        show_numpy = X.cpu().detach().numpy()
        show_graph(show_numpy, show_numpy.shape[0])
        X = torch.matmul(intput, torch.transpose(X,0,1))
        return X
    
if __name__ == '__main__':
     p_condition()