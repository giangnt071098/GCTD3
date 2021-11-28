### this code is based on https://github.com/tkipf/pygcn/

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#from torch.nn.modules.module import Module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# layer Graph Convolution
class GraphConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature).to(device))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self, init_mode = None):
        if init_mode == "xavier":
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.bias is not None:
                self.bias.data.fill_(0.01)
        else:
            stdv = 1. / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        DAD = adj
        XW = torch.matmul(input, self.weight)
        DADXW = torch.matmul(DAD, XW)
        if self.bias is not None:
            return DADXW + self.bias
        else:
            return DADXW
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_feature) + ' -> ' \
               + str(self.out_feature) + ')'
# Model GCN
class GCN(nn.Module):
    def __init__(self, num_joint, feature_dim, hidden_dim =32, dropout=0.2):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(feature_dim, hidden_dim).to(device)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim).to(device)
        self.gc3 = GraphConvolution(hidden_dim, hidden_dim//2).to(device)
        # self.gc4 = GraphConvolution(hidden_dim//2, hidden_dim//2).to(device)

        # self.gc3_1 = GraphConvolution(hidden_dim*2, hidden_dim*2).to(device)
        # self.gc3_2 = GraphConvolution(hidden_dim*2, hidden_dim).to(device)
        # self.gc3_3 = GraphConvolution(hidden_dim, hidden_dim).to(device)
        # self.gc3_4 = GraphConvolution(hidden_dim, hidden_dim//2).to(device)

        self.dropout = dropout
        # self.output_dim = self.cal_dim(num_joint, feature_dim)
    def readout_mean(self, x):
        return torch.mean(x, 1)
    def cal_dim(self, num_joint, feature_dim):
        adj = torch.zeros((num_joint, num_joint))
        fea = torch.zeros((1, num_joint, feature_dim))
        dims = self.gc1(fea, adj)
        dims = self.gc2(dims, adj)
        dims = self.gc3(dims, adj)
        # dims = self.gc4(dims, adj)

        # dims = self.gc1_1(fea, adj)
        # dims = self.gc1_2(dims, adj)
        # dims = self.gc1_3(dims, adj)
        # dims = self.gc1_4(dims, adj)
        # dims = self.gc3_1(dims, adj)
        # dims = self.gc3_2(dims, adj)
        # dims = self.gc3_3(dims, adj)
        # dims = self.gc3_4(dims, adj)

        return int(np.prod(dims.size()))
    def forward(self, x, adj, training = False):
        # x = F.relu(self.gc1_1(x, adj))
        # x = F.relu(self.gc1_2(x, adj))
        # # x = F.relu(self.gc1_3(x, adj))
        # x = F.relu(self.gc1_4(x, adj))

        # x = F.relu(self.gc3_1(x, adj))
        # x = F.relu(self.gc3_2(x, adj))
        # x = F.relu(self.gc3_3(x, adj))
        # x = F.relu(self.gc3_4(x, adj))
    

        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        # x = F.relu(self.gc4(x, adj))

        
        #readout mean layer
        # x = x.mean(dim=1)

        # x = x.view(x.size()[0], -1)
        # return x

        # outr = x[:,[0,1,2,3,5,7],:].view(x.size()[0],-1)
        # outl = x[:,[0,2,4,6,7],:].view(x.size()[0],-1)

        # return outr, outl, x.view(x.size()[0], -1)
        return x.view(x.size()[0], -1)


if __name__ =="__main__":
    #graphModel = GraphConvolution(2,5) # 2== number of feature of input, 5 is hyperparameter

    adj = torch.FloatTensor(np.array([[1, 1, 1, 0],[1,1,1,0],[1,1,1,1],[0,1,0,1]]))

    #adj_hat = torch.FloatTensor(adj + np.eye(adj.shape[0], dtype=np.float64))
    input = torch.FloatTensor(np.array([[[1.2,0.2],[-0.1,0.2],[0.5, 0.1],[0.2,0.2]],[[0.2, 0.1],[-0.7,0.2],[0,0.01],[-1,0]]]))
    input.view(input.size()[0], -1)
    #print(input)


    Model = GCN(4, 2, 16) #4 is number of joints, 2 is feature of each joint
    Model(input, adj)
    print(np.unique(Model.gc1.weight.data)[-4:])
    #print(Model.output_dim)