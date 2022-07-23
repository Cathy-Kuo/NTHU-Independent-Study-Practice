import torch 
import torch.nn as nn
import networkx as nx
import numpy as np

'''
neighbors_sum = torch.sparse.mm(adj_list , emb_matrix[0])
neighbors_sum = neighbors_sum.view(batch_size , neighbors_sum.shape[0] , neighbors_sum.shape[1])
'''

class embedding_network(nn.Module):
    
    def __init__(self , emb_dim = 64 , T = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.T = T
        self.W1 = nn.Linear( 1 , emb_dim , bias = False)
        self.W2 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W3 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W4 = nn.Linear( 1 , emb_dim , bias = False)
        self.W5 = nn.Linear(emb_dim*2,1 , bias = False)
        self.W6 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W7 = nn.Linear(emb_dim , emb_dim , bias = False)
        
        self.relu = nn.ReLU()
        
    def forward(self , graph , Xv ):
        batch_size = Xv.shape[0]
        n_vertex = Xv.shape[1]
        graph_edge = torch.unsqueeze(graph , 3)
        

        emb_matrix = torch.zeros([batch_size , n_vertex , self.emb_dim ]).type(torch.DoubleTensor)
        
        if 'cuda' in Xv.type():
            emb_matrix = emb_matrix.cuda()
        for t in range(self.T):
            neighbor_sum = torch.bmm(graph , emb_matrix )
            v1 = self.W1(Xv)
            v2 = self.W2(neighbor_sum)
            v3 = self.W4(graph_edge)
            v3 = self.W3(torch.sum(v3 , 2))
            
            v = v1 + v2 + v3
            emb_matrix = v.clone()
            emb_matrix = self.relu(emb_matrix)
            #print(v1 , v2 , v3)
            #print('=================')
            #print(v[0][0])
        #print(emb_matrix.shape)
        emb_sum = torch.sum(emb_matrix , 1)
        v6 = self.W6(emb_sum)
        v6 = v6.repeat(1,n_vertex )
        v6 = v6.view(batch_size , n_vertex , self.emb_dim)
        v7 = self.W7(emb_matrix)
        ct = self.relu(torch.cat([v6 , v7] , 2))
        
        return torch.squeeze(self.W5(ct) , 2)