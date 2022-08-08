import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from BigCLAM import  BerpoDecoder
from scipy import sparse
import dgl.data

Embedding_dims=37
#dataset = dgl.data.CoraGraphDataset()
dataset = dgl.data.PubmedGraphDataset()
g = dataset[0]

######################################################################
# Prepare training and testing sets
# Split edge set for training and testing
u, v = g.edges()
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.9)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
print(test_pos_u)
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

train_g = dgl.remove_edges(g, eids[:test_size])



from dgl.nn import SAGEConv

# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, 256, 'mean')
        self.conv2 = SAGEConv(256, h_feats, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = F.normalize(h, dim=1)
        return h




train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())



import dgl.function as fn

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            #print(g.edata['score'][:, 0])
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']



model = GraphSAGE(train_g.ndata['feat'].shape[1],Embedding_dims)
pred = MLPPredictor(Embedding_dims)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_ap(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy() #真实标签值
    y_test=labels
    y_pred=scores
    ap = average_precision_score(y_test,y_pred)
    return ap


optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

# ----------- training -------------------------------- #

bp_decoder = BerpoDecoder(train_g.num_nodes(), train_g.num_edges(), True)

from sklearn.metrics import roc_auc_score
all_logits = []
sp_adj = sparse.csr_matrix(train_g.adjacency_matrix().to_dense().numpy())  # 稀疏矩阵
auc_list=[]
ap_list=[]
for e in range(1000):
    # forward
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)
    loss +=1e+2 * bp_decoder.loss_full(h, sp_adj)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if e % 5 == 0:
        with torch.no_grad():
            pos_score = pred(test_pos_g, h)
            neg_score = pred(test_neg_g, h)
            auc=compute_auc(pos_score, neg_score)
            ap=compute_ap(pos_score, neg_score)
            auc_list.append(auc)
            ap_list.append(ap)
            print('In epoch {}, loss: {:.3f}, AUC = {:.3f}, AP = {:.3f}'.format(e, loss, auc,ap))



auc_list=np.array(auc_list)
ap_list=np.array(ap_list)
print('MAX-AUC = {:.3f}, AVE-AUC = {:.3f},STD-AUC={:.3f}，MAX-AP = {:.3f}，AVE-AP = {:.3f},STD-AP={:.3f}'.format(auc_list.max(),auc_list.mean(),np.std(auc_list,ddof=0),ap_list.max(),ap_list.mean(),np.std(ap_list)))

