import numpy as np
import torch
import copy
import scipy.sparse as sp
def get_adjacency_matrix(n_joint, parent_child):
    # undA = np.zeros((n_joint, n_joint))
    directA = np.zeros((n_joint, n_joint))
    for ver in parent_child:
        # undA[ver[0], ver[1]] = 1.0
        # undA[ver[1], ver[0]] = 1.0
        directA[ver[0], ver[1]] = 1.0
    # undA_hat = undA + np.eye(n_joint)
    # undA_hat = get_Laplacian_matrix(torch.FloatTensor(undA_hat))
    # return  undA_hat
    directA_hat = directA + np.eye(n_joint)
    directA_hat = get_Laplacian_matrix(torch.FloatTensor(directA_hat))
    return directA_hat
def get_sparse_adjacency_matrix(n_joint, parent_child):
    #print(np.ones(len(parent_child)), np.array(parent_child)[:,0])
    adj = sp.coo_matrix((np.ones(len(parent_child)), (np.array(parent_child)[:,0], np.array(parent_child)[:,1])), shape=(n_joint, n_joint), dtype=np.float32)
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(n_joint))
    return sparse_mx_to_torch_sparse_tensor(adj)
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def get_Q_input(state, action):
    '''
    state: batchxNx2 (torch) (N = 8)
    action: batchx6x1 (torch) (6 need to pad -> 8)
    - pad 0 to top and down of acton tensor 
    - concate state with action -> (batchsize, N, 3)
    '''
    act = copy.copy(action)
    # sort position of action to be suitable with state
    act = torch.index_select(act, 1, torch.LongTensor([0,3,1,4,2,5]))
    padding = torch.zeros(act.size()[0], 1)

    act = torch.cat([padding, act], 1)
    act = torch.cat([act, padding], 1)
    act = act.reshape(*act.size(),1)
    out = torch.cat([state, act], 2)
    
    return out
def get_Laplacian_matrix(adj_hat):
    Dl = torch.diag_embed(torch.pow(adj_hat.sum(axis = 0), -0.5))
    DlADl = torch.matmul(torch.matmul(Dl, adj_hat), Dl)
    return DlADl
def flip_transition(state, action, next_state, reward, done):
    flip_state = copy.deepcopy(state)
    flip_action = copy.deepcopy(action)
    flip_next_state = copy.deepcopy(next_state)

    flip_state[-1][0], flip_state[-1][1] = flip_state[-1][1], flip_state[-1][0]
    flip_next_state[-1][0], flip_next_state[-1][1] = flip_next_state[-1][1], flip_next_state[-1][0]
    flip_action[:3], flip_action[3:] = flip_action[3:], flip_action[:3]
    for i in range(1, len(state)-1, 2):
        flip_state[i], flip_state[i+1] = flip_state[i+1], flip_state[i]
        flip_next_state[i], flip_next_state[i+1] = flip_next_state[i+1], flip_next_state[i]
    return [flip_state, flip_action, flip_next_state, reward, done]
def modify_action(action):
    act = action.copy()
    H, L, h, l = 1, -1, 0.8, -0.8
    # act[1] = (act[0] -(L*l - H*h)/(l-h))*(l-h)/(H-L)
    # act[4] = (act[3] -(L*l - H*h)/(l-h))*(l-h)/(H-L)
    # act[4] = -act[3]
    act[1] = -act[0]
    return act
if __name__ == "__main__":
    '''
    0: COM
    1: Right Hip
    2: left Hip
    3: Right knee
    4: Left knee
    5: Right ankle
    6: Left ankle
    7: ground state of foot [1, 0] [0, 1], [0, 0] or [1, 1]
    '''

    # parent_child = ((1,0),(2,0),(1,2), (3,1),(2,1),(4,2),(5,3),(6,4))
    # uA = get_adjacency_matrix(8, 2, parent_child)
    # uA = torch.FloatTensor(uA)
    # print(uA)
    
    # DlADl = get_Laplacian_matrix(uA)
    # print(DlADl)

    state = torch.FloatTensor([[[0.3, 0.6], [-0.1, 0.2], [1.2,0.2],[-0.1,0.2],[0.5, 0.1],[0.2,0.2], [0.5, 0.1],[0, 1]],[[0.2, 0.1],[-0.7,0.2],[0,0.01],[-1,0],[0,0],[1,1], [0.5, 0.1],[1,1]]])
    # print(state.size())
    # print(torch.matmul(DlADl, state))
    
    action = torch.FloatTensor([[0.2,0,0.1,0.3,0.1,0.1],[0.5,-0.6,0.1,0.2,0.3,0.4]])
    Q_input = get_Q_input(state,action)
    print(Q_input)

    # state = [[0.3, 0.6], [-0.1, 0.2], [1.2,0.2],[-0.1,0.2],[0.5, 0.1],[0.2,0.2], [0.5, 0.1],[0,1]]
    # next_state = [[0.2, 0.1],[-0.7,0.2],[0,0.01],[-1,0],[0,0],[1,1],[0,1],[1,1]]
    # action = [0.2,0,0.1,0.3,0.5,0.8]
    # trans = flip_transition(state, action, next_state, -2.3, 0)
    # print(trans[0])
    # print(state)