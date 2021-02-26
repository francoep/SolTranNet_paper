import torch
import numpy as np
from data_utils import load_data_from_df as ldfd_dep
from data_utils import construct_loader as cl_dep
from data_utils import load_data_from_smiles as ldfs_dep
from transformer import make_model as make_dep_model
import time

def complex_measure_dep(model,list_of_smiles,batch_size=1):
    '''
    A more complicated measure of the forward-pass.
    
    This function takes in a list of smile strings, then will create the needed molecular graphs & run the forward-pass
    
    Assumes the model is in evaluate mode
    '''
    t0=time.time()
    ls=[0.0 for x in list_of_smiles]
    t1=time.time()
    X,y=ldfs_dep(list_of_smiles,ls,add_dummy_node=True,one_hot_formal_charge=True)
    tload=time.time()-t1
    
    data_loader=cl_dep(X,y,batch_size)
    
    for batch in data_loader:
        adjacency_matrix, node_features, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        y_pred = model(node_features, batch_mask, adjacency_matrix, None)
    elapsed_fp=time.time()-t0
    return elapsed_fp,tload

d_atom=28

stn_model_params = {
    'd_atom': d_atom,
    'd_model': 8,
    'N': 8,
    'h': 2,
    'N_dense': 1,
    'lambda_attention': 0.5, 
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'dropout': 0.1,
    'aggregation_type': 'mean'
}

stn_cpu=make_dep_model(**stn_model_params)
stn_cpu.to('cpu')

smis=open('test').readlines()
smis=[x.split(',')[-1].rstrip() for x in smis[1:]]

for size in [1,8,16,32,64,128,132,150,200,250,600]:
    print('---------------',size,'-----------------')
    scpu_2d=[]
    for run in range(10):
        stn_cpu_times_2d=[]
        stn_cpu_times_2d.append(complex_measure_dep(stn_cpu,smis,batch_size=size))
        
        scpu_2d.append(np.mean([x[0] for x in stn_cpu_times_2d]))
    print(f'STN - 2D {np.mean(scpu_2d)/len(smis)}  {np.std(scpu_2d)/len(smis)}')