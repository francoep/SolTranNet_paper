import torch
import numpy as np
import argparse
import time
from transformer import make_model as make_dep_model
import gzip

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

#setting up the arguments
parser=argparse.ArgumentParser(description="Create Training Jobs for hyper parameter sweeps")
parser.add_argument('-i','--infile',type=str,required=True,help='PATH to the file you wish to use. Assumes the content is 1 SMILE per line')
parser.add_argument('--batchsize',type=int,default=[1,8,16,32,64],nargs='+',help='Batch size(s) for loading the data from <infile>. Defaults to [1,8,16,32,64].')
parser.add_argument('--numruns',type=int,default=10,help='Number of runs to perform per batchsize. Defaults to 10.')
parser.add_argument('--cpu',action='store_true',help='Flag to force a CPU version of the model to be used.')
parser.add_argument('--permolecule',action='store_true',help='Flag to divide the stats by the number of SMILES evaluate.')
args=parser.parse_args()


#importing the spcific versions of the model for testing.
if args.cpu:
    from cpu_data_utils import load_data_from_df as ldfd_dep
    from cpu_data_utils import construct_loader as cl_dep
    from cpu_data_utils import load_data_from_smiles as ldfs_dep
else:
    from data_utils import load_data_from_df as ldfd_dep
    from data_utils import construct_loader as cl_dep
    from data_utils import load_data_from_smiles as ldfs_dep


model_params = {
    'd_atom': 28,
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

model=make_dep_model(**model_params)
if not args.cpu:
    model.to('cuda')

if args.infile.endswith('.gz'):
    smis=gzip.open(args.infile)
else:
    smis=open(args.infile).readlines()
smis=[x.rstrip() for x in smis]

for size in args.batchsize:
    print('--------------- Batch Size:',size,' -----------------')
    scpu_2d=[]
    for run in range(args.numruns):
        stn_cpu_times_2d=[]
        stn_cpu_times_2d.append(complex_measure_dep(model,smis,batch_size=size))
        
        scpu_2d.append(np.mean([x[0] for x in stn_cpu_times_2d]))

    if args.permolecule:
        print(f'STN  time:{np.mean(scpu_2d)/len(smis)}  std:{np.std(scpu_2d)/len(smis)}')
    else:
        print(f'STN  time:{np.mean(scpu_2d)}  std:{np.std(scpu_2d)}')
