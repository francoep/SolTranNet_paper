import soltrannet as stn
import numpy as np
import torch
import argparse, gzip, multiprocessing, time

#setting up the arguments
parser=argparse.ArgumentParser(description="Create Training Jobs for hyper parameter sweeps")
parser.add_argument('-i','--infile',type=str,required=True,help='PATH to the file you wish to use. Assumes the content is 1 SMILE per line')
parser.add_argument('--batchsize',type=int,default=[1,8,16,32,64],nargs='+',help='Batch size(s) for loading the data from <infile>. Defaults to [1,8,16,32,64].')
parser.add_argument('--numruns',type=int,default=10,help='Number of runs to perform per batchsize. Defaults to 10.')
parser.add_argument('--cpus',type=int,default=multiprocessing.cpu_count(),help='Number of CPU cores to use for the data loader. Defaults to all available cores. Pass 0 to run on 1 CPU core.')
parser.add_argument('--cpu_predict',action='store_true',help='Flag to force a CPU version of the model to be used.')
parser.add_argument('--permolecule',action='store_true',help='Flag to divide the stats by the number of SMILES evaluate.')
args=parser.parse_args()

#loading the smiles into memory
if args.infile.endswith('.gz'):
    smis=gzip.open(args.infile)
else:
    smis=open(args.infile).readlines()
smis=[x.rstrip() for x in smis]

for size in args.batchsize:
    print('--------------- Batch Size:',size,' -----------------')
    scpu_2d=[]
    for run in range(args.numruns):
        start=time.time()
        
        if args.cpu_predict:
            predictions=list(stn.predict(smiles,batch_size=size, num_workers=args.cpus,device=torch.device('cpu')))
        else:
            predictions=list(stn.predict(smiles,batch_size=size, num_workers=args.cpus))
        scpu_2d.append(time.time()-start)

    if args.permolecule:
        print(f'STN  time:{np.mean(scpu_2d)/len(smis)}  std:{np.std(scpu_2d)/len(smis)}')
    else:
        print(f'STN  time:{np.mean(scpu_2d)}  std:{np.std(scpu_2d)}')
