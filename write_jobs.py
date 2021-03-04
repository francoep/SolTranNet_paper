#!/usr/bin/env python3

#This script will generate a text file containing the training jobs/sweeps for wandb in a grid search of the options specified.

#NOTE -- this script will make all combinations -- some of which might fail during actual training. train.py handle the sanitization of the inputs.

import argparse,itertools

parser=argparse.ArgumentParser(description="Create Training Jobs for hyper parameter sweeps")

#options that are the same for each run in the sweep
parser.add_argument('--trainfile',type=str,required=True,help='PATH to the training file you wish to use')
parser.add_argument('--testfile',type=str,required=True,help='PATH to the testing file you wish to use')
parser.add_argument('--datadir',type=str,default='sweep',help='Absolute path to where the output of the training will be placed. Defaults to sweep')
parser.add_argument('--savemodel',action='store_true',help='Flag to have the training save the final weights of the model')
parser.add_argument('-e','--epochs',default=100,help='Maximum number of epochs to run the training for. Defaults to 100.')
parser.add_argument('--lr',type=float,default=0.04,help='Learning rate for the given sweep. Defaults to 0.04.')
parser.add_argument('--loss',type=str,default='huber',help='Loss function to be used for training. Defaults to huber. Must be in [mse,mae,huber].')


#variable options
parser.add_argument('--dropout',type=float,default=0,nargs='+',help='Applying Dropout to model weights when training. Accepts any number of arguments.')
parser.add_argument('--ldist',type=float,default=0.33,nargs='+',help='Lambda for model attention to the distance matrix. Accepts any number of arguments. Defaults to 0.33 (even between dist, attention, and graph features)')
parser.add_argument('--lattn',type=float,default=0.33,nargs='+',help='Lambda for model attention to the attention matrix. Accepts any number of arguments. Defaults to 0.33 (even between dist, attenttion, and graph features)')
parser.add_argument('--ndense',type=int,default=1,nargs='+',help='Number of Dense blocks in FeedForward section. Accepts any number of arguments. Defaults to 1')
parser.add_argument('--heads',type=int,default=16,nargs='+',help='Number of attention heads in MultiHeaded Attention. Accepts any number of arguments. **Needs to evenly divide dmodel** Defaults to 16.')
parser.add_argument('--dmodel',type=int,default=1024,nargs='+',help='Dimension of the hidden layer for the model. Accepts any number of arguments. Defaults to 1024.')
parser.add_argument('--nstacklayers',type=int,default=8,nargs='+',help='Number of stacks in the Encoder layer. Accepts any number of arguments. Defaults to 8')
parser.add_argument('--seed',type=int,default=420,nargs='+',help='Random seed for training the models. Accepts any number of arguments. Defaults to 420.')
parser.add_argument('--dynamic',type=int,default=0,nargs='+',help='If set, the maximum number of epochs a model can not improve on the training set before stopping training. Defaults to not being set. Can accept any number of arguments.')

#additional add on options
parser.add_argument('--twod',action='store_true',help='Flag to only use 2D conformers for making the distance matrix.')
parser.add_argument('--wandb',default=None,help='Project name for weights and biases integration.')
parser.add_argument('--cpu',action='store_true',help='Flag to use CPU models for the train.py job.')
parser.add_argument('-o','--outname',default='grid_sweep.cmds',help='Output filename. Defaults to grid_sweep.cmds')
args=parser.parse_args()

#create the grid of the specified parameters
combos=list(itertools.product([args.dropout,args.ldist,args.lattn,args.ndense,args.heads,args.dmodel,args.nstacklayers,args.seed,args.dynamic]))
print(combos)
with open(args.outname,'w') as outfile:
    for c in combos:
        print(c)
        drop,lam_dist,lam_attn,nden,head,dim,nsl,s,dyn=c
        print(drop,lam_dist,lam_attn,nden,head,dim,nsl,s,dyn)
        sent=f'python3 train.py --trainfile {args.trainfile} --testfile{args.testfile} --datadir {args.datadir} --epochs {args.epochs} --lr {args.lr} --loss {args.loss} --dropout {drop} --ldist {lam_dist} --lattn {lam_attn} --Ndense {nden} --heads {head} --dmodel {dim} --nstacklayers {nsl} --seed {s} --dynamic {dyn}'
        if args.twod:
            sent+=' --twod'
        if args.wandb:
            sent+=f' --wandb {args.wandb}'
        if args.cpu:
            sent+=' --cpu'

        outfile.write(sent+'\n')
