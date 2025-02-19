import numpy as np
import torch.nn as nn
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import argparse
from network_5 import PointUNet
from trainer_6 import run_training

def build_path(args):
    '''Build path to save results of training'''
    dir_name = f"{args.dir_name}{args.arch_name}_denoising/{args.data_name}/{args.set_size}"
    if args.swap:
        dir_name = dir_name + '_swapped'
    return dir_name

def make_loader(train_set, test_set, args):
    train_dataset = TensorDataset(train_set)
    test_dataset = TensorDataset(test_set)
    
    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return trainloader, testloader

def main():
    parser = argparse.ArgumentParser(description='training a point denoiser')
    
    # Architecture
    parser.add_argument('--arch_name', type=str, default='PointUNet')
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--num_kernels', type=int, default=64)
    parser.add_argument('--num_enc_conv', type=int, default=2)
    parser.add_argument('--num_mid_conv', type=int, default=2)
    parser.add_argument('--num_dec_conv', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=2) #dimension of data
    parser.add_argument('--bias', default=False)
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr_freq', type=int, default=100)
    
    # Dataset
    parser.add_argument('--noise_level_range', nargs=2, type=float, default=[0.0001, 1.0])
    parser.add_argument('--noise_type', type=str, default='gaussian')
    parser.add_argument('--swap', action='store_true')
    parser.add_argument('--set_size', type=int, default=None)
    parser.add_argument('--data_name', type=str, default='gmm')
    
    # Directory
    parser.add_argument('--data_root_path', default='datasets/')
    parser.add_argument('--dir_name', default='denoisers/noise_tests/')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--SLURM_ARRAY_TASK_ID', type=int, default=1)
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set size based on SLURM task ID
    args.set_size = 10**args.SLURM_ARRAY_TASK_ID
    
    # Load data
    data_path = os.path.join(args.data_root_path, args.data_name, 'gmm_100000_samples_100d.pt')
    data = torch.load(data_path)
    print('all data:', data.size())
    
    if data.shape[0] < args.set_size:
        print(f"Warning: requested set_size {args.set_size} is larger than available data {data.shape[0]}")
        args.set_size = data.shape[0] // 2
    
    # Create train/test split
    train_set = data[:args.set_size]
    test_set = data[-1000:]
    print('Split - train:', train_set.size(), 'test:', test_set.size())
    
    if args.debug:
        train_set = train_set[:args.batch_size]
        test_set = test_set[:args.batch_size]
        args.num_epochs = 5
    
    # Create save directory
    args.dir_name = build_path(args)
    os.makedirs(args.dir_name, exist_ok=True)
    
    trainloader, testloader = make_loader(train_set, test_set, args)
    
    # Initialize model
    model = PointUNet(args).to(args.device)
    print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Set up training
    criterion = nn.MSELoss(reduction='sum')  # Using MSE for direct denoising
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Save args
    torch.save(args, os.path.join(args.dir_name, 'args.pt'))
    
    # Train
    model = run_training(model, trainloader, testloader, criterion, optimizer, args)
    
if __name__ == "__main__":
    main()