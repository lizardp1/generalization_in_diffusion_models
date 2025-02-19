import numpy as np
import torch.nn as nn
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import argparse
from network_3 import SimpleNet  # Changed to use simpler network
from trainer_4 import run_training

def build_path(args):
    '''Build path to save results of training'''
    dir_name = f"{args.dir_name}{args.arch_name}/{args.data_name}/{args.set_size}"
    if args.swap:
        dir_name = dir_name + '_swapped'
    return dir_name

def make_loader(train_set, test_set, args):
    # Convert to TensorDataset
    train_dataset = TensorDataset(train_set)
    test_dataset = TensorDataset(test_set)
    
    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return trainloader, testloader

def prep_data_swap(data, args):
    """Prepare train and test splits"""
    # Convert set_size to int if it's a tensor
    if torch.is_tensor(args.set_size):
        args.set_size = args.set_size.item()
    
    args.set_size = min(args.set_size, len(data) - 1000)  # Ensure we leave enough for test set
    
    if not args.swap:
        train_set = data[:args.set_size]
        test_set = data[-1000:]
    else:
        train_set = data[-args.set_size:]
        test_set = data[:1000]
    
    # Save datasets
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)
    torch.save(train_set, os.path.join(args.dir_name, 'train_set_gmm_2d.pt'))
    torch.save(test_set, os.path.join(args.dir_name, 'test_set_gmm_2d.pt'))
    
    return train_set, test_set

def repeat_images(train_set, args, N_total): 
    """Repeat the training set to achieve desired total size"""
    if len(train_set) == 0:
        raise ValueError("Training set is empty!")
    
    n = max(1, int(N_total/len(train_set)))
    repeated_data = train_set.repeat(n, 1)
    
    # If we need more data to reach N_total exactly
    remaining = N_total - len(repeated_data)
    if remaining > 0:
        repeated_data = torch.cat([repeated_data, train_set[:remaining]])
    
    return repeated_data

def main():
    parser = argparse.ArgumentParser(description='training a score network for GMM')
    
    # Architecture
    parser.add_argument('--arch_name', type=str, default='SimpleNet')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr_freq', type=int, default=100)
    
    # Dataset
    parser.add_argument('--noise_level_range', default=[0.1, 1.0])
    parser.add_argument('--swap', default=False)
    parser.add_argument('--set_size', default=None)
    parser.add_argument('--data_name', type=str, default='gmm')
    
    # Directory
    parser.add_argument('--data_root_path', default='datasets/')
    parser.add_argument('--dir_name', default='denoisers/')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', default=False)
    parser.add_argument('--SLURM_ARRAY_TASK_ID', type=int, default=1)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set size based on SLURM task ID
    args.set_size = 10**args.SLURM_ARRAY_TASK_ID
    args.data_path = os.path.join(args.data_root_path, args.data_name)
    
    # Load data
    train_set = torch.load(os.path.join(args.data_path, 'gmm_100000_samples_2d.pt'))
    print('all data:', train_set.size())
    
    if train_set.shape[0] < args.set_size:
        print(f"Warning: requested set_size {args.set_size} is larger than available data {train_set.shape[0]}")
        args.set_size = train_set.shape[0] // 2
    
    # Create save directory
    args.dir_name = build_path(args)
    os.makedirs(args.dir_name, exist_ok=True)
    
    # Prepare data
    train_set, test_set = prep_data_swap(train_set, args)
    print('train:', train_set.size(), 'test:', test_set.size())
    
    if args.debug:
        train_set = train_set[0:args.batch_size]
        test_set = test_set[0:args.batch_size]
        args.num_epochs = 5
    else:
        train_set = repeat_images(train_set, args, N_total=20000)
    
    print('train:', train_set.size(), 'test:', test_set.size())
    trainloader, testloader = make_loader(train_set, test_set, args)
    
    # Initialize model
    model = SimpleNet(input_dim=2, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    if torch.cuda.is_available():
        print('[ Using CUDA ]')
        model = model.cuda()
    
    print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Set up training
    criterion = nn.MSELoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Save args
    torch.save(args, os.path.join(args.dir_name, 'args.pt'))
    
    # Train
    model = run_training(model, trainloader, testloader, criterion, optimizer, args)
    
if __name__ == "__main__":
    main()