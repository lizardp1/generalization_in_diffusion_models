import numpy as np
import torch.nn as nn
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import argparse
import functools
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, SVHN
from network import ScoreNet, ScoreNetGMM
from trainer import marginal_prob_std, loss_fn, run_training

def build_path(args):
    '''Build path to save results of training'''
    dir_name = f"{args.dir_name}/gmm_score_model/{args.data_name}"
    return dir_name

def get_dataset(args):
    data_path = os.path.join(args.data_root_path, args.data_name, f'gmm_{args.set_size}_samples_{args.data_dim}d.pt')
    
    data = torch.load(data_path)
    print(f'Loaded data shape: {data.shape}')
    
    # Split into train and test
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    
    # Create dataloaders
    trainloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    testloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return trainloader, testloader

def main():
    parser = argparse.ArgumentParser(description='training a score-based model')
    
    # Architecture
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 256, 512])
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--data_dim', type=int, default=2, help='dim of GMM data')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Dataset
    parser.add_argument('--set_size', type=int, default=10000)
    parser.add_argument('--data_name', type=str, default='gmm')
    parser.add_argument('--sigma', type=float, default=25.0)
    
    # Directory
    parser.add_argument('--data_root_path', default='datasets/')
    parser.add_argument('--dir_name', default='score_models/')
    parser.add_argument('--save_freq', type=int, default=10)
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save directory
    args.dir_name = build_path(args)
    os.makedirs(args.dir_name, exist_ok=True)

    # Create train/test split
    trainloader, testloader = get_dataset(args)
    
    # Initialize model
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)
    score_model = ScoreNetGMM(
        data_dim=args.data_dim,
        hidden_dims=args.hidden_dims,
        embed_dim=args.embed_dim
    )
    score_model.marginal_prob_std = marginal_prob_std_fn

    print('Number of parameters:', sum(p.numel() for p in score_model.parameters() if p.requires_grad))

    # Save args
    torch.save(args, os.path.join(args.dir_name, 'args.pt'))
    
    # Train
    run_training(
        score_model=score_model,
        marginal_prob_std_fn=marginal_prob_std_fn,
        args=args,
        trainloader=trainloader,
        testloader=testloader
    )

if __name__ == "__main__":
    main()