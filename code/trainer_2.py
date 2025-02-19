## This module takes a network, a loss function, optimizer and data and trains a DNN denoiser 

import numpy as np
import torch.nn as nn
import os
import time
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
from network_2 import *
from quality_metrics_func import batch_ave_psnr_torch, calc_psnr
from plotting_func import plot_loss,plot_psnr, plot_denoised_range

########################################################### replacement point dataloader functions ###########################################################

def add_noise_to_points(points, noise_range, device, quadratic_noise=True):
    """Add noise to point coordinates"""
    batch_size = points.size(0)
    if quadratic_noise:
        noise_level = torch.sqrt(torch.rand(size=(batch_size, 1, 1), device=device)) * (noise_range[1] - noise_range[0]) + noise_range[0]
    else:
        noise_level = torch.rand(size=(batch_size, 1, 1), device=device).expand_as(points) * (noise_range[1] - noise_range[0]) + noise_range[0]
    
    noise = torch.randn_like(points) * noise_level #* 0.01  # Scale factor for point coordinates
    noisy_points = points + noise
    return noisy_points, noise

def add_noise_to_points_range(points, noise_range, device):
    """Add different noise levels to points for visualization"""
    noisy_points = []
    for noise_level in noise_range:
        noise = torch.randn_like(points) * noise_level * 0.01
        noisy_points.append(points + noise)
    return torch.stack(noisy_points), None

def compute_point_mse(pred_points, target_points):
    """Compute MSE between point sets"""
    return torch.mean((pred_points - target_points) ** 2)

########################################################### training ###########################################################
def one_iter(model, batch ,criterion, args):
    ## clean data
    clean = batch.to(args.device)
    if args.rescale:
        clean = clean * torch.rand(size=(batch.size(0), 1, 1), device=args.device)

    ## add noise 
    noisy, noise = add_noise_to_points(clean, args.noise_level_range, args.device, args.quadratic_noise)
    
    ## denoise 
    output = model(noisy)
    
    ## handle skip and coarse vs fine
    if args.skip:
        target = noise
        denoised = noisy - output
    else:
        target = clean
        denoised = output
  
    ## compute loss
    loss = criterion(output, target) / clean.size(0)
    
    # Compute MSE as quality metric (instead of PSNR)
    mse = compute_point_mse(clean, denoised)

    return model, loss, mse



def train_epoch(model, trainloader,criterion,optimizer, args):
    loss_sum = 0
    mse_sum = 0
    model.train()    

    for i, batch in enumerate(trainloader, 0):
       
        optimizer.zero_grad()
        model, loss, mse = one_iter(model, batch,criterion, args)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        mse_sum += mse.item()


    return model, loss_sum/(i+1), mse_sum/(i+1)



def test_epoch(model, testloader,criterion, args):
    loss_sum = 0
    mse_sum = 0
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(testloader, 0):
            model, loss, mse = one_iter(model, batch,criterion, args)
            loss_sum+= loss.item()
            mse_sum += mse.item()

    return loss_sum/(i+1), mse_sum/(i+1)

def visualize_point_denoising(writer, model, points, noise_range, epoch, prefix="train"):
    """Visualize point denoising at different noise levels"""
    model.eval()
    with torch.no_grad():
        # Print shapes for debugging
        print(f"Original points shape: {points.shape}")
        noisy_points, _ = add_noise_to_points_range(points, noise_range, points.device)
        print(f"Noisy points shape: {noisy_points.shape}")
        denoised_points = model(noisy_points)
        if denoised_points.shape != noisy_points.shape:
            denoised_points = denoised_points.transpose(1, 2)
        print(f"Denoised points shape: {denoised_points.shape}")
        
        # Take the first sample from the batch for visualization
        for i, noise_level in enumerate(noise_range):
            fig = plt.figure(figsize=(10, 10))
            
            # Adjust indexing based on actual tensor shapes
            points_reshaped = points[0].view(-1, 2)  # Reshape to [num_points, 2]
            noisy_reshaped = noisy_points[i, 0].view(-1, 2)
            denoised_reshaped = denoised_points[i, 0].view(-1, 2)
            
            # Plot points
            plt.scatter(points_reshaped[:, 0].cpu(), points_reshaped[:, 1].cpu(), 
                       label='Original', color='blue')
            plt.scatter(noisy_reshaped[:, 0].cpu(), noisy_reshaped[:, 1].cpu(), 
                       label='Noisy', color='red')
            plt.scatter(denoised_reshaped[:, 0].cpu(), denoised_reshaped[:, 1].cpu(), 
                       label='Denoised', color='green')
            
            plt.legend()
            plt.title(f'Noise Level: {noise_level:.2f}')
            plt.grid(True)
            
            writer.add_figure(f'Points/{prefix}_noise_{noise_level:.2f}', 
                            fig, 
                            global_step=epoch)
            plt.close()


def run_training(model, trainloader, testloader,criterion,optimizer, args):
    '''
    trains a denoiser neural network
    '''

    ###
    start_time_total = time.time()
    epoch_loss_list_train = [] #delet if TB works
    epoch_mse_list_train = []#delet if TB works
    epoch_loss_list_test = []#delet if TB works
    epoch_mse_list_test = []#delet if TB works
    writer = SummaryWriter(log_dir=args.dir_name)

    train_points = next(iter(trainloader))[0].to(args.device)
    test_points = next(iter(testloader))[0].to(args.device)
    
    ### loop over number of epochs
    for epoch in range(args.num_epochs):
        print('epoch', epoch)
        if epoch >= args.lr_freq and epoch%args.lr_freq==0:
            for param_group in optimizer.param_groups:
                args.lr = args.lr/2
                param_group["lr"] = args.lr

        #train
        model, epoch_loss_train, epoch_mse_train = train_epoch(model, trainloader,criterion,optimizer, args)
        epoch_loss_list_train.append(epoch_loss_train)#delet if TB works
        epoch_mse_list_train.append(epoch_mse_train)#delet if TB works
        writer.add_scalar('MSE/Train', epoch_mse_train, global_step=epoch)
        writer.add_scalar('Loss/Train', epoch_loss_train, global_step=epoch)
        print('train loss = ', epoch_loss_train, 'train mse = ',epoch_mse_train )
        
        #eval
        epoch_loss_test, epoch_mse_test = test_epoch(model, testloader,criterion, args)
        epoch_loss_list_test.append(epoch_loss_test)#delet if TB works
        epoch_mse_list_test.append(epoch_mse_test)#delet if TB works
        writer.add_scalar('MSE/Test', epoch_mse_test, global_step=epoch)
        writer.add_scalar('Loss/Test', epoch_loss_test, global_step=epoch)
        print('test loss = ', epoch_loss_test, 'test mse = ',epoch_mse_test )
        
        #plot and save
        noise_range = torch.logspace(0, 2.5, 4, device=args.device)
        visualize_point_denoising(writer, model, train_points, noise_range, epoch, "train")
        visualize_point_denoising(writer, model, test_points, noise_range, epoch, "test")

        torch.save(model.state_dict(), os.path.join(args.dir_name, 'model.pt'))


    print("--- %s seconds ---" % (time.time() - start_time_total))
    writer.close()
    
    return model