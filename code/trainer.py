import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import matplotlib.pyplot as plt


def add_noise(x, noise_range, device, noise_type='gaussian', model=None, t=None):
    """
    noise_range: [min_noise, max_noise] for noise level
    noise_type: Type of noise to add ('gaussian', 'uniform', 'poisson', 'saltpepper', 'exponential')
    """
    noise_level = torch.rand(x.size(0), 1, device=device) * (noise_range[1] - noise_range[0]) + noise_range[0]
    
    if noise_type == 'gaussian':
        noise = torch.randn_like(x) * noise_level.expand_as(x)
        
    elif noise_type == 'uniform':
        noise = (2.0 * torch.rand_like(x) - 1.0) * noise_level.expand_as(x)
        
    elif noise_type == 'poisson':
        # Scale data to positive values if needed
        scaled_x = x - x.min() + 1e-3
        # Poisson noise scales with signal intensity
        rate = scaled_x / noise_level.expand_as(x)
        noise = torch.poisson(rate) * noise_level.expand_as(x) - rate

    elif noise_type == 'laplace':
        # Laplace(0, b) with b = noise_level
        # Use PyTorch distributions if available, or do it manually
        laplace_dist = torch.distributions.Laplace(loc=0.0, scale=1.0)
        base_noise = laplace_dist.sample(x.shape).to(device)
        noise = base_noise * noise_level

    elif noise_type == 'langevin':
        if model is None:
            raise ValueError("Model required for Langevin sampling")

        drift_weight = 1.0
        diffusion_decay = 0.995
        min_temp = 0.1
            
        with torch.no_grad():
            score = (x - model(x)) / (noise_level.expand_as(x))**2  #score estimated from denoiser
            current_temp = min_temp if t is None else max(min_temp, diffusion_decay**t)
            
            drift = drift_weight * noise_level.expand_as(x) * score
            diffusion = current_temp * torch.sqrt(noise_level).expand_as(x) * torch.randn_like(x)
            noise = drift + diffusion
        
    elif noise_type == 'saltpepper':
        noise = torch.zeros_like(x)
        probability = noise_level.expand_as(x)
        
        # Salt noise (white pixels)
        salt_mask = torch.rand_like(x) < (probability / 2)
        noise[salt_mask] = 1.0
        
        # Pepper noise (black pixels)
        pepper_mask = torch.rand_like(x) < (probability / 2)
        noise[pepper_mask] = -1.0
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return x + noise, noise, noise_level


def plot_denoising_results(model, data, noise_level, device, epoch, writer):
    """Plot original, noisy, and denoised points"""
    model.eval()
    with torch.no_grad():
        # Add noise
        noisy_points, noise, _ = add_noise(data, [noise_level, noise_level], device)
        
        # Get denoised points
        denoised_points = model(noisy_points)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot points
        ax1.scatter(data[:, 0].cpu(), data[:, 1].cpu(), alpha=0.5, label='Original')
        ax1.scatter(noisy_points[:, 0].cpu(), noisy_points[:, 1].cpu(), alpha=0.5, label='Noisy')
        ax1.scatter(denoised_points[:, 0].cpu(), denoised_points[:, 1].cpu(), alpha=0.5, label='Denoised')
        ax1.legend()
        ax1.set_title(f'Point Denoising (σ={noise_level:.2f})')
        
        # Plot error vectors
        errors = denoised_points - data
        ax2.quiver(data[:, 0].cpu(), data[:, 1].cpu(),
                  errors[:, 0].cpu(), errors[:, 1].cpu(),
                  alpha=0.5)
        ax2.set_title('Denoising Error Vectors')
        
        # Save to tensorboard
        writer.add_figure(f'denoising/noise_{noise_level:.2f}', fig, epoch)
        plt.close()

def one_iter(model, batch, criterion, args, iter_count=None):
    """Single training iteration"""
    # Get data and move to device
    clean = batch[0].to(args.device)
    
    noisy, noise, noise_levels = add_noise(
        clean, 
        args.noise_level_range, 
        args.device, 
        args.noise_type,
        model=model if args.noise_type == 'langevin' else None,
        t=iter_count
    )
    # Get denoised points
    denoised = model(noisy)
    
    # Compute loss
    loss = criterion(denoised, clean)
    
    # Compute metrics
    with torch.no_grad():
        l2_dist = torch.norm(clean - denoised, dim=1).mean()
    
    return model, loss / clean.size(0), l2_dist

def train_epoch(model, trainloader, criterion, optimizer, args, start_iter=0):
    """Train for one epoch"""
    loss_sum = 0
    l2_dist_sum = 0
    model.train()

    curr_iter = start_iter
    
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()
        model, loss, l2_dist = one_iter(model, batch, criterion, args, curr_iter)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        l2_dist_sum += l2_dist.item()

        curr_iter += 1
    
    return model, loss_sum/(i+1), l2_dist_sum/(i+1), curr_iter

def test_epoch(model, testloader, criterion, args):
    """Evaluate on test set"""
    loss_sum = 0
    l2_dist_sum = 0
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            model, loss, l2_dist = one_iter(model, batch, criterion, args)
            loss_sum += loss.item()
            l2_dist_sum += l2_dist.item()
    
    return loss_sum/(i+1), l2_dist_sum/(i+1)

def run_training(model, trainloader, testloader, criterion, optimizer, args):
    """Main training loop"""
    writer = SummaryWriter(log_dir=args.dir_name)
    start_time = time.time()
    
    # Get sample of training data for visualization
    train_data = next(iter(trainloader))[0].to(args.device)
    
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch}')
        
        # Learning rate scheduling
        if epoch >= args.lr_freq and epoch % args.lr_freq == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        # Train
        model, train_loss, train_l2, total_iters = train_epoch(model, trainloader, criterion, optimizer, args)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('L2_Distance/Train', train_l2, epoch)
        print(f'Train - Loss: {train_loss:.4f}, L2 Dist: {train_l2:.4f}')
        
        # Evaluate
        test_loss, test_l2 = test_epoch(model, testloader, criterion, args)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('L2_Distance/Test', test_l2, epoch)
        print(f'Test - Loss: {test_loss:.4f}, L2 Dist: {test_l2:.4f}')
        
        # Visualize denoising at different noise levels
        if epoch % 10 == 0:
            for noise_level in [0.1, 0.5, 1.0]:
                plot_denoising_results(model, train_data, noise_level, args.device, epoch, writer)
        
        # Save model
        torch.save(model.state_dict(), f'{args.dir_name}/denoiser_epoch_{epoch}.pt')
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    writer.close()
    
    return model

def evaluate_noise_levels(model, data, noise_levels, device):
    """Evaluate denoising performance across noise levels"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for noise_level in noise_levels:
            noisy, _, _ = add_noise(data, [noise_level, noise_level], device)
            denoised = model(noisy)
            l2_dist = torch.norm(data - denoised, dim=1).mean()
            results.append((noise_level, l2_dist.item()))
    
    return results