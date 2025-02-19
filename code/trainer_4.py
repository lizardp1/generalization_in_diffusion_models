import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import matplotlib.pyplot as plt

def add_noise(x, noise_range, device):
    """Add Gaussian noise to the data"""
    noise_level = torch.rand(x.size(0), 1, device=device) * (noise_range[1] - noise_range[0]) + noise_range[0]
    noise = torch.randn_like(x) * noise_level
    return x + noise, noise

def denoise_points(model, noisy_points, noise_level):
    """
    Denoise points using score-based method
    Uses Euler-Maruyama solver for simplicity
    """
    dt = 0.1  # step size
    steps = 100  # number of steps
    x = noisy_points.clone()
    
    for _ in range(steps):
        score = model(x)
        x = x + dt * (noise_level**2 * score)
    
    return x

def compute_l2_distance(clean, denoised):
    """Compute average L2 distance between clean and denoised points"""
    return torch.norm(clean - denoised, dim=1).mean()

def plot_scores(model, data, noise_level, device, epoch, writer):
    """Plot the predicted scores on a grid"""
    model.eval()
    with torch.no_grad():
        # Create a grid of points
        x = np.linspace(-4, 4, 50)
        y = np.linspace(-4, 4, 50)
        X, Y = np.meshgrid(x, y)
        points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), device=device, dtype=torch.float32)
        
        # Add noise and compute scores
        noisy_points = points + torch.randn_like(points) * noise_level
        scores = model(noisy_points)
        
        # Denoise points
        denoised_points = denoise_points(model, noisy_points, noise_level)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Score field plot
        ax1.quiver(points[:, 0].cpu(), points[:, 1].cpu(), 
                  scores[:, 0].cpu(), scores[:, 1].cpu(),
                  alpha=0.5)
        ax1.set_title(f'Score Field (Noise Level: {noise_level:.2f})')
        
        # Points plot
        ax2.scatter(points[:, 0].cpu(), points[:, 1].cpu(), alpha=0.3, label='Original')
        ax2.scatter(noisy_points[:, 0].cpu(), noisy_points[:, 1].cpu(), alpha=0.3, label='Noisy')
        ax2.scatter(denoised_points[:, 0].cpu(), denoised_points[:, 1].cpu(), alpha=0.3, label='Denoised')
        ax2.legend()
        ax2.set_title('Point Denoising Visualization')
        
        # Save to tensorboard
        writer.add_figure(f'visualization/noise_{noise_level:.2f}', fig, epoch)
        plt.close()

def one_iter(model, batch, criterion, args):
    """Single training iteration"""
    # Get data and move to device
    clean = batch[0].to(args.device)
    
    # Add noise
    noisy, noise = add_noise(clean, args.noise_level_range, args.device)
    
    # Compute score
    pred_score = model(noisy)
    
    # True score is negative noise / (noise_level^2)
    noise_level = torch.norm(noise, dim=1, keepdim=True)
    true_score = -noise / (noise_level**2 + 1e-5)
    
    # Compute loss
    loss = criterion(pred_score, true_score)
    
    # Compute metrics
    with torch.no_grad():
        cos_sim = nn.functional.cosine_similarity(pred_score, true_score).mean()
        # Denoise points and compute L2 distance
        denoised = denoise_points(model, noisy, noise_level.mean())
        l2_dist = compute_l2_distance(clean, denoised)
    
    return model, loss / clean.size(0), cos_sim, l2_dist

def train_epoch(model, trainloader, criterion, optimizer, args):
    """Train for one epoch"""
    loss_sum = 0
    cos_sim_sum = 0
    l2_dist_sum = 0
    model.train()
    
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()
        model, loss, cos_sim, l2_dist = one_iter(model, batch, criterion, args)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        cos_sim_sum += cos_sim.item()
        l2_dist_sum += l2_dist.item()
    
    return model, loss_sum/(i+1), cos_sim_sum/(i+1), l2_dist_sum/(i+1)

def test_epoch(model, testloader, criterion, args):
    """Evaluate on test set"""
    loss_sum = 0
    cos_sim_sum = 0
    l2_dist_sum = 0
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            model, loss, cos_sim, l2_dist = one_iter(model, batch, criterion, args)
            loss_sum += loss.item()
            cos_sim_sum += cos_sim.item()
            l2_dist_sum += l2_dist.item()
    
    return loss_sum/(i+1), cos_sim_sum/(i+1), l2_dist_sum/(i+1)

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
        model, train_loss, train_cos, train_l2 = train_epoch(model, trainloader, criterion, optimizer, args)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('CosSim/Train', train_cos, epoch)
        writer.add_scalar('L2_Distance/Train', train_l2, epoch)
        print(f'Train - Loss: {train_loss:.4f}, Cos Sim: {train_cos:.4f}, L2 Dist: {train_l2:.4f}')
        
        # Evaluate
        test_loss, test_cos, test_l2 = test_epoch(model, testloader, criterion, args)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('CosSim/Test', test_cos, epoch)
        writer.add_scalar('L2_Distance/Test', test_l2, epoch)
        print(f'Test - Loss: {test_loss:.4f}, Cos Sim: {test_cos:.4f}, L2 Dist: {test_l2:.4f}')
        
        # Visualize at different noise levels
        if epoch % 10 == 0:
            for noise_level in [0.1, 0.5, 1.0]:
                plot_scores(model, train_data, noise_level, args.device, epoch, writer)
        
        # Save model
        torch.save(model.state_dict(), f'{args.dir_name}/model_epoch_{epoch}.pt')
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    writer.close()
    
    return model