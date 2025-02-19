import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import os
from network import *

#@title Set up the SDE

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)
  
#diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

#@title Define the loss function

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


def run_training(score_model, marginal_prob_std_fn, trainloader, testloader, args):

  score_model = score_model.to(args.device)
  optimizer = Adam(score_model.parameters(), lr=args.lr)

  for epoch in range(args.num_epochs):
        avg_loss = 0.
        num_items = 0
        
        # Training loop
        score_model.train()
        for x in trainloader:  # Ignore labels
            if isinstance(x, (list, tuple)):
                x = x[0]  # Handle cases where dataset returns multiple items
            x = x.to(args.device)
            
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        
        # Print training progress
        print(f'Epoch {epoch}: Average Loss: {avg_loss / num_items:.5f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.dir_name, f'ckpt_{epoch+1}.pth')
            torch.save(score_model.state_dict(), ckpt_path)
            print(f'Saved checkpoint to {ckpt_path}')