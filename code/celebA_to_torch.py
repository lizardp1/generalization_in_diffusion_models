import numpy as np
import os
import time
import torch
from dataloader_func import load_Reach_dataset, patch_Reach_dataset,prep_Reach_patches, split_dataset, load_CelebA_dataset, prep_celeba
from quality_metrics_func import remove_repeats
from plotting_func import plot_many_denoised,  show_im_set
import matplotlib.pyplot as plt

def show_images(dataset, num_images=10, save_path='sample_images.png'):
    # Extract the first num_images images from the dataset
    sample_images = dataset[:num_images]
    
    # Convert the tensor to numpy (you may need to convert to CPU if it's on GPU)
    sample_images_np = sample_images.permute(0, 2, 3, 1).cpu().numpy()
    
    # Plot the images in a grid
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
    
    for i in range(num_images):
        # If the images are grayscale (i.e. single channel), remove the last channel dim
        if sample_images_np.shape[-1] == 1:
            image = sample_images_np[i, :, :, 0]
            axes[i].imshow(image, cmap='gray')
        else:
            axes[i].imshow(sample_images_np[i])
        axes[i].axis('off')  # Turn off axis for clearer display
    
    # Save the figure to a file
    plt.savefig(save_path)
    print(f'Sample images saved to {save_path}')


def main():
    start_time_total = time.time()
    
    source_folder_path = 'datasets/img_align_celeba/img_align_celeba'
    train_folder_path = 'train/img_align_celeba/'
    test_folder_path = 'test/'

    #split_dataset(source_folder_path, train_folder_path, test_folder_path)

    all_ims = load_CelebA_dataset( train_folder_path, test_folder_path, s=.5)
    train_set, test_set  = prep_celeba(all_ims)
        
    print('train: ', train_set.shape )
    print('test: ', test_set.shape )

    show_images(train_set, num_images=10)

    # remove repeated images 
    #data_cleaned = remove_repeats(train_set, threshold=0.95, chunk_size=500, batch_size=100)
    #print('shape after removed repeats:' , data_cleaned.shape)
    
    torch.save(train_set, 'train80x80_no_repeats.pt', pickle_protocol=4)
    torch.save(test_set, 'test80x80.pt', pickle_protocol=4)
        
    print("--- %s seconds ---" % (time.time() - start_time_total))

    

if __name__ == "__main__" :
    main()    
    
    
    
