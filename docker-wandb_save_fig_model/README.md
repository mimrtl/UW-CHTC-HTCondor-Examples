# Tutorial: Using Weights and Biases (wandb) for Code, Figures, and Model Logging

This tutorial provides a step-by-step guide for setting up and using the `wandb` package to log code, figures, and models. It demonstrates setting up a custom cache directory for `wandb` data (useful when running inside containers), initializing `wandb`, logging code, plotting and saving figures, and finally uploading model weights. Additionally, a section on HTCondor is provided for managing remote access, job submission, and file transfer on high-throughput computing systems.

## Table of Contents
1. [Basic Setup and Initialization](#basic-setup-and-initialization)
2. [Logging Code](#logging-code)
3. [Logging Figures](#logging-figures)
4. [Saving and Uploading Model Weights](#saving-and-uploading-model-weights)
5. [HTCondor Guide for Remote Access and Job Submission](#htcondor-guide-for-remote-access-and-job-submission)

---

### 1. Basic Setup and Initialization

Inside containerized environments, the default `wandb` directories might be inaccessible. This section demonstrates how to define and create custom directories for `wandb` logging, followed by initializing a `wandb` run.

```python
import os
import wandb

# Define and create custom cache directories for wandb
base_cache_dir = './cache'
cache_dirs = {
    'WANDB_DIR': os.path.join(base_cache_dir, 'wandb'),
    'WANDB_DATA_DIR': os.path.join(base_cache_dir, 'wandb_data'),
    'WANDB_CACHE_DIR': os.path.join(base_cache_dir, 'wandb_cache'),
    'WANDB_CONFIG_DIR': os.path.join(base_cache_dir, 'config'),
    'TRANSFORMERS_CACHE': os.path.join(base_cache_dir, 'transformers'),
    'MPLCONFIGDIR': os.path.join(base_cache_dir, 'mplconfig')
}

# Create directories and set environment variables
os.makedirs(base_cache_dir, exist_ok=True)
for key, path in cache_dirs.items():
    os.makedirs(path, exist_ok=True)
    os.environ[key] = path

# Initialize wandb with custom directories
wandb.login(key="your_wandb_key")
wandb_run = wandb.init(
    project="your_project_name",
    dir=os.getenv("WANDB_DIR", "./cache/wandb"),
    config={
        "msg": "Hello from WandB!",
        "WANDB_DIR": os.getenv("WANDB_DIR"),
        "WANDB_CACHE_DIR": os.getenv("WANDB_CACHE_DIR"),
        "WANDB_CONFIG_DIR": os.getenv("WANDB_CONFIG_DIR"),
        "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
        "MPLCONFIGDIR": os.getenv("MPLCONFIGDIR")
    }
)
```

### 2. Logging Code

You can log specific code files or directories to keep a record of the exact code used during an experiment. This can help with experiment reproducibility.

```python
# Log all Python files in the current directory
wandb_run.log_code(root=".", include_globs=["*.py"])
```

```python
# Log one specific Python file
wandb_run.log_code(root=".", name="try_wandb.py")
```

### 3. Logging Figures

There are two main ways to log figures: using custom plots created with `matplotlib` and uploading images directly.

#### Custom Plotting with `matplotlib`

The example function below generates 3D slices of two arrays, `x` and `xrec`, and logs the plot as an image. 

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_x_xrec(x, xrec, num_per_direction=1, savename=None, wandb_commitment=True):
    x = x[0, 0, :, :, :]
    xrec = xrec[0, 0, :, :, :]
    x_clip = np.clip(x, 0, 1)
    rec_clip = np.clip(xrec, 0, 1)
    fig_width = num_per_direction * 3
    fig_height = 4
    fig, axs = plt.subplots(3, fig_width, figsize=(fig_width, fig_height), dpi=100)
    
    # Plot images across three anatomical directions
    for i in range(num_per_direction):
        # Axial
        img_x = x_clip[x_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        img_rec = rec_clip[rec_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        axs[0, 3*i].imshow(img_x, cmap="gray")
        axs[1, 3*i].imshow(img_rec, cmap="gray")
        axs[2, 3*i].imshow(img_x - img_rec, cmap="bwr")
        
        # Save and log the plot
    plt.tight_layout()
    plt.savefig(savename)
    wandb_run.log({"val_snapshots": fig}, commit=wandb_commitment)
    plt.close()
    print(f"Save the plot to {savename}")

# Test with random data and log plot
x = np.random.rand(1, 1, 64, 64, 64)
xrec = np.random.rand(1, 1, 64, 64, 64)
plot_and_save_x_xrec(x, xrec, num_per_direction=1, savename="x_xrec_plot.png")
```

#### Logging Built-In Plots with `wandb`

You can also use `wandb`’s built-in functions to create histograms and upload image files directly:

```python
# Upload pre-saved image
wandb.upload_file("x_xrec_plot.png", root=".")
wandb.log({"val_snapshots": wandb.Image("x_xrec_plot.png")})

# Log a histogram of values in x
wandb_run.log({"gradients": wandb.Histogram(np.ravel(x))})
```

### 4. Saving and Uploading Model Weights

To store and share model weights, save the model locally and log it to `wandb` as an artifact.

```python
import monai
import torch

# Build a simple UNet model
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(4, 8, 16),
    strides=(2, 2),
    num_res_units=2
)

# Save model weights
torch.save(model.state_dict(), "try_wandb_model.pth")

# Log model weights to wandb
wandb_run.log_model(
    path="try_wandb_model.pth",
    name="my_model_artifact",
    aliases=["production"],
)

print("Done!")
wandb.finish()
```

With this setup, `wandb` will capture the experiment’s code, figures, and model weights, enabling streamlined experiment tracking and reproducibility.


### 5. HTCondor Guide for Remote Access and Job Submission

For projects that require remote access, job management, and file transfer on a high-throughput computing system (HTCondor), a guide is available in HTCondor_101.md. This file includes commands for:

- Remote Access: Securely connect to HTCondor servers.
File Transfer: Transfer files between your local machine and the HTCondor server.
- Job Management with HTCondor: Submit, monitor, and manage jobs on the HTCondor system.
- File Compression and Decompression: Compress and decompress files as part of data preparation or job management.

Refer to HTCondor_101.md for a detailed command list and examples for working with HTCondor.