
# MONAI Training Script

This repository contains scripts to set up and train a medical image segmentation model using the MONAI framework.

## Files

- `monai_train_v1.py`: Python script for training a U-Net model using MONAI.
- `monai_train_v1.sh`: Shell script to prepare data and run the training script.
- `monai_train_v1.sub`: HTCondor job submission script to run the training on a high-performance computing cluster.

## Prerequisites

- Python environment with required libraries (MONAI, PyTorch, Nibabel, Matplotlib)
- Docker
- HTCondor (if running on a cluster)

## Setup and Training

### Local Setup

1. **Prepare the Environment**

    Set up the required environment variables and ensure necessary directories are created.

    ```python
    import os

    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    ```

2. **Training Script**

    The `monai_train_v1.py` script includes the following steps:
    
    - Define data transformations and augmentations using MONAI.
    - Load and preprocess training data.
    - Define a U-Net model and loss function.
    - Train the model, log metrics, and save the best model.
    
    To run the training script:
    
    ```bash
    python monai_train_v1.py
    ```

### Cluster Setup

1. **Prepare Data**

    The `monai_train_v1.sh` script handles data preparation and cleanup before and after training.
    
    ```bash
    unzip ALTS.zip
    unzip WORD.zip
    rm WORD.zip ALTS.zip
    mv WORD ./ALTS/data_dir/
    cd ALTS
    python monai_train_v1.py
    rm -rf ./data_dir/WORD
    cd ../
    tar -czvf proj_d5f48r5e20000.tar.gz ALTS/proj_dir/
    rm -rf ALTS
    ```

2. **Submit Job to HTCondor**

    The `monai_train_v1.sub` script is an HTCondor submission script that specifies the resources required and the executable to run.
    
    ```sub
    container_image = docker://projectmonai/monai:latest
    universe = container

    log = job_$(Cluster)_$(Process).log
    error = job_$(Cluster)_$(Process).err
    output = job_$(Cluster)_$(Process).out

    executable = monai_train_v1.sh

    should_transfer_files = YES
    when_to_transfer_output = ON_EXIT_OR_EVICT
    transfer_input_files = ALTS.zip, WORD.zip, monai_train_v1.sh

    +WantGPULab = true
    +GPUJobLength = "long"
    request_gpus = 1
    require_gpus = GlobalMemoryMb >= 20000
    request_cpus = 16
    request_memory = 80GB
    request_disk = 50GB

    queue
    ```

    Submit the job to HTCondor with:

    ```bash
    condor_submit monai_train_v1.sub
    ```

## Hyperparameters

The training script logs hyperparameters used for the experiment, such as learning rate, model architecture, dataset, epochs, batch size, and optimizer settings. These are saved in a JSON file for future reference.

## Outputs

- The trained model and related metrics are saved in the specified project directory.
- Log files for the HTCondor job are generated for monitoring the job status.

## Support

For further support, please contact: [wchen376@wisc.edu](mailto:wchen376@wisc.edu)
