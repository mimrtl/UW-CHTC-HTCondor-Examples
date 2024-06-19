
# MONAI Training Script - Version 2

This repository contains scripts to set up and train a medical image segmentation model using the MONAI framework. This version includes integration with Weights & Biases (wandb) for experiment tracking.

## Files

- `monai_train_v2_wb.py`: Python script for training a U-Net model using MONAI and wandb.
- `monai_train_v2_wb.sh`: Shell script to prepare data and run the training script with arguments.
- `monai_train_v2_wb.sub`: HTCondor job submission script to run the training on a high-performance computing cluster.
- `monai_train_v2_wb_args.txt`: Text file containing arguments for different training configurations.

## Prerequisites

- Python environment with required libraries (MONAI, PyTorch, Nibabel, Matplotlib, wandb)
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

2. **Weights & Biases Setup**

    Ensure you have a Weights & Biases account and generate an API key. Replace the placeholder key in the script with your actual key.

    ```python
    import wandb

    wandb.login(key="your_wandb_api_key")
    ```

3. **Training Script**

    The `monai_train_v2_wb.py` script includes the following steps:
    
    - Define data transformations and augmentations using MONAI.
    - Load and preprocess training data.
    - Define a U-Net model and loss function.
    - Track experiments with wandb.
    - Train the model, log metrics, and save the best model.
    
    Key wandb integration points:

    ```python
    import wandb

    # Login to wandb
    wandb.login(key="your_wandb_api_key")

    # Initialize wandb project
    wandb.init(
        project="WORD_UNet",
        config={
            "learning_rate": args.learning_rate,
            "architecture": "U-Net",
            "dataset": "WORD",
            "epochs": 3000,
            "batch_size": 4,
            "loss": "DiceCELoss",
            "model_depth": args.model_depth,
            "model_start_channels": args.model_start_channels,
            "model_num_res_units": args.model_num_res_units,
            "model_norm": args.model_norm,
            "model_act": "PReLU",
            "model_ordering": "NDA",
            "pixdim": (1.5, 1.5, 1.5),
            "roi_size": (160, 160, 80),
            "optimizer": "AdamW",
            "weight_decay": 1e-5,
            "CT_clip_range": (-1024, 2976),
            "CT_scale_range": (0.0, 1.0),
            "random_seed": 729,
        }
    )

    # Logging metrics
    wandb.log({"val_dice": mean_dice_val})
    wandb.log({"train_loss": current_epoch_loss, "average_grad": average_grad})
    ```

    To run the training script:

    ```bash
    python monai_train_v2_wb.py --learning_rate 1e-4 --model_depth 5 --model_start_channels 32 --model_num_res_units 4 --model_norm "INSTANCE"
    ```

### Cluster Setup

1. **Prepare Data**

    The `monai_train_v2_wb.sh` script handles data preparation and cleanup before and after training.
    
    ```bash
    pip list
    unzip ALTS.zip
    unzip WORD.zip
    rm WORD.zip ALTS.zip
    mv WORD ./ALTS/data_dir/
    cd ALTS

    # Run the Python script with the arguments passed to this shell script
    python monai_train_v2_wb.py --model_depth $1 --model_start_channels $2 --model_num_res_units $3

    tar_filename="proj_"$1"_"$2"_"$3"_"$4".tar.gz"

    # Tar the project directory and name it uniquely
    tar -czvf $tar_filename "ALTS/proj_dir/UNet_depth_${1}_channels_${2}_resunits_${3}"

    # Cleanup
    rm -rf ./data_dir/WORD
    cd ..
    rm -rf ALTS
    ```

2. **Submit Job to HTCondor**

    The `monai_train_v2_wb.sub` script is an HTCondor submission script that specifies the resources required and the executable to run.
    
    ```sub
    universe = container
    container_image = docker://docker.io/convez376/monai_with_wb:v1.0.2

    log = job_$(Cluster)_$(Process).log
    error = job_$(Cluster)_$(Process).err
    output = job_$(Cluster)_$(Process).out

    executable = monai_train_v2_wb.sh
    arguments = $(arg)

    should_transfer_files = YES
    when_to_transfer_output = ON_EXIT_OR_EVICT
    transfer_input_files = ALTS.zip, WORD.zip, monai_train_v2_wb.sh, monai_train_v2_wb_args.txt

    +WantGPULab = true
    +GPUJobLength = "medium"
    request_gpus = 1
    require_gpus = GlobalMemoryMb >= 15000
    request_cpus = 16
    request_memory = 80GB
    request_disk = 50GB

    queue arg from monai_train_v2_wb_args.txt
    ```

    Submit the job to HTCondor with:

    ```bash
    condor_submit monai_train_v2_wb.sub
    ```

## Hyperparameters

The training script logs hyperparameters used for the experiment, such as learning rate, model architecture, dataset, epochs, batch size, and optimizer settings. These are saved in a JSON file for future reference.

## Outputs

- The trained model and related metrics are saved in the specified project directory.
- Log files for the HTCondor job are generated for monitoring the job status.
- Experiment tracking and results are logged with wandb.

## Support

For further support, please contact: [wchen376@wisc.edu](mailto:wchen376@wisc.edu)
