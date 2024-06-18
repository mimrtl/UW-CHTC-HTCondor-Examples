import os
import wandb

wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")

# Set the environment variable for the Matplotlib configuration directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'

# Continue with your other configurations
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    RandSpatialCropd,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
)

import torch

import numpy as np
import nibabel as nib
from matplotlib.colors import ListedColormap

import getpass

# Now proceed with your regular imports and script logic
import matplotlib.pyplot as plt

# print_config()
import argparse

# CLI argument setup
parser = argparse.ArgumentParser(description="Run UNet model training with configurable parameters.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
parser.add_argument("--model_depth", type=int, default=5, help="Depth of the UNet model")
parser.add_argument("--model_start_channels", type=int, default=16, help="Initial number of channels in the UNet model")
parser.add_argument("--model_num_res_units", type=int, default=2, help="Number of residual units in the UNet model")
parser.add_argument("--model_norm", type=str, default="INSTANCE", help="Normalization type for the UNet model")
args = parser.parse_args()

# print the parsed arguments
print("Parsed arguments:")
print("Learning rate:", args.learning_rate)
print("Model depth:", args.model_depth)
print("Model start channels:", args.model_start_channels)
print("Model number of residual units:", args.model_num_res_units)
print("Model normalization type:", args.model_norm)
print("=====================================")

data_dir = "data_dir/WORD"
# root_dir = f"proj_dir/UNet_lr_{args.learning_rate}_depth_{args.model_depth}_channels_{args.model_start_channels}_resunits_{args.model_num_res_units}_norm_{args.model_norm}"
root_dir = f"proj_dir/UNet_depth_{args.model_depth}_channels_{args.model_start_channels}_resunits_{args.model_num_res_units}"

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# creat root directory if not exists
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=2976,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # NormalizeIntensityd(subtrahend=mean_of_dataset, divisor=std_dev_of_dataset),  # mean and std_dev should be precomputed from your dataset
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # random crop to the target size of (160, 160, 80)
        RandSpatialCropd(keys=["image", "label"], roi_size=(160, 160, 80), random_center=True, random_size=False),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2976, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandSpatialCropd(keys=["image", "label"], roi_size=(160, 160, 80), random_center=True, random_size=False),
    ]
)


import json
import glob

data_json_path = os.path.join(data_dir, "dataset.json")
# load data json
with open(data_json_path, "r") as f:
    data_json = json.load(f)
    
# for key in data_json.keys():
#     print(f"{key}: {data_json[key]}")

training_files = data_json["training"]

# add data directory to each filename
training_files = [{"image": os.path.join(data_dir, d["image"]), "label": os.path.join(data_dir, d["label"])} for d in training_files]
validation_folders = data_json["validation"]
validation_files = []

# search for the file with the same file name in the validation_files
validation_candidates = sorted(glob.glob(os.path.join(data_dir, validation_folders[0], "*.nii.gz")))
for candidate in validation_candidates:
    label_candidate = os.path.join(data_dir, validation_folders[1], os.path.basename(candidate))
    if os.path.exists(label_candidate):
        validation_files.append({"image": candidate, "label": label_candidate})

print(training_files)
print(validation_files)
        

numTraining = len(training_files)
numValidation = len(validation_files)

numClasses = len(data_json["labels"].keys())
print("Train:", numTraining, "Validation:", numValidation, "Classes:", numClasses)


# data_dir = "/dataset/"
# split_json = "dataset_0.json"

# datasets = data_dir + split_json
# datalist = load_decathlon_datalist(datasets, True, "training")
# val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=training_files,
    transform=train_transforms,
    cache_num=numTraining,
    cache_rate=0.16,
    num_workers=6,
)
val_ds = CacheDataset(
    data=validation_files, 
    transform=val_transforms, 
    cache_num=numValidation,
    cache_rate=0.2, 
    num_workers=2)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=numClasses,  # Assuming numClasses is defined somewhere
    channels=[args.model_start_channels * (2 ** i) for i in range(args.model_depth)],
    strides=tuple([2] * (args.model_depth - 1)),
    num_res_units=args.model_num_res_units,
    norm=args.model_norm
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)



def plot_image(image, label, output, root_dir, epoch, mean_dice):

    # colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
    #           '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']
    colors = ['#000000',  # Black for background
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', 
            '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', 
            '#800000', '#aaffc3']
    custom_cmap = ListedColormap(colors)

    # plot the middle slice of the image
    idx = image.shape[4] // 2

    plt.figure(figsize=(18, 6), dpi=300)
    plt.subplot(131)
    img_to_plot = np.squeeze(np.rot90(image[:, :, :, :, idx], 3))
    plt.imshow(img_to_plot, cmap="gray")
    plt.axis("off")
    plt.title("image")

    plt.subplot(132)
    img_to_plot = np.squeeze(np.rot90(label[:, :, :, :, idx], 3))
    plt.imshow(img_to_plot, cmap=custom_cmap)
    plt.axis("off")
    plt.title("label")

    plt.subplot(133)
    img_to_plot = np.squeeze(np.rot90(output[:, :, :, :, idx], 3))
    # img_to_plot is [17, 160, 160] as one-hot
    # convert to [160, 160] as label from 0 to 16
    img_to_plot = np.argmax(img_to_plot, axis=0)
    plt.imshow(img_to_plot, cmap=custom_cmap)
    plt.axis("off")
    plt.title("output")

    plt.title(f"Epoch: {epoch}, Mean Dice: {mean_dice}")
    save_path = os.path.join(root_dir, f"epoch_{epoch:.0f}_mean_dice_{mean_dice:.4f}.png")
    plt.savefig(save_path)
    plt.close()

def validation(epoch, epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = model(val_inputs)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            # epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038

        image = batch["image"].detach().cpu()
        label = batch["label"].detach().cpu()
        output = val_outputs.detach().cpu()
        mean_dice_val = dice_metric.aggregate().item()
        wandb.log({"val_dice": mean_dice_val})
        plot_image(image, label, output, root_dir, epoch, mean_dice_val)
        # print mean dice
        print("Epoch %d Validation Dice: %f" % (epoch, mean_dice_val))
        dice_metric.reset()
    return mean_dice_val

# Function to get memory in GB
def get_gpu_memory_map():
    """Get the current gpu usage."""
    assert torch.cuda.is_available(), "CUDA is not available. No GPU found!"
    
    # Collects GPU details
    device_count = torch.cuda.device_count()
    details = {}
    for i in range(device_count):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        details[f'GPU {i}'] = f'Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB'
    return details

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    sum_grad = 0
    num_params = 0
    # epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(train_loader):
        print("Step:", step, "Global Step:", global_step, "Max Iterations:", max_iterations)
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("NaN or Inf found in input tensor")
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("NaN or Inf found in target tensor")
        
        # print("Input shape:", x.shape, "Target shape:", y.shape)
        # print("Input dtype:", x.dtype, "Target dtype:", y.dtype)
        optimizer.zero_grad()
        logit_map = model(x)
        gpu_memory = get_gpu_memory_map()
        for gpu, memory in gpu_memory.items():
            print(f'GPU {gpu}: {memory}')
        # print("Output shape:", logit_map.shape, "Output dtype:", logit_map.dtype)
        loss = loss_function(logit_map, y)
        
        loss.backward()
        epoch_loss += loss.item()

        # Compute average absolute gradient
        for param in model.parameters():
            if param.grad is not None:
                sum_grad += param.grad.abs().mean().item()
                num_params += 1

        optimizer.step()
        
        # epoch_iterator.set_description(  # noqa: B038
        #     "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        # )
        print("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            # epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(global_step, val_loader)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    
    average_grad = sum_grad / num_params if num_params > 0 else 0
    current_epoch_loss = epoch_loss / step
    wandb.log({"train_loss": current_epoch_loss, "average_grad": average_grad})

    return global_step, dice_val_best, global_step_best

# set random seed as 729
torch.manual_seed(729)
np.random.seed(729)

print("Wandb path:", wandb.__path__)
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="WORD_UNet",

    # track hyperparameters and run metadata
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

# CLI controlles
# "learning_rate",
# "model_depth",
# "model_start_channels",
# "model_num_res_units",
# "model_norm",



max_iterations = 3000
eval_num = 50
post_label = AsDiscrete(to_onehot=numClasses)
post_pred = AsDiscrete(argmax=True, to_onehot=numClasses)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
np.save(os.path.join(root_dir, "epoch_loss_values.npy"), epoch_loss_values)
np.save(os.path.join(root_dir, "metric_values.npy"), metric_values)
wandb.finish()