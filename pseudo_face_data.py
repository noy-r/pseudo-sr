import os, sys
import numpy as np
import cv2
from numpy.lib.type_check import imag
from torchvision.transforms import Resize
# Import necessary modules from PyTorch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as nnF
import torch
from PIL import Image

import glob
# Define a custom dataset class that inherits from PyTorch's Dataset
class faces_data(Dataset):
    def __init__(self, data_lr, data_hr=None, b_train=True, rgb=True, img_range=1, shuffle=True, z_size=(16, 16)):
        # Determine if high-resolution images are available
        self.ready_hr = data_hr is not None
        # If high-resolution images are provided, find all .jpg files in the specified directory
        if data_hr is not None:
            self.hr_files = glob.glob(os.path.join(data_hr, "**/*.png"), recursive=True)
            self.hr_files.sort()
            if not self.hr_files:
                print(f"No high-resolution images found in {data_hr}")
        # Find all .jpg files in the low-resolution directory
        self.lr_files = glob.glob(os.path.join(data_lr, "*.png"), recursive=True)
        self.lr_files.sort()
        if not self.lr_files:
            print(f"No low-resolution images found in {data_lr}")
        # Shuffle file lists if specified
        if shuffle:
            if data_hr is not None:
                np.random.shuffle(self.hr_files)
            np.random.shuffle(self.lr_files)
        # Store parameters and flags
        self.training = b_train
        self.rgb = rgb
        self.z_size = z_size
        self.img_min_max = (0, img_range)
        # Define preprocessing transformations for training or evaluation
        if self.training:
            self.preproc = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)], p=0.5),
                transforms.Resize((64, 64)),
                transforms.ToTensor()])

        else:
            self.preproc = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()])
    # Return the number of samples in the dataset

    def _find_image_files(self, directory, extension):
        print(f"Searching for {extension} files in directory: {directory}")
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
        return files
    def __len__(self):
        if self.ready_hr:
            return min([len(self.lr_files), len(self.hr_files)])
        else:
            return len(self.lr_files)
    # Get a data sample by index
    def __getitem__(self, index):
        data = dict()
        if np.prod(self.z_size) > 0:
            data["z"] = torch.randn(1, *self.z_size, dtype=torch.float32)

        lr_idx = index % len(self.lr_files)
        lr = cv2.imread(self.lr_files[lr_idx])
        if self.rgb:
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        data["lr"] = self.preproc(lr) * self.img_min_max[1]
        data["lr_path"] = self.lr_files[lr_idx]
        data["lr_upsample"] = nnF.interpolate(data["lr"].unsqueeze(0), scale_factor=2, mode="bicubic", align_corners=False).clamp(min=self.img_min_max[0], max=self.img_min_max[1]).squeeze(0)
        if self.ready_hr:
            hr_idx = index % len(self.hr_files)
            hr = cv2.imread(self.hr_files[hr_idx])
            if self.rgb:
                hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            hr_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),  # High resolution size
                transforms.ToTensor()])
            #data["hr"] = self.preproc(hr) * self.img_min_max[1]
            data["hr"] = hr_transform(hr) * self.img_min_max[1]
            data["hr_path"] = self.hr_files[hr_idx]

            hr_down_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),  # Downsampled high resolution size
                transforms.ToTensor()])
            data["hr_down"] = hr_down_transform(hr) * self.img_min_max[1]
            #data["hr_down"] = nnF.interpolate(data["hr"].unsqueeze(0), scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=False).clamp(min=self.img_min_max[0], max=self.img_min_max[1]).squeeze(0)
        return data
    def get_noises(self, n):
        return torch.randn(n, 1, *self.z_size, dtype=torch.float32)
    # Shuffle the high- and low-resolution datasets independently
    def permute_data(self):
        if self.ready_hr:
            np.random.shuffle(self.hr_files)
        np.random.shuffle(self.lr_files)

# Example usage: Load data, preprocess it, and visualize
if __name__ == "__main__":
    # Set paths to high- and low-resolution training data folders
    if "DATA_TRAIN" not in os.environ:
        os.environ["DATA_TRAIN"] = "/Users/noymachluf/Desktop/pseudo-sr2/dataset/train"
    high_folder = os.path.join(os.environ["DATA_TRAIN"], "HIGH")
    print(high_folder)
    low_folder = os.path.join(os.environ["DATA_TRAIN"], "LOW/wider_lnew")
    print(low_folder)
    if "DATA_TEST" not in os.environ:
        os.environ["DATA_TEST"] = "/Users/noymachluf/Desktop/pseudo-sr2/dataset/testset"
    test_folder = os.path.join(os.environ["DATA_TEST"])
    # Set image scaling range (0 to 1)
    img_range = 1
    # Initialize the dataset with specified folders and parameters
    data = faces_data(low_folder, high_folder, img_range=img_range)
    # Iterate through the dataset to visualize each data sample
    for i in range(len(data)):
        d = data[i]
        for elem in d:
            # Skip noise and file path elements
            if elem in ['z', 'lr_path', 'hr_path']: continue
            print(f"{elem} shape:", d[elem].shape)
            # Convert tensor back to image for visualization
            img = np.around((d[elem].numpy().transpose(1, 2, 0) / img_range) * 255.0).astype(np.uint8)
            cv2.imshow(elem, img[:, :, ::-1])
        cv2.waitKey()
    print("fin.")






