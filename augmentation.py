
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import torch
import torchvision.transforms.functional as TF
import random
import torchvision

# Define transformation pipeline without normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
])

# Specify the path to your original images folder - change this to your path!
original_data_path = r'/Users/noymachluf/Desktop/pseudo-sr2/Data'


# Create the ImageFolder dataset for the original images
original_dataset = ImageFolder(root=original_data_path, transform=transform)

# Create a data loader for the original dataset
original_data_loader = DataLoader(original_dataset, batch_size=1, shuffle=True)




# Augmentations with zoom and translation
augmentations = [
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
    transforms.RandomRotation(90),  # Randomly rotate the image by a maximum of 90 degrees
    transforms.RandomRotation(45),  # Randomly rotate the image by a maximum of 45 degrees
    transforms.RandomRotation(135),  # Randomly rotate the image by a maximum of 90 degrees
    # Random zoom
    transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.8, 1.2), shear=0),
    transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.5, 0.8), shear=0),
    transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(1.5, 1.7), shear=0),
    # Random translation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(1, 1), shear=0),
    transforms.RandomAffine(degrees=0, translate=(0.25, 0.25), scale=(1, 1), shear=0),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(1, 1), shear=0),
]

# Create the output folder for augmented images one directory above the original data folder
output_folder_aug = os.path.join(os.path.dirname(original_data_path), 'augmented')
if not os.path.exists(output_folder_aug):
    os.makedirs(output_folder_aug)



# Apply all augmentations to all the images in the dataset
for i, (image, label) in enumerate(original_data_loader):
    for j, aug in enumerate(augmentations):
        augmented_image = aug(image)  # Apply augmentation which should return a tensor in [0, 1]
        # Convert tensor to PIL image
        augmented_image = transforms.ToPILImage()(augmented_image.squeeze())
        # Save the image
        augmented_image.save(os.path.join(output_folder_aug, f'image_{i}_aug_{j}.png'))
        print(f'Image {i} augmented with augmentation {j} and saved to disk')





