import os
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode
import argparse
from models.pseudo_model import Pseudo_Model
from tools.pseudo_face_data import faces_data
from torchvision import transforms
from PIL import Image
from models.face_model import Face_Model
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def display_images(images, title):
    """Displays a batch of images using matplotlib."""
    fig, axs = plt.subplots(nrows=1, ncols=len(images), figsize=(15, 5))
    for ax, (img, t) in zip(axs, images):
        # Convert tensor to numpy array and transpose the dimensions
        img = img.detach().cpu().numpy().transpose((1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        ax.imshow(img)
        ax.set_title(t)
        ax.axis('off')
    plt.show()


def main():
    # Define argparse parser
    main_parse = argparse.ArgumentParser()
    main_parse.add_argument("yaml", type=str, help="Path to YAML configuration file")
    main_parse.add_argument("--port", type=int, default=2357, help="Port number (default: 2357)")
    main_args = main_parse.parse_args()

    # Load configuration from YAML file
    with open(main_args.yaml, "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()

    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model = Face_Model(device, CFG)



    # Initialize dataset and dataloader
    trainset = faces_data(data_lr=os.path.join(CFG.DATA.FOLDER, "train/LOW/wider_lnew"),
                          data_hr=os.path.join(CFG.DATA.FOLDER, "train/HIGH"), img_range=CFG.DATA.IMG_RANGE,
                          rgb=CFG.DATA.RGB)
    dataloader = DataLoader(trainset, batch_size=CFG.TRAIN.BATCH_SIZE, shuffle=True)

    # Define number of epochs and interval for generating output images
    num_epochs = CFG.TRAIN.NUM_EPOCHS
    output_interval = 1
    total_batches = CFG.TRAIN.BATCH_SIZE
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch [{epoch}/{num_epochs}]")
        total_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            lr = data['lr'].to(device)
            hr = data.get('hr', None)
            if hr is not None:
                hr = hr.to(device)
            hr_down = data.get('hr_down', None)
            if hr_down is not None:
                hr_down = hr_down.to(device)
            z = data.get('z', None)
            if z is not None:
                z = z.to(device)

            if batch_idx == 0:  # Display images for the first batch of each epoch
                images = [(lr[0], 'Low Resolution'), (hr_down[0], 'Downsampled HR'), (hr[0], 'High Resolution')]
                display_images(images, f'Epoch {epoch} Visualization')

            # Check the shapes for debugging
            print(
                f'LR shape: {lr.shape}')  # Expected shape: [batch_size, 3, 64, 64] or similar, based on your dataset resizing
            if hr is not None:
                print(f'HR shape: {hr.shape}')  # Expected shape: [batch_size, 3, 128, 128] or similar
            if hr_down is not None:
                print(f'HR downsampled shape: {hr_down.shape}')  # Expected shape: [batch_size, 3, 64, 64] or similar
            if z is not None:
                print(f'Noise shape: {z.shape}')  # Expected shape: [batch_size, 1, 8, 8] or similar


            # Perform training step
            losses = model.train_step(hr, lr, hr_down, z)

            # Accumulate total loss
            total_loss += sum(losses.values())

            print(f"Batch {batch_idx + 1}/{total_batches} done.")
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}], Avg. Loss: {avg_loss:.4f}")

        # Generate output images every 3 epochs
        if epoch % output_interval == 0:
            print("Generating output images...")
            with torch.no_grad():
                # Generate output images
                output_lr, output_sr, output_hr = model.test_sample(lr, hr_down, z)
                # Save or visualize the output images as needed
                save_output_images(output_lr, output_sr, output_hr, epoch)

        # Save model weights after every epoch
        model_save_path = f"trained_model_epoch_{epoch}.pth"
        torch.save(model, model_save_path)

def save_output_images(lr, sr, hr, epoch):
    """
    Save output images for each epoch.
    """
    # Define the directory to save the output images
    output_dir = f"output_images_epoch_{epoch}"
    os.makedirs(output_dir, exist_ok=True)

    # Assuming lr, sr, hr are batches of images, we take the first image of each batch to save
    save_image(lr[0], os.path.join(output_dir, "lr.png"))  # Save the first low-resolution image
    save_image(sr[0], os.path.join(output_dir, "sr.png"))  # Save the first super-resolution image
    save_image(hr[0], os.path.join(output_dir, "hr.png"))  # Save the first high-resolution image

    # Display the images using matplotlib
    images = [(lr[0], 'Generated LR'), (sr[0], 'Super-Resolution'), (hr[0], 'High Resolution')]
    display_images(images, f'Output Images Epoch {epoch}')

def save_image(tensor, path):
    """
    Saves an image from a tensor, normalizing it to ensure that the colors
    are properly scaled to [0, 1], matching the visualization seen in matplotlib.
    """
    # Normalize the tensor to [0, 1] as in the display function
    tensor = tensor.detach().cpu()  # Move tensor to CPU and detach from the computation graph
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]

    # Convert the normalized tensor to a PIL Image
    image = to_pil_image(tensor)

    # Save the image to the specified path
    image.save(path)

# Example of how to call save_image
# Assuming 'tensor' is your image tensor and 'path' is where you want to save the image:
# save_image(tensor, 'path/to/save/image.png')



if __name__ == "__main__":
    main()
