import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import numpy as np


def resize_image(image, target_size):
    """Resize image to the target size."""
    return image.resize(target_size, Image.BICUBIC)


def calculate_metrics(sr_path, hr_path):
    """Calculate PSNR and SSIM between super-resolved and high-resolution images."""
    hr_image = Image.open(hr_path).convert('RGB')
    sr_image = Image.open(sr_path).convert('RGB')

    if hr_image.size != sr_image.size:
        sr_image = resize_image(sr_image, hr_image.size)

    hr_image = np.array(hr_image, dtype=np.float32) / 255.0
    sr_image = np.array(sr_image, dtype=np.float32) / 255.0

    psnr_value = compare_psnr(hr_image, sr_image, data_range=1.0)
    ssim_value = compare_ssim(hr_image, sr_image, multichannel=True, data_range=1.0, win_size=3)

    return psnr_value, ssim_value


def process_directory(directory):
    """Process a single directory to calculate metrics for the images it contains."""
    sr_path = os.path.join(directory, 'sr.png')
    hr_path = os.path.join(directory, 'hr.png')

    if os.path.exists(sr_path) and os.path.exists(hr_path):
        psnr_value, ssim_value = calculate_metrics(sr_path, hr_path)
        return psnr_value, ssim_value
    else:
        print(f'{directory} - Missing sr.png or hr.png')
        return None, None


def main(base_dir):
    """Main function to process all directories within a base directory and calculate average metrics."""
    psnr_values = []
    ssim_values = []

    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            psnr_value, ssim_value = process_directory(dir_path)
            if psnr_value is not None and ssim_value is not None:
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                print(f'{dir_name} - PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}')

    if psnr_values and ssim_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        print(f'Average PSNR: {avg_psnr:.4f}')
        print(f'Average SSIM: {avg_ssim:.4f}')
    else:
        print('No valid image pairs found.')


if __name__ == "__main__":
    base_dir = "/Users/noymachluf/Desktop/modified_model results"
    main(base_dir)
