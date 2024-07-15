import cv2
import numpy as np

# Convert a PyTorch tensor to an image using OpenCV
def tensor_to_image(tensor, img_range, rgb=True):
    # Remove unnecessary dimensions and move tensor to CPU
    m_tens = tensor.detach().squeeze().cpu()
    # Ensure the input tensor has three dimensions (channels, height, width)
    assert len(m_tens.size()) == 3
    # Convert the tensor to a NumPy array and normalize it to the range [0, 1]
    arrays = np.clip(m_tens.numpy().transpose(1, 2, 0), a_min=0, a_max=img_range) / img_range
    # Scale to the range [0, 255] and convert to an 8-bit unsigned integer format
    img = np.around(arrays * 255).astype(np.uint8)
    # Convert from RGB to BGR color space if necessary
    if rgb: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Return the processed image
    return img

# Save a PyTorch tensor as an image file
def save_tensor_image(file_name, tensor, img_range, rgb=True):
    # Convert the tensor to an image
    img = tensor_to_image(tensor, img_range=img_range, rgb=rgb)
    # Save the image to the specified file path using OpenCV
    cv2.imwrite(file_name, img)

# A class to compute and track the average of data (e.g., losses)
class AverageMeter():
    # Initialize the number of samples and the initial average
    def __init__(self, data=None, ema_alpha=None):
        self.n_samples = 1 if data is not None else 0
        self.average = data if data is not None else 0
        self.ema_alpha = ema_alpha # Exponential moving average alpha

    # Update the average with new data
    def update(self, data):
        # Compute a simple moving average
        if self.ema_alpha is None:
            self.average = (self.n_samples) / (self.n_samples + 1) * self.average + 1 / (self.n_samples + 1) * data
        else:
            # Compute an exponential moving average if alpha is provided
            self.average = (1 - self.ema_alpha) * self.average + self.ema_alpha * data if self.n_samples >= 1 else data
        self.n_samples += 1

    # Retrieve the current average value
    def get_avg(self):
        return self.average

    # Reset the average meter to its initial state
    def reset(self):
        self.n_samples = 0
        self.average = 0