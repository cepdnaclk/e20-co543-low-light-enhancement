import torch
from .model import enhance_net_nopool  # Import the model class
import numpy as np
import cv2
from PIL import Image
import os

def load_model(weight_file='Epoch99.pth'):
    """
    Load the pre-trained Zero-DCE model.
    """
    weights_path = os.path.join(os.path.dirname(__file__), weight_file)
    model = enhance_net_nopool()  # Initialize the model
    # Map model weights to CPU
    # model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda')))
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess(image):
    """
    Preprocess the input image for the Zero-DCE model.
    Args:
        image: PIL.Image or numpy.ndarray, input image in RGB format.
    Returns:
        torch.Tensor: Preprocessed image as a PyTorch tensor.
    """
    # Convert PIL image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Normalize to [0, 1] and convert to float32
    image = image.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensor and permute dimensions to [C, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1)
    
    # Add batch dimension [1, C, H, W]
    image = image.unsqueeze(0)
    
    return image

def postprocess(enhanced_image):
    """
    Postprocess the enhanced image from the Zero-DCE model.
    Args:
        enhanced_image: torch.Tensor, enhanced image as a PyTorch tensor.
    Returns:
        numpy.ndarray: Postprocessed image in RGB format.
    """
    # Remove batch dimension and permute dimensions back to [H, W, C]
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0)
    
    # Detach the tensor from the computation graph and convert to numpy array
    # enhanced_image = enhanced_image.detach().numpy()
    enhanced_image = enhanced_image.cpu().detach().numpy()

    # Scale to [0, 255] and convert to uint8
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    
    # Convert BGR to RGB
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    
    return enhanced_image