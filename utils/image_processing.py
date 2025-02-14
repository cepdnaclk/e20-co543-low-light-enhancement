import cv2
import numpy as np

def adjust_brightness(image, value):
    # Normalize the value to be in the range of -100 to 100, which will scale the change in brightness
    brightness_factor = value / 200.0
    
    # Convert the image to float32 for safe arithmetic operations
    img = np.float32(image)
    
    # Adjust the brightness: adding the brightness_factor will increase/decrease pixel values
    result = img + (brightness_factor * 255)
    
    # Clip the values to be within the valid range [0, 255]
    result = np.clip(result, 0, 255)
    
    # Convert the result back to uint8
    result = np.uint8(result)
    
    return result

def gamma_transform(image, gamma):
    """
    Apply gamma correction using the provided gamma value.
    The lower the gamma value (<1), the brighter the low-light image becomes.
    """
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def log_transform(img):
  """
  Apply logarithmic transformation for contrast enhancement.
  """
  # Convert the image to float32 for safe arithmetic operations
  img = np.float32(img)
  
  # Apply the log transformation to each channel
  c = 255 / np.log(1 + np.max(img))  # Scaling constant
  result = c * np.log(1 + img)
  
  # Clip the values to be within the valid range [0, 255]
  result = np.clip(result, 0, 255)
  
  # Convert the result back to uint8
  result = np.uint8(result)
  
  return result

def apply_noise_reduction(image, strength):
    """
    Apply Bilateral Filtering for noise reduction while preserving edges and minimizing color change.
    
    Parameters:
    - image: Input image (NumPy array)
    - strength: Controls the filter strength. Higher values reduce more noise.

    Returns:
    - Denoised image
    """
    # Convert the image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Extract the Y (luminance) channel for noise reduction
    y_channel = yuv_image[:, :, 0]

    # Define bilateral filter parameters
    d = 9  # Diameter of pixel neighborhood
    sigma_color = 75  # Fixed value for color filtering (reduce this value if colors are still affected)
    sigma_space = strength * 5  # Strength of spatial filtering
    
    # Apply bilateral filter only on the luminance (Y) channel
    y_channel_filtered = cv2.bilateralFilter(y_channel, d, sigma_color, sigma_space)

    # Replace the filtered Y channel back into the YUV image
    yuv_image[:, :, 0] = y_channel_filtered

    # Convert back to BGR color space
    denoised_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return denoised_image


def sharpen_image(image, strength):
    """
    Apply unsharp masking to enhance image sharpness without affecting colors and brightness.
    
    Parameters:
    - image: Input image (NumPy array)
    - strength: Controls the sharpening intensity (recommended range: 0 to 5)
    
    Returns:
    - Sharpened image
    """
    # Convert the image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Extract the Y (luminance) channel for sharpening
    y_channel = yuv_image[:, :, 0]

    # Convert strength to a reasonable factor (avoiding excessive sharpening)
    alpha = 1.0 + (strength / 5.0)  # Scaling factor for detail enhancement

    # Apply Gaussian blur to extract low-frequency components
    blurred = cv2.GaussianBlur(y_channel, (3, 3), 0)

    # Compute the sharpened image using Unsharp Masking
    sharpened_y_channel = cv2.addWeighted(y_channel, alpha, blurred, -0.5, 0)

    # Replace the sharpened Y channel back into the YUV image
    yuv_image[:, :, 0] = sharpened_y_channel

    # Convert back to BGR color space
    sharpened_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return sharpened_image

def apply_median_blur(image, strength):
    """
    Apply median blur for noise reduction while preserving edges.
    
    Parameters:
    - image: Input image (NumPy array)
    - strength: Strength of the blur (must be an odd number)
    
    Returns:
    - Blurred image with reduced noise
    """
    kernel_size = 2 * strength + 1  # Ensure kernel size is always odd
    return cv2.medianBlur(image, kernel_size)
