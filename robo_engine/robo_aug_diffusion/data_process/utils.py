import cv2

def resize_with_padding(image, target_size):
    """
    Resize an image with padding to make it a square of the target size.
    
    Args:
        image (numpy.ndarray): The input image.
        target_size (int): The desired size (both width and height) of the output image.
    
    Returns:
        numpy.ndarray: The resized image with padding.
    """
    # Get original image dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate the scaling factor to resize the image
    scale = target_size / max(original_width, original_height)
    
    # Resize the image while keeping the aspect ratio
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding for width and height
    pad_top = (target_size - new_height) // 2
    pad_bottom = target_size - new_height - pad_top
    pad_left = (target_size - new_width) // 2
    pad_right = target_size - new_width - pad_left
    
    # Apply padding
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black padding; you can change the value for different padding colors
    )
    
    return padded_image