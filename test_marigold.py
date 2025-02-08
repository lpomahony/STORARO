from diffusers import MarigoldDepthPipeline
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use the non-LCM version
model_id = "prs-eth/marigold-depth-v1-0"

# Load the pipeline
pipe = MarigoldDepthPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Load a test image
image_path = "images/test_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to the correct format
image = np.array(image) / 255.0

# Perform depth estimation
with torch.no_grad():
    output = pipe(image)
    depth_map = output.prediction
    print(f"Original depth_map shape: {depth_map.shape}")  # Inspect the shape *before* squeeze
    depth_map = np.squeeze(depth_map)  # Remove singleton dimensions
    print(f"Squeezed depth_map shape: {depth_map.shape}")  # Check the new shape


# Visualize the depth map
depth_colormap = cv2.applyColorMap(cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_VIRIDIS)
plt.imshow(depth_colormap)
plt.title("Marigold Depth Map")
plt.axis('off')
plt.show()

print("Marigold loaded and depth map generated successfully!")