import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
#from diffusers import MarigoldDepthPipeline # No longer using Marigold
import matplotlib.pyplot as plt
import torch

torch.classes.__path__ = []  # Workaround for Streamlit/PyTorch issue

@st.cache_resource()
def load_yolo_model():
    return YOLO('yolov8n.pt')

# @st.cache_resource() # No longer using marigold
# def load_marigold_model():
#     return MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-0", torch_dtype=torch.float32)

# --- MiDaS Model Loading ---
@st.cache_resource()
def load_midas_model():
    model_type = "MiDaS_small"  # Or "MiDaS_large" or other options.  "MiDaS_small" is faster.
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    return midas

# --- MiDaS Depth Estimation ---
def estimate_depth_midas(image):
    """Estimates depth using MiDaS."""
    midas = load_midas_model()

    # Use CUDA if available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load the appropriate transform based on the model type
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform #hard coding transform

    # Transform the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MiDaS expects RGB input
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        # Resize to original image dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map

def detect_objects(image, model):
    results = model(image)
    boxes = []
    class_ids = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            class_ids.append(int(box.cls[0].item()))
    return boxes, class_ids

def draw_boxes_and_grid(image, boxes, class_ids, model):
    annotated_image = image.copy()
    height, width, _ = image.shape
    grid_x = [width // 3, 2 * width // 3]
    grid_y = [height // 3, 2 * height // 3]
    for x in grid_x:
        cv2.line(annotated_image, (x, 0), (x, height), (255, 255, 255), 2)
    for y in grid_y:
        cv2.line(annotated_image, (0, y), (width, y), (255, 255, 255), 2)
    for (x, y, w, h), class_id in zip(boxes, class_ids):
        label = model.names[class_id]
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return annotated_image

# def estimate_depth_marigold(image): #no longer using marigold
#     """Estimates depth using Marigold."""
#     depth_estimator = load_marigold_model()
#     with torch.no_grad():
#         output = depth_estimator(image)
#         depth_map = output.prediction
#         depth_map = np.squeeze(depth_map)
#     return depth_map

def visualize_depth(depth_map, colormap=cv2.COLORMAP_VIRIDIS):
    depth_colormap = cv2.applyColorMap(cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U), colormap)
    return depth_colormap

def calculate_object_depth_layer(depth_map, box, num_layers=3):
    x, y, w, h = box
    object_region = depth_map[y:y+h, x:x+w]
    if object_region.size == 0:
        return 0
    average_depth = np.mean(object_region)
    max_depth = np.max(depth_map)
    min_depth = np.min(depth_map)
    layer_boundaries = np.linspace(min_depth, max_depth, num_layers + 1)
    for i in range(num_layers):
        if average_depth >= layer_boundaries[i] and average_depth < layer_boundaries[i+1]:
            return i
    return num_layers - 1

def analyze_object_size_and_placement(image, boxes, class_ids, model):
    image_height, image_width, _ = image.shape
    image_area = image_width * image_height
    analysis_results = []
    for (x, y, w, h), class_id in zip(boxes, class_ids):
        label = model.names[class_id]
        box_area = w * h
        relative_size = (box_area / image_area) * 100
        if relative_size > 10:
            size_category = "Large"
        elif relative_size > 2:
            size_category = "Medium"
        else:
            size_category = "Small"
        center_x = x + w // 2
        center_y = y + h // 2
        x_position = "Center"
        y_position = "Center"
        if center_x < image_width // 3:
            x_position = "Left"
        elif center_x > 2 * image_width // 3:
            x_position = "Right"
        if center_y < image_height // 3:
            y_position = "Top"
        elif center_y > 2 * image_height // 3:
            y_position = "Bottom"
        analysis_results.append(f"{label}: Size = {size_category} ({relative_size:.2f}%), Position = {y_position}-{x_position}")
    return analysis_results

def perform_rule_of_thirds_analysis(boxes, class_ids, model, grid_x, grid_y):
    analysis_results = []
    for (x, y, w, h), class_id in zip(boxes, class_ids):
        label = model.names[class_id]
        percentages = calculate_intersection_percentage((x, y, w, h), grid_x, grid_y)
        intersections = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        for i, percentage in enumerate(percentages):
            if percentage > 25:
                analysis_results.append(f"{label} aligns with the {intersections[i]} intersection (overlap: {percentage:.2f}%)")
    return analysis_results

def calculate_intersection_percentage(box, grid_x, grid_y):
    x, y, w, h = box
    box_area = w * h
    percentages = []
    for gy in grid_y:
        for gx in grid_x:
            ix1 = max(x, gx - 5)
            iy1 = max(y, gy - 5)
            ix2 = min(x + w, gx + 5)
            iy2 = min(y + h, gy + 5)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection_area = iw * ih
            percentage = (intersection_area / box_area) * 100 if box_area > 0 else 0
            percentages.append(percentage)
    return percentages

def perform_depth_layer_analysis(boxes, class_ids, model, depth_map):
    analysis_results = []
    for (x, y, w, h), class_id in zip(boxes, class_ids):
        label = model.names[class_id]
        layer = calculate_object_depth_layer(depth_map, (x, y, w, h))
        layer_names = ["Foreground", "Midground", "Background"]
        analysis_results.append(f"{label} is in the {layer_names[layer]}")
    return analysis_results

def delight_image(image, depth_map):
    """
    Performs a simplified de-lighting operation, combining depth-based
    division with bilateral filtering and CLAHE.
    """

    # 1. Normalize and Invert Depth Map (as a proxy for lighting)
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    depth_map_inverted = 1 - depth_map_normalized
    depth_map_inverted = depth_map_inverted + 0.01  # Avoid division by zero


    # Ensure the depth map is the same size as the image and has 3 channels.
    depth_map_inverted = np.squeeze(depth_map_inverted)
    if depth_map_inverted.ndim == 2:
        depth_map_inverted_rgb = cv2.cvtColor(depth_map_inverted.astype(np.float32), cv2.COLOR_GRAY2BGR)
    elif depth_map_inverted.ndim == 3:
        depth_map_inverted_rgb = depth_map_inverted
    else:
        raise ValueError("Depth map must be 2D (grayscale) or 3D")

    # 2. Divide Image by (Inverted) Depth
    image = image.astype(np.float32) / 255.0  # Normalize image to [0, 1]
    delit_image = image / depth_map_inverted_rgb
    delit_image = np.clip(delit_image, 0, 1) # Clip to valid range

    # 3. Bilateral Filtering
    delit_image_8bit = (delit_image * 255).astype(np.uint8) # Convert to 8-bit for OpenCV
    delit_image_filtered = cv2.bilateralFilter(delit_image_8bit, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(delit_image_filtered, cv2.COLOR_RGB2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)  # Split into L, a, b channels
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)  # Apply CLAHE to the L channel
    limg = cv2.merge((cl, a, b))  # Merge the channels back
    delit_image_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)  # Convert back to RGB

    # 5. Shadows/Highlights Adjustment (Simplified)
    # Use the inverted depth map as a VERY rough mask for shadows/highlights
    #  (This is a simplification - a proper implementation would need more sophisticated masking)
    delit_image_clahe = delit_image_clahe.astype(np.float32) / 255.0
    shadow_mask = depth_map_normalized
    highlight_mask = 1 - depth_map_normalized

    # Expand dimensions to make the masks 3D (matching the image)
    shadow_mask = np.expand_dims(shadow_mask, axis=2)  # Add a channel dimension
    highlight_mask = np.expand_dims(highlight_mask, axis=2)

    # Adjust shadows and highlights (linear scaling)
    shadow_factor = 1.2  # Brighten shadows
    highlight_factor = 0.8  # Darken highlights

    delit_image_final = delit_image_clahe * (1 + (shadow_mask * (shadow_factor - 1)))  # Apply shadow adjustment
    delit_image_final = delit_image_final * (1 + (highlight_mask * (highlight_factor - 1)))  # Apply highlight adjustment
    delit_image_final = np.clip(delit_image_final, 0, 1) # Ensure values stay within [0, 1]


    return (delit_image_final * 255).astype(np.uint8) #convert back to 8 bit for display.

def multiscale_retinex(img, sigma_list):
    """Applies a simplified Multi-Scale Retinex (MSR) algorithm."""
    img = img.astype(np.float32) / 255.0
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log10(img + 0.01) - np.log10(blurred + 0.01)  # Avoid log(0)
    retinex = retinex / len(sigma_list)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)

def delight_guided_filter(image, depth_map, radius=15, eps=1e-3):
    """Applies guided filtering for de-lighting, using the depth map as guidance."""

    # Convert image to float32 and normalize
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    # Normalize depth map and ensure correct type
    guide = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    guide = np.squeeze(guide)
    print(f"Guide shape: {guide.shape}, Guide type: {guide.dtype}")


    # Convert guide to 3-channel if it's grayscale
    if guide.ndim == 2:
        guide = cv2.cvtColor(guide, cv2.COLOR_GRAY2BGR)


    # Guided filtering on the *entire image* (all channels at once)
    smoothed = cv2.ximgproc.guidedFilter(guide=guide, src=image, radius=radius, eps=eps, dDepth=-1)
    print(f"Smoothed shape: {smoothed.shape}, Smoothed type: {smoothed.dtype}")


    # Avoid division by zero
    smoothed = np.maximum(smoothed, 1e-5)  # Add small constant

    # Calculate reflectance in log domain
    reflectance = np.log10(image + 0.01) - np.log10(smoothed + 0.01)
    print(f"Reflectance shape: {reflectance.shape}, Reflectance type: {reflectance.dtype}")


    # Normalize the result and convert to 8-bit for display.
    reflectance = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(f"Reflectance shape after norm: {reflectance.shape}, Reflectance type after norm: {reflectance.dtype}")
    return reflectance

# --- Streamlit App ---
st.title("STORARO")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add sliders for guided filter parameters
radius = st.slider("Guided Filter Radius", min_value=1, max_value=50, value=15, step=1)
eps = st.slider("Guided Filter Epsilon (x 1e-3)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
eps = eps * 1e-3  # Scale eps

# Add a selection for de-lighting method
delight_method = st.selectbox("Select De-lighting Method:", ["None", "Original (Depth-Based)", "MSR", "Guided Filter"])


if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_bytes_np = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)
    print(f"Original Image shape: {image.shape}, Original Image type: {image.dtype}")

    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo_model = load_yolo_model()
        boxes, class_ids = detect_objects(image, yolo_model)
        annotated_image = draw_boxes_and_grid(image, boxes, class_ids, yolo_model)
        height, width, _ = image.shape
        grid_x = [width // 3, 2 * width // 3]
        grid_y = [height // 3, 2 * height // 3]

        # --- Switched to MiDaS ---
        #depth_map = estimate_depth_marigold(image) #Using MiDaS Instead
        depth_map = estimate_depth_midas(image) #using midas
        print(f"Depth Map shape: {depth_map.shape}, Depth Map type: {depth_map.dtype}, Min value: {np.min(depth_map)}, Max value: {np.max(depth_map)}")
        depth_colormap = visualize_depth(depth_map)

        # De-lighting options
        col1, col2, col3, col4 = st.columns(4) #set up columns
        with col1:
            st.subheader("Object Detection")
            st.image(annotated_image, use_container_width=True)
        with col2:
            st.subheader("Depth Map")
            st.image(depth_colormap, use_container_width=True)
            fig, ax = plt.subplots(figsize=(2, 0.5))
            norm = plt.Normalize(vmin=depth_map.min(), vmax=depth_map.max())
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=ax, orientation='horizontal')
            cbar.set_label('Depth (Relative)')
            st.pyplot(fig)
        with col3:
            st.subheader("Delit Image")
            if delight_method == "Original (Depth-Based)":
                delit_image = delight_image(image, depth_map)
                st.image(delit_image, use_container_width=True, channels = "BGR")
            elif delight_method == "None":
                st.image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), use_container_width=True)
            else:
                st.empty()  # Placeholder for when no de-lighting is selected
        with col4:
            st.subheader("MSR/Guided Filter")
            if delight_method == "MSR":
                sigma_list = [15, 80, 250]
                msr_image = multiscale_retinex(image, sigma_list)
                st.image(msr_image, use_container_width=True, channels="BGR")
            elif delight_method == "Guided Filter":
                guided_delit_image = delight_guided_filter(image, depth_map, radius=radius, eps=eps) # Pass parameters
                st.image(guided_delit_image, use_container_width=True, channels="BGR")
            else:
                st.empty()


        size_placement_results = analyze_object_size_and_placement(image, boxes, class_ids, yolo_model)
        rule_of_thirds_results = perform_rule_of_thirds_analysis(boxes, class_ids, yolo_model, grid_x, grid_y)
        depth_layer_results = perform_depth_layer_analysis(boxes, class_ids, yolo_model, depth_map)

        st.subheader("Analysis Results")
        st.markdown("**Object Size and Placement:**")
        for result in size_placement_results:
            st.text(result)
        st.markdown("**Rule of Thirds Analysis:**")
        for result in rule_of_thirds_results:
            st.text(result)
        st.markdown("**Depth Layer Analysis:**")
        for result in depth_layer_results:
            st.text(result)