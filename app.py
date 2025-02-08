import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from diffusers import MarigoldDepthPipeline
import matplotlib.pyplot as plt
import torch  # Import torch at the top level


@st.cache_resource()
def load_yolo_model():
    return YOLO('yolov8n.pt')


@st.cache_resource()
def load_marigold_model():
    # No need to import torch here, it's already imported globally
    return MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-0", torch_dtype=torch.float32)


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


def estimate_depth_marigold(image):
    """Estimates depth using Marigold."""
    depth_estimator = load_marigold_model()
    with torch.no_grad():
        output = depth_estimator(image)
        depth_map = output.prediction
        depth_map = np.squeeze(depth_map)  # Remove singleton dimensions
    return depth_map

def visualize_depth(depth_map, colormap=cv2.COLORMAP_VIRIDIS):
    """Visualizes the depth map with a specified colormap."""
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


# --- Streamlit App ---
st.title("STORARO: STILL ANALYSIS")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file into a bytes object
    file_bytes = uploaded_file.read()
    # Convert to a NumPy array
    file_bytes_np = np.frombuffer(file_bytes, dtype=np.uint8)
    # Decode the image
    image = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)

    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB
        yolo_model = load_yolo_model()  # Use consistent variable name
        boxes, class_ids = detect_objects(image, yolo_model)  # Use yolo_model
        annotated_image = draw_boxes_and_grid(image, boxes, class_ids, yolo_model)  # Use yolo_model
        height, width, _ = image.shape
        grid_x = [width // 3, 2 * width // 3]
        grid_y = [height // 3, 2 * height // 3]

        # Depth Estimation (Marigold)
        depth_map = estimate_depth_marigold(image)
        depth_colormap = visualize_depth(depth_map)

        # --- Analysis ---
        size_placement_results = analyze_object_size_and_placement(image, boxes, class_ids, yolo_model) # Use yolo_model
        rule_of_thirds_results = perform_rule_of_thirds_analysis(boxes, class_ids, yolo_model, grid_x, grid_y) # Use yolo_model
        depth_layer_results = perform_depth_layer_analysis(boxes, class_ids, yolo_model, depth_map) # Use yolo_model

        # --- Display Results using st.columns ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Object Detection and Rule of Thirds")
            st.image(annotated_image, use_container_width=True)
        with col2:
            st.subheader("Depth Map")
            st.image(depth_colormap, use_container_width=True)
            # Add a colorbar (using Matplotlib)
            fig, ax = plt.subplots(figsize=(2, 0.5))
            norm = plt.Normalize(vmin=depth_map.min(), vmax=depth_map.max())
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=ax, orientation='horizontal')
            cbar.set_label('Depth (Relative)')
            st.pyplot(fig)  # Pass the figure object

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