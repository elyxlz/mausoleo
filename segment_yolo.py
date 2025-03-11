import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from matplotlib.patches import Rectangle


def load_model():
    """Load the DocLayout YOLO model from HuggingFace"""
    filepath = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    )
    return YOLOv10(filepath)


def detect_segments(
    model, image_path, conf_threshold=0.10, image_size=1024, device="cpu"
):
    """Detect document segments using the model"""
    det_res = model.predict(
        image_path,
        imgsz=image_size,
        conf=conf_threshold,
        device=device,
    )
    return det_res


def results_to_boxes(results):
    """Convert detection results to a list of box dictionaries"""
    boxes = []
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = results[0].names[class_id]

        box_dict = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
            "area": (x2 - x1) * (y2 - y1),
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "class": class_name,
            "confidence": confidence,
            "id": i,
        }
        boxes.append(box_dict)

    return boxes


def calculate_overlap(box1, box2):
    """Calculate the vertical and horizontal overlap between two boxes"""
    x_overlap_start = max(box1["x1"], box2["x1"])
    x_overlap_end = min(box1["x2"], box2["x2"])
    y_overlap_start = max(box1["y1"], box2["y1"])
    y_overlap_end = min(box1["y2"], box2["y2"])

    if x_overlap_end <= x_overlap_start or y_overlap_end <= y_overlap_start:
        return 0.0, 0.0, 0.0

    overlap_width = x_overlap_end - x_overlap_start
    overlap_height = y_overlap_end - y_overlap_start
    overlap_area = overlap_width * overlap_height

    box1_area = box1["width"] * box1["height"]
    box2_area = box2["width"] * box2["height"]
    min_area = min(box1_area, box2_area)

    overlap_percentage = (overlap_area / min_area) * 100

    return overlap_area, overlap_percentage, min_area


def merge_boxes(box1, box2):
    """Merge two boxes into one"""
    return {
        "x1": min(box1["x1"], box2["x1"]),
        "y1": min(box1["y1"], box2["y1"]),
        "x2": max(box1["x2"], box2["x2"]),
        "y2": max(box1["y2"], box2["y2"]),
        "width": max(box1["x2"], box2["x2"]) - min(box1["x1"], box2["x1"]),
        "height": max(box1["y2"], box2["y2"]) - min(box1["y1"], box2["y1"]),
        "center_x": (min(box1["x1"], box2["x1"]) + max(box1["x2"], box2["x2"])) / 2,
        "center_y": (min(box1["y1"], box2["y1"]) + max(box1["y2"], box2["y2"])) / 2,
        "class": box1["class"],  # Keep class of first box
        "confidence": max(box1["confidence"], box2["confidence"]),
        "id": min(box1["id"], box2["id"]),
        "area": (max(box1["x2"], box2["x2"]) - min(box1["x1"], box2["x1"]))
        * (max(box1["y2"], box2["y2"]) - min(box1["y1"], box2["y1"])),
    }


def merge_overlapping_boxes(boxes, min_overlap_percent=5, max_iterations=10):
    if not boxes:
        return []

    iteration = 0
    merged = True
    merged_boxes = boxes.copy()

    while merged and iteration < max_iterations:
        merged = False
        i = 0
        while i < len(merged_boxes):
            j = i + 1
            while j < len(merged_boxes):
                box1 = merged_boxes[i]
                box2 = merged_boxes[j]

                same_class = box1["class"] == box2["class"]
                text_and_title = box1["class"] in ["text", "title"] and box2[
                    "class"
                ] in ["text", "title"]

                if not (same_class or text_and_title):
                    j += 1
                    continue

                _, overlap_percent, _ = calculate_overlap(box1, box2)

                if overlap_percent >= min_overlap_percent:
                    merged_box = merge_boxes(box1, box2)
                    merged_boxes[i] = merged_box
                    merged_boxes.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1
        iteration += 1

    print(f"Merged {len(boxes) - len(merged_boxes)} boxes in {iteration} iterations")
    return merged_boxes


def remove_subset_boxes(boxes):
    non_subset_boxes = []
    subset_count = 0
    for i, box in enumerate(boxes):
        is_subset = False
        for j, other_box in enumerate(boxes):
            if i != j:
                if (
                    box["x1"] >= other_box["x1"]
                    and box["y1"] >= other_box["y1"]
                    and box["x2"] <= other_box["x2"]
                    and box["y2"] <= other_box["y2"]
                ):
                    is_subset = True
                    break
        if not is_subset:
            non_subset_boxes.append(box)
        else:
            subset_count += 1
    print(f"Removed {subset_count} subset boxes")
    return non_subset_boxes


def filter_boxes(boxes, min_area=100, min_confidence=0.1):
    """Filter out boxes that are too small or have low confidence"""
    return [
        box
        for box in boxes
        if box["area"] >= min_area and box["confidence"] >= min_confidence
    ]


def visualize_segments(image_path, boxes, output_path=None):
    """Visualize segmentation boxes on the original image"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)

    class_colors = {
        "text": "#FF0000",  # Red
        "title": "#00FF00",  # Green
        "figure": "#0000FF",  # Blue
        "table": "#FF00FF",  # Magenta
        "caption": "#00FFFF",  # Cyan
        "footer": "#FFFF00",  # Yellow
        "header": "#FF8800",  # Orange
        "reference": "#8800FF",  # Purple
        "equation": "#00FF88",  # Mint
        "list": "#FF0088",  # Pink
        "abandon": "#888888",  # Grey
    }

    unique_classes = set(
        box["class"] for box in boxes if box["class"] not in class_colors
    )
    additional_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    for i, cls in enumerate(unique_classes):
        rgba_color = additional_colors[i]
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255)
        )
        class_colors[cls] = hex_color

    for box in boxes:
        x1, y1, _x2, _y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        width = box["width"]
        height = box["height"]
        box_class = box["class"]
        conf = box["confidence"]

        color = class_colors.get(box_class, "#FFFFFF")
        rect = Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            label=f"{box_class} {conf:.2f}",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            f"{box_class} {conf:.2f}",
            color=color,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    legend_elements = []
    for cls, color in class_colors.items():
        if cls in [box["class"] for box in boxes]:
            legend_elements.append(
                Rectangle((0, 0), 1, 1, color=color, fill=False, linewidth=2, label=cls)
            )

    if legend_elements:
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def extract_segments(image_path, boxes, output_dir):
    """Extract individual segments from the image based on bounding boxes"""
    image = cv2.imread(str(image_path))
    os.makedirs(output_dir, exist_ok=True)
    file_base = os.path.splitext(os.path.basename(image_path))[0]
    saved_paths = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
        class_name = box["class"]
        segment = image[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"{file_base}_{class_name}_{i}.jpg")
        cv2.imwrite(output_path, segment)
        saved_paths.append(output_path)

    return saved_paths


def process_document_layout(
    image: str,
    output: str = "output",
    conf: float = 0.03,
    min_area: int = 500,
    min_overlap: float = 40,
    image_size: int = 1024,
    device: str = "cpu",
):
    os.makedirs(output, exist_ok=True)

    print("Loading model...")
    model = load_model()

    print(f"Processing image: {image}")
    results = detect_segments(
        model, image, conf_threshold=conf, image_size=image_size, device=device
    )

    print("Processing detection results...")
    boxes = results_to_boxes(results)

    # print(f"Filtering boxes (original count: {len(boxes)})...")
    boxes = filter_boxes(boxes, min_area=min_area, min_confidence=conf)
    # print(f"Removed {len(boxes) - len(boxes)} small or low-confidence boxes")
    #
    # print("Merging overlapping boxes...")
    # boxes = merge_overlapping_boxes(boxes, min_overlap_percent=min_overlap)
    #
    # print("Removing subset boxes...")
    # boxes = remove_subset_boxes(boxes)

    image_name = os.path.basename(image)
    output_image = os.path.join(output, f"annotated_{image_name}")
    print(f"Saving annotated image to: {output_image}")
    visualize_segments(image, boxes, output_image)

    segments_dir = os.path.join(output, "segments")
    print(f"Extracting segments to: {segments_dir}")
    segment_paths = extract_segments(image, boxes, segments_dir)
    print(f"Extracted {len(segment_paths)} segments")

    print("Processing complete!")
    return segment_paths


if __name__ == "__main__":
    # Replace 'path/to/your/image.jpg' with your actual image path or call process_document_layout with your parameters.
    default_image = "./data/1880/January/8/1.jpeg"
    process_document_layout(image=default_image)
