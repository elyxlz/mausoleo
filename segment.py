import argparse
import datetime
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from matplotlib.patches import Rectangle
from tqdm import tqdm


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
    saved_paths = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
        class_name = box["class"]
        segment = image[y1:y2, x1:x2]
        
        # Use top-left corner (x1, y1) as part of the filename
        output_path = os.path.join(output_dir, f"{int(x1)}_{int(y1)}_{class_name}.jpg")
        cv2.imwrite(output_path, segment)
        saved_paths.append(output_path)

    return saved_paths


def process_document_layout(
    image: str,
    output: str = "output",
    conf: float = 0.01,
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


def get_day_paths(data_dir, start_date=None, end_date=None):
    """
    Get all day paths in the specified data directory that need processing.
    Returns paths in chronological order.
    """
    data_dir = Path(data_dir)
    day_paths = []
    
    # If no dates are specified, get all years
    if start_date is None and end_date is None:
        years = [y for y in data_dir.iterdir() if y.is_dir() and y.name.isdigit()]
    else:
        # Get years between start_date and end_date
        start_year = start_date.year if start_date else 1800
        end_year = end_date.year if end_date else 3000
        years = [
            y for y in data_dir.iterdir() 
            if y.is_dir() and y.name.isdigit() and start_year <= int(y.name) <= end_year
        ]
    
    # Process each year
    for year_dir in sorted(years, key=lambda y: int(y.name)):
        # Process each month in this year
        for month_dir in sorted(year_dir.iterdir(), key=lambda m: datetime.datetime.strptime(m.name, "%B").month if m.is_dir() else 0):
            if not month_dir.is_dir():
                continue
                
            # Process each day in this month
            for day_dir in sorted(month_dir.iterdir(), key=lambda d: int(d.name) if d.is_dir() and d.name.isdigit() else 0):
                if not day_dir.is_dir() or not day_dir.name.isdigit():
                    continue
                    
                # Create date object for this day
                try:
                    month_num = datetime.datetime.strptime(month_dir.name, "%B").month
                    day_date = datetime.date(int(year_dir.name), month_num, int(day_dir.name))
                    
                    # Skip if before start_date or after end_date
                    if start_date and day_date < start_date:
                        continue
                    if end_date and day_date > end_date:
                        continue
                        
                    # Add this day's path
                    day_paths.append((day_date, day_dir))
                except ValueError:
                    # Skip invalid dates
                    continue
    
    # Sort by date
    return sorted(day_paths, key=lambda x: x[0])


def get_unprocessed_pages(day_dir):
    """
    Return a list of page image files that have not been processed yet
    (i.e., don't have a corresponding segment directory)
    """
    day_dir = Path(day_dir)
    unprocessed_pages = []
    
    for file_path in day_dir.iterdir():
        if file_path.is_file() and file_path.name.endswith(('.jpg', '.jpeg', '.png')):
            # Skip files that start with "annotated_" as they are output files
            if file_path.name.startswith("annotated_"):
                continue
                
            # Check if there's already a segments directory for this page
            segments_dir = day_dir / "segments"
            page_prefix = f"{int(file_path.stem)}_" if file_path.stem.isdigit() else f"{file_path.stem}_"
            
            # If there's no segments directory or no segments for this page, it needs processing
            if not segments_dir.exists() or not any(
                f.name.startswith(page_prefix) for f in segments_dir.iterdir() if segments_dir.exists()
            ):
                unprocessed_pages.append(file_path)
                
    return sorted(unprocessed_pages, key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf'))


def main():
    """Main function to run the document segmentation process"""
    parser = argparse.ArgumentParser(description="Segment newspaper pages into regions")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="/media/sdr", 
        help="Directory containing the newspaper data (default: /media/sdr)"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="Resume from date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        default=None, 
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.01, 
        help="Detection confidence threshold (default: 0.01)"
    )
    parser.add_argument(
        "--min-area", 
        type=int, 
        default=500, 
        help="Minimum area for detected regions (default: 500)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to use for inference (default: cpu)"
    )
    args = parser.parse_args()
    
    # Parse dates if provided
    start_date = None
    if args.resume:
        try:
            start_date = datetime.datetime.strptime(args.resume, "%Y-%m-%d").date()
            print(f"Resuming from {start_date}")
        except ValueError:
            print(f"Invalid resume date format: {args.resume}. Using earliest available date.")
    
    end_date = None
    if args.end_date:
        try:
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
            print(f"Processing up to {end_date}")
        except ValueError:
            print(f"Invalid end date format: {args.end_date}. Processing all available dates.")
    
    # Get all day directories to process
    print(f"Scanning {args.data_dir} for newspaper pages...")
    day_paths = get_day_paths(args.data_dir, start_date, end_date)
    
    if not day_paths:
        print("No days found to process within the specified date range.")
        return
        
    print(f"Found {len(day_paths)} days to process.")
    
    # Process each day
    model = None  # Will be loaded on first use
    
    for day_date, day_dir in tqdm(day_paths, desc="Processing days"):
        # Get unprocessed pages for this day
        pages = get_unprocessed_pages(day_dir)
        
        if not pages:
            continue
            
        print(f"\nProcessing {len(pages)} pages for {day_date} in {day_dir}")
        
        # Load model on first use
        if model is None:
            print("Loading model...")
            model = load_model()
            print("Model loaded.")
        
        # Process each page
        for page_path in tqdm(pages, desc=f"Pages for {day_date}"):
            segments_dir = os.path.join(day_dir, "segments")
            os.makedirs(segments_dir, exist_ok=True)
            
            try:
                # Process this page
                print(f"Processing {page_path}")
                results = detect_segments(
                    model, 
                    str(page_path), 
                    conf_threshold=args.conf, 
                    device=args.device
                )
                
                # Convert results to boxes
                boxes = results_to_boxes(results)
                
                # Filter boxes
                boxes = filter_boxes(boxes, min_area=args.min_area, min_confidence=args.conf)
                
                # Extract segments
                extract_segments(str(page_path), boxes, segments_dir)
                
                # Create annotated version
                visualize_segments(
                    str(page_path), 
                    boxes, 
                    str(day_dir / f"annotated_{page_path.name}")
                )
                
                print(f"Extracted {len(boxes)} segments from {page_path}")
                
            except Exception as e:
                print(f"Error processing {page_path}: {e}")
    
    print("Document segmentation complete!")


if __name__ == "__main__":
    main()