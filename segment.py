import argparse
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from matplotlib.patches import Rectangle
from tqdm import tqdm


def load_model() -> YOLOv10:
    filepath = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    )
    return YOLOv10(filepath)


def detect_segments(
    model: YOLOv10,
    image_path: str | list[str],
    conf_threshold: float = 0.10,
    image_size: int = 1024,
    device: str = "cpu",
    batch_size: int = 1,
) -> Any:
    det_res = model.predict(
        image_path,
        imgsz=image_size,
        conf=conf_threshold,
        device=device,
        batch=batch_size,
    )
    return det_res


def results_to_boxes(results: list[Any]) -> list[dict[str, Any]]:
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


def calculate_overlap(box1: dict[str, Any], box2: dict[str, Any]) -> tuple[float, float, float]:
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


def merge_boxes(box1: dict[str, Any], box2: dict[str, Any]) -> dict[str, Any]:
    return {
        "x1": min(box1["x1"], box2["x1"]),
        "y1": min(box1["y1"], box2["y1"]),
        "x2": max(box1["x2"], box2["x2"]),
        "y2": max(box1["y2"], box2["y2"]),
        "width": max(box1["x2"], box2["x2"]) - min(box1["x1"], box2["x1"]),
        "height": max(box1["y2"], box2["y2"]) - min(box1["y1"], box2["y1"]),
        "center_x": (min(box1["x1"], box2["x1"]) + max(box1["x2"], box2["x2"])) / 2,
        "center_y": (min(box1["y1"], box2["y1"]) + max(box1["y2"], box2["y2"])) / 2,
        "class": box1["class"],
        "confidence": max(box1["confidence"], box2["confidence"]),
        "id": min(box1["id"], box2["id"]),
        "area": (max(box1["x2"], box2["x2"]) - min(box1["x1"], box2["x1"])) * (max(box1["y2"], box2["y2"]) - min(box1["y1"], box2["y1"])),
    }


def merge_overlapping_boxes(
    boxes: list[dict[str, Any]],
    min_overlap_percent: float = 5,
    max_iterations: int = 10,
) -> list[dict[str, Any]]:
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
                text_and_title = box1["class"] in ["text", "title"] and box2["class"] in ["text", "title"]

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


def remove_subset_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def filter_boxes(boxes: list[dict[str, Any]], min_area: float = 100, min_confidence: float = 0.1) -> list[dict[str, Any]]:
    return [box for box in boxes if box["area"] >= min_area and box["confidence"] >= min_confidence]


def visualize_segments(
    image_path: str | Path,
    boxes: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> None:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)

    class_colors = {
        "text": "#FF0000",
        "title": "#00FF00",
        "figure": "#0000FF",
        "table": "#FF00FF",
        "caption": "#00FFFF",
        "footer": "#FFFF00",
        "header": "#FF8800",
        "reference": "#8800FF",
        "equation": "#00FF88",
        "list": "#FF0088",
        "abandon": "#888888",
    }

    unique_classes = set(box["class"] for box in boxes if box["class"] not in class_colors)
    additional_colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(unique_classes)))
    for i, cls in enumerate(unique_classes):
        rgba_color = additional_colors[i]
        hex_color = f"#{int(rgba_color[0] * 255):02x}{int(rgba_color[1] * 255):02x}{int(rgba_color[2] * 255):02x}"
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
            legend_elements.append(Rectangle((0, 0), 1, 1, color=color, fill=False, linewidth=2, label=cls))

    if legend_elements:
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def _save_segment(args: tuple[np.ndarray, dict[str, Any], str, str]) -> str:
    image_data, box, output_dir, page_number = args
    x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
    segment = image_data[y1:y2, x1:x2]

    output_path = os.path.join(output_dir, f"{page_number}_{int(x1)}_{int(y1)}.jpg")
    cv2.imwrite(output_path, segment)
    return output_path


def extract_segments(
    image_path: str | Path,
    boxes: list[dict[str, Any]],
    output_dir: str | Path,
    page_number: str | None = None,
) -> list[str]:
    image = cv2.imread(str(image_path))
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    if page_number is None:
        image_basename = os.path.basename(image_path)
        page_number = os.path.splitext(image_basename)[0]
        if not page_number.isdigit():
            raise ValueError(f"Invalid page number in filename: {image_basename}")

    with ThreadPoolExecutor() as executor:
        save_args = [(image, box, output_dir, page_number) for box in boxes]
        saved_paths = list(executor.map(_save_segment, save_args))

    return saved_paths


def get_day_paths(
    data_dir: str | Path,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> list[tuple[datetime.date, Path]]:
    data_dir = Path(data_dir)
    day_paths = []

    if start_date is None and end_date is None:
        years = [y for y in data_dir.iterdir() if y.is_dir() and y.name.isdigit()]
    else:
        start_year = start_date.year if start_date else 1800
        end_year = end_date.year if end_date else 3000
        years = [y for y in data_dir.iterdir() if y.is_dir() and y.name.isdigit() and start_year <= int(y.name) <= end_year]

    for year_dir in sorted(years, key=lambda y: int(y.name)):
        for month_dir in sorted(
            year_dir.iterdir(),
            key=lambda m: datetime.datetime.strptime(m.name, "%B").month if m.is_dir() else 0,
        ):
            if not month_dir.is_dir():
                continue

            for day_dir in sorted(
                month_dir.iterdir(),
                key=lambda d: int(d.name) if d.is_dir() and d.name.isdigit() else 0,
            ):
                if not day_dir.is_dir() or not day_dir.name.isdigit():
                    continue

                try:
                    month_num = datetime.datetime.strptime(month_dir.name, "%B").month
                    day_date = datetime.date(int(year_dir.name), month_num, int(day_dir.name))

                    if start_date and day_date < start_date:
                        continue
                    if end_date and day_date > end_date:
                        continue

                    day_paths.append((day_date, day_dir))
                except ValueError:
                    continue

    return sorted(day_paths, key=lambda x: x[0])


def get_unprocessed_pages(day_dir: str | Path) -> list[Path]:
    day_dir = Path(day_dir)
    unprocessed_pages = []

    for file_path in day_dir.iterdir():
        if file_path.is_file() and file_path.name.endswith((".jpg", ".jpeg", ".png")):
            if file_path.name.startswith("annotated_"):
                continue

            segments_dir = day_dir / "segments"
            page_prefix = f"{int(file_path.stem)}_" if file_path.stem.isdigit() else f"{file_path.stem}_"

            if not segments_dir.exists() or not any(f.name.startswith(page_prefix) for f in segments_dir.iterdir() if segments_dir.exists()):
                unprocessed_pages.append(file_path)

    return sorted(
        unprocessed_pages,
        key=lambda x: int(x.stem) if x.stem.isdigit() else float("inf"),
    )


def process_batch(
    model: YOLOv10,
    page_paths: list[Path],
    args: argparse.Namespace,
    pbar: tqdm | None = None,
) -> list[tuple[Path, int]]:
    results = []

    str_page_paths = [str(path) for path in page_paths]

    try:
        detection_results = model.predict(
            str_page_paths,
            imgsz=1024,
            conf=args.conf,
            device=args.device,
            batch=len(str_page_paths),
        )

        for idx, page_path in enumerate(page_paths):
            try:
                page_number = os.path.splitext(os.path.basename(str(page_path)))[0]
                day_dir = os.path.dirname(page_path)
                segments_dir = os.path.join(day_dir, "segments")
                os.makedirs(segments_dir, exist_ok=True)

                if not page_number.isdigit():
                    print(f"Skipping {page_path}: Invalid page number format")
                    continue

                existing_segments = [
                    f for f in os.listdir(segments_dir) if f.startswith(f"{page_number}_") and os.path.isfile(os.path.join(segments_dir, f))
                ]
                for f in existing_segments:
                    os.remove(os.path.join(segments_dir, f))

                this_result = detection_results[idx] if isinstance(detection_results, list) else detection_results[0]

                boxes = results_to_boxes([this_result])

                boxes = filter_boxes(boxes, min_area=args.min_area, min_confidence=args.conf)

                extract_segments(str(page_path), boxes, segments_dir, page_number)

                if args.save_annotated:
                    visualize_segments(
                        str(page_path),
                        boxes,
                        os.path.join(day_dir, f"annotated_{os.path.basename(str(page_path))}"),
                    )

                print(f"Extracted {len(boxes)} segments from {page_path}")
                results.append((page_path, len(boxes)))

                if pbar:
                    pbar.update(1)

            except Exception as e:
                print(f"Error processing result for {page_path}: {e}")
                if pbar:
                    pbar.update(1)

    except Exception as e:
        print(f"Error in batch processing: {e}")
        if pbar:
            pbar.update(len(page_paths))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment newspaper pages into regions")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/media/sdr",
        help="Directory containing the newspaper data (default: /media/sdr)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Detection confidence threshold (default: 0.01)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum area for detected regions (default: 500)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated images with bounding boxes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of images to process in each batch (default: 256)",
    )
    args = parser.parse_args()

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

    args.device = "cuda" if args.device == "cpu" and __import__("torch").cuda.is_available() else args.device

    print(f"Scanning {args.data_dir} for newspaper pages...")
    day_paths = get_day_paths(args.data_dir, start_date, end_date)

    if not day_paths:
        print("No days found to process within the specified date range.")
        return

    print(f"Found {len(day_paths)} days to process.")

    all_pages = []
    for day_date, day_dir in day_paths:
        pages = get_unprocessed_pages(day_dir)
        if pages:
            all_pages.extend(pages)

    total_pages = len(all_pages)
    print(f"Found {total_pages} total pages to process.")

    if total_pages == 0:
        print("No unprocessed pages found.")
        return

    print("Loading model...")
    model = load_model()
    print("Model loaded.")

    pbar = tqdm(total=total_pages, desc="Processing pages")

    batch_size = args.batch_size
    print(f"Processing with batch size: {batch_size}")

    for i in range(0, len(all_pages), batch_size):
        batch_pages = all_pages[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(all_pages) + batch_size - 1)//batch_size} with {len(batch_pages)} pages")
        process_batch(model, batch_pages, args, pbar)

    pbar.close()
    print("Document segmentation complete!")


if __name__ == "__main__":
    main()
