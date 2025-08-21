import base64
import time
from threading import Lock
from typing import Any, NamedTuple, Optional

import cv2
import numpy as np
import uvicorn
from doclayout_yolo import YOLOv10
from fastapi import Depends, FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel


class YoloRequest(BaseModel):
    images: list[str]  # List of base64 encoded images


class YoloContext(NamedTuple):
    model: Optional[Any] = None
    batch_size: int = 8
    device: str = "cpu"
    conf_threshold: float = 0.10
    min_area: float = 100
    processing_interval_seconds: float = 0.5
    batch_queue: list = []
    results: dict = {}
    lock: Lock = Lock()
    processing: bool = False
    last_process_time: float = 0.0


def load_model(context: YoloContext) -> YOLOv10:
    """Load the YOLOv10 model from HuggingFace Hub."""
    if context.model is None:
        filepath = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        # Since we can't modify the namedtuple directly, we need to return the model
        # The caller is responsible for managing the model instance
        return YOLOv10(filepath)
    return context.model


def results_to_boxes(results: list[Any]) -> list[dict[str, Any]]:
    """Convert YOLO results to a list of box dictionaries."""
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


def filter_boxes(boxes: list[dict[str, Any]], min_area: float = 100, min_confidence: float = 0.1) -> list[dict[str, Any]]:
    return [box for box in boxes if box["area"] >= min_area and box["confidence"] >= min_confidence]


def extract_segment(image_data: np.ndarray, box: dict[str, Any]) -> tuple[Optional[np.ndarray], dict[str, int]]:
    try:
        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])

        height, width = image_data.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None, {}

        segment = image_data[y1:y2, x1:x2]

        if segment.size == 0:
            return None, {}

        position = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
        }

        return segment, position

    except Exception as e:
        print(f"Error extracting segment: {e}")
        return None, {}


def process_batch(model: YOLOv10, images: list[np.ndarray], conf_threshold: float, device: str, min_area: float) -> list[dict[str, Any]]:
    assert images

    try:
        detection_results = model.predict(
            images,
            imgsz=1024,
            conf=conf_threshold,
            device=device,
            batch=len(images),
        )

        all_results = []

        for i, result in enumerate(detection_results):
            image = images[i]
            boxes = results_to_boxes([result])
            boxes = filter_boxes(boxes, min_area=min_area, min_confidence=conf_threshold)

            segments = []
            for box in boxes:
                segment_img, position = extract_segment(image, box)
                if segment_img is not None:
                    _, buffer = cv2.imencode(".jpg", segment_img)
                    b64_segment = base64.b64encode(buffer).decode("utf-8")

                    segments.append({"image": b64_segment, "position": position, "class": box["class"], "confidence": box["confidence"]})

            all_results.append({"segments": segments, "total_segments": len(segments)})

        return all_results

    except Exception as e:
        print(f"Error in batch processing: {e}")
        return [{"error": str(e)}] * len(images)


def check_and_process_batch(context_dict: dict) -> None:
    context = context_dict["context"]
    model = context_dict["model"]

    if context.processing:
        return

    current_time = time.time()
    should_process = len(context.batch_queue) >= context.batch_size or (
        len(context.batch_queue) > 0 and current_time - context.last_process_time > context.processing_interval_seconds
    )

    if should_process:
        with context.lock:
            context_dict["context"] = context._replace(processing=True)
            context = context_dict["context"]

        try:
            batch_to_process = []
            image_ids = []

            with context.lock:
                for _ in range(min(context.batch_size, len(context.batch_queue))):
                    image_id, image = context.batch_queue[0]
                    context_dict["context"] = context._replace(batch_queue=context.batch_queue[1:])
                    context = context_dict["context"]
                    batch_to_process.append(image)
                    image_ids.append(image_id)

            results = process_batch(model, batch_to_process, context.conf_threshold, context.device, context.min_area)

            with context.lock:
                new_results = dict(context.results)
                for i, image_id in enumerate(image_ids):
                    new_results[image_id] = results[i]

                context_dict["context"] = context._replace(results=new_results, last_process_time=time.time())
                context = context_dict["context"]

        except Exception as e:
            print(f"Error in batch processing: {e}")

        finally:
            with context.lock:
                context_dict["context"] = context._replace(processing=False)


def process_image(context_dict: dict, image_data: np.ndarray) -> dict[str, Any]:
    context = context_dict["context"]
    image_id = f"{time.time()}_{id(image_data)}"

    with context.lock:
        new_batch_queue = list(context.batch_queue)
        new_batch_queue.append((image_id, image_data))
        context_dict["context"] = context._replace(batch_queue=new_batch_queue)

    check_and_process_batch(context_dict)

    max_wait_time = 30  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        context = context_dict["context"]
        with context.lock:
            if image_id in context.results:
                result = context.results[image_id]
                new_results = dict(context.results)
                del new_results[image_id]
                context_dict["context"] = context._replace(results=new_results)
                return result

        check_and_process_batch(context_dict)
        time.sleep(0.1)

    return {"error": "Processing timeout"}


def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        raise ValueError(f"Error decoding base64 image: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="YOLO Document Detection API", version="1.0.0")

    # Store context in a mutable container (dict) so we can update it
    context_dict = {"context": YoloContext(last_process_time=time.time()), "model": None}

    @app.get("/")
    async def root():
        return {"message": "YOLO Document Detection API. Use /detect endpoint to process images."}

    @app.post("/detect")
    async def detect_images(request: YoloRequest):
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided")

        # Ensure model is loaded
        if context_dict["model"] is None:
            context_dict["model"] = load_model(context_dict["context"])

        results = []

        for img_base64 in request.images:
            try:
                img = decode_base64_image(img_base64)
                result = process_image(context_dict, img)
                results.append(result)
            except ValueError as e:
                results.append({"error": str(e)})
            except Exception as e:
                results.append({"error": f"Internal error: {str(e)}"})

        return results

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

