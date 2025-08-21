import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sdr.processing_apis.yolo import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def test_image():
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Draw a rectangle (document-like region)
    cv2.rectangle(img, (100, 100), (400, 200), (0, 0, 0), -1)
    # Draw another rectangle (section-like region)
    cv2.rectangle(img, (150, 300), (350, 450), (0, 0, 255), -1)

    return img


@pytest.fixture
def base64_image(test_image):
    _, buffer = cv2.imencode(".jpg", test_image)
    return base64.b64encode(buffer).decode("utf-8")


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "YOLO Document Detection API" in response.json()["message"]


def test_detect_endpoint_empty_request(client):
    response = client.post("/detect", json={"images": []})
    assert response.status_code == 400
    assert "No images provided" in response.json()["detail"]


def test_detect_endpoint_invalid_base64(client):
    response = client.post("/detect", json={"images": ["not-valid-base64"]})
    assert response.status_code == 200
    assert "error" in response.json()[0]


def test_detect_endpoint_single_image(client, base64_image, monkeypatch):
    # Mock the YOLO model to avoid actual inference
    from unittest.mock import MagicMock, patch
    
    # Create mock model and results
    mock_result = MagicMock()
    mock_result.boxes.xyxy = [np.array([[100, 100, 400, 200]])]
    mock_result.boxes.cls = [np.array([0])]
    mock_result.boxes.conf = [np.array([0.95])]
    mock_result.names = {0: "text"}
    
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    
    # Patch load_model to return our mock
    with patch("src.sdr.processing_apis.yolo.load_model", return_value=mock_model):
        response = client.post("/detect", json={"images": [base64_image]})
        
        assert response.status_code == 200
        
        # Verify response structure
        result = response.json()[0]
        assert "segments" in result
        assert "total_segments" in result
        
        # Verify segments structure
        if result["segments"]:
            segment = result["segments"][0]
            assert "image" in segment
            assert "position" in segment
            assert "class" in segment
            assert "confidence" in segment
            
            # Verify position structure
            position = segment["position"]
            assert all(k in position for k in ["x1", "y1", "x2", "y2", "width", "height"])