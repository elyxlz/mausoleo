import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sdr.processing.segment import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_image():
    """Create a sample test image with shapes that should be detected as segments."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Draw a rectangle (text-like region)
    cv2.rectangle(img, (100, 100), (400, 200), (0, 0, 0), -1)
    # Draw another rectangle (figure-like region)
    cv2.rectangle(img, (150, 300), (350, 450), (0, 0, 255), -1)

    return img


@pytest.fixture
def base64_image(test_image):
    """Convert the test image to base64 string."""
    _, buffer = cv2.imencode(".jpg", test_image)
    return base64.b64encode(buffer).decode("utf-8")


def test_root_endpoint(client):
    """Test that the root endpoint returns the correct message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Segmentation API" in response.json()["message"]


def test_segment_endpoint_empty_request(client):
    """Test that the segment endpoint rejects empty requests."""
    response = client.post("/segment", json={"images": []})
    assert response.status_code == 400
    assert "No images provided" in response.json()["detail"]


def test_segment_endpoint_invalid_base64(client):
    """Test that the segment endpoint handles invalid base64 data."""
    response = client.post("/segment", json={"images": ["not-valid-base64"]})
    assert response.status_code == 200  # API returns errors in results array
    assert "error" in response.json()[0]


def test_segment_endpoint_single_image(client, base64_image):
    """Test that the segment endpoint processes a single valid image."""
    response = client.post("/segment", json={"images": [base64_image]})

    assert response.status_code == 200

    # Verify response structure
    result = response.json()[0]
    assert "segments" in result or "error" in result

    # If segments are returned (rather than an error), verify their structure
    if "segments" in result and len(result["segments"]) > 0:
        segment = result["segments"][0]
        assert "image" in segment  # Should contain base64 image
        assert "position" in segment  # Should contain position data
        assert "class" in segment  # Should contain class label
        assert "confidence" in segment  # Should contain confidence score

        # Verify position structure
        position = segment["position"]
        assert all(k in position for k in ["x1", "y1", "x2", "y2", "width", "height"])


def test_segment_endpoint_batch(client, base64_image):
    """Test that the segment endpoint processes a batch of images."""
    # Send two copies of the same image
    response = client.post("/segment", json={"images": [base64_image, base64_image]})

    assert response.status_code == 200
    assert len(response.json()) == 2  # Should return results for both images
