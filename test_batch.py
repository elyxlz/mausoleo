import sys
from segment import load_model

# Get all paths
image_paths = ['test.jpeg', 'test2.jpeg', 'test3.jpeg']

# Load model
print("Loading model...")
model = load_model()

# Process batch
print(f"Processing batch of {len(image_paths)} images...")
results = model.predict(
    image_paths,
    imgsz=1024,
    conf=0.01,
    device='cpu',
    batch=len(image_paths)
)

# Print results info
print(f"Got {len(results)} results")
for i, result in enumerate(results):
    print(f"Result {i+1} has {len(result.boxes)} boxes")