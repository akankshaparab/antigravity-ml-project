import numpy as np
from PIL import Image

# Load image
img_path = 'produc_vers/online_routing_flowchart.png'
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
gray = np.mean(arr, axis=2)

# Width, height
w, h = img.size

# Let's search for horizontal lines:
print("Horizontal lines on the right:")
for y in range(h):
    line = gray[y, 800:1010] < 50
    max_run = 0
    current_run = 0
    start_x = -1
    best_start = -1
    for i, val in enumerate(line):
        if val:
            if current_run == 0:
                start_x = 800 + i
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                best_start = start_x
        else:
            current_run = 0
    if max_run > 120:
        print(f"Row {y}: horizontal line of length {max_run} starting at {best_start}")

# Let's search for vertical lines on the right:
print("\nVertical lines on the right:")
for x in range(800, 1010):
    line = gray[:, x] < 50
    max_run = 0
    current_run = 0
    start_y = -1
    best_start = -1
    for y, val in enumerate(line):
        if val:
            if current_run == 0:
                start_y = y
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                best_start = start_y
        else:
            current_run = 0
    if max_run > 60:
        print(f"Col {x}: vertical line of length {max_run} starting at {best_start}")
