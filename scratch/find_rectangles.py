import numpy as np
from PIL import Image

# Load image
img_path = 'produc_vers/online_routing_flowchart.png'
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
gray = np.mean(arr, axis=2)

# We want to find the rectangles on the right.
# Let's find white components (255, 255, 255) that are surrounded by black borders,
# or simply find the exact coordinates of the rectangular boxes by checking where the black borders are.
# Let's scan each row for continuous segments of white pixels flanked by black.
# Even simpler, let's just find the bounding box of the white area inside the boxes.

# Let's inspect the region x in [800, 1000], y in [350, 650]
# Let's print out the row averages or write a script to find the box coordinates by detecting the rectangular borders.
# A border is a line of black pixels.
# Let's find the vertical lines on the right:
# A vertical line at x has black pixels at many y.
# Let's print the count of black pixels for each column x in range(800, 1010)
print("Black pixel count per column:")
for x in range(800, 1010):
    black_count = np.sum(gray[:, x] < 50)
    if black_count > 20:
        print(f"Col {x}: {black_count} black pixels")

print("\nBlack pixel count per row in range(350, 650):")
for y in range(350, 650):
    black_count = np.sum(gray[y, 800:1010] < 50)
    if black_count > 20:
        print(f"Row {y}: {black_count} black pixels")
