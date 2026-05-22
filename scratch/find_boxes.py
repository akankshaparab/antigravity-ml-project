import numpy as np
from PIL import Image

# Load image
img_path = 'produc_vers/online_routing_flowchart.png'
img = Image.open(img_path).convert('RGB')
arr = np.array(img)

# Width, height
w, h = img.size

# Let's find contiguous regions of dark pixels (border)
# Specifically, we want the coordinates of the two boxes on the right.
# Let's inspect columns from 800 to 1024.
# Let's find vertical lines: columns that have many black pixels.
# Let's print out rows and columns where color is less than 50 (black border).
black_pixels = np.where(np.mean(arr, axis=2) < 50)
y_coords, x_coords = black_pixels

# Filter for x > 800
right_x = x_coords[x_coords > 800]
right_y = y_coords[x_coords > 800]

# Let's cluster these coordinates into two boxes:
# Box 1 is at the top right, Box 2 is at the bottom right.
# Let's separate by y coordinate (say y < 500 and y >= 500)
top_box_x = right_x[right_y < 500]
top_box_y = right_y[right_y < 500]

bottom_box_x = right_x[right_y >= 500]
bottom_box_y = right_y[right_y >= 500]

print("Top box bounding box:")
print(f"X: {np.min(top_box_x)} to {np.max(top_box_x)}")
print(f"Y: {np.min(top_box_y)} to {np.max(top_box_y)}")

print("Bottom box bounding box:")
print(f"X: {np.min(bottom_box_x)} to {np.max(bottom_box_x)}")
print(f"Y: {np.min(bottom_box_y)} to {np.max(bottom_box_y)}")
