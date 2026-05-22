import numpy as np
from PIL import Image

# Load image
img_path = 'produc_vers/online_routing_flowchart.png'
img = Image.open(img_path).convert('RGB')
arr = np.array(img)

print(f"Image size: {img.size}")

# Let's find white and black areas to locate the boxes.
# The boxes have black borders (intensity close to 0) and white fill (intensity close to 255).
# Let's write a simple bounding box finder:
# We can find connected components or just scan for horizontal/vertical black lines.
# Since it's a clean computer-generated image, we can find contours by scanning for borders.

# Let's scan from right to left to find the two rightmost boxes.
# We will print out bounding boxes where there is a black border.
# Let's list some rows and columns with dark pixels.
gray = np.mean(arr, axis=2)
dark = gray < 100

# Find columns from right to left that contain dark pixels
w, h = img.size
print("Scanning right side for boxes...")
# Let's find bounding boxes of all rectangles
# A rectangle has a top border, bottom border, left border, right border.
# Let's find connected components of dark pixels (excluding arrows)
# Since we just want to find where "Gemini Flash (Tier 1)" and "Gemini Pro (Tier 2)" are:
# We know they are on the right side. Let's crop the right 30% of the image and find bounding boxes.
right_margin = int(w * 0.7)
right_half = dark[:, right_margin:]

# Find rows and cols of dark pixels in the right half
rows_with_dark = np.where(np.any(right_half, axis=1))[0]
cols_with_dark = np.where(np.any(right_half, axis=0))[0] + right_margin

# Let's find contiguous regions in rows/cols to find the boxes.
# Even simpler: let's save a copy of the image with a grid overlay so we can see the coordinates precisely!
# We can draw horizontal and vertical lines every 50 pixels with coordinates labeled.
from PIL import ImageDraw, ImageFont
grid_img = img.copy()
draw = ImageDraw.Draw(grid_img)
for x in range(0, w, 50):
    draw.line([(x, 0), (x, h)], fill=(200, 200, 200), width=1)
    draw.text((x, 5), str(x), fill=(100, 100, 100))
for y in range(0, h, 50):
    draw.line([(0, y), (w, y)], fill=(200, 200, 200), width=1)
    draw.text((5, y), str(y), fill=(100, 100, 100))

grid_img.save('scratch/grid_image.png')
print("Saved grid image to scratch/grid_image.png")
