import numpy as np
from PIL import Image
from collections import deque

# Load image
img_path = 'produc_vers/online_routing_flowchart.png'
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
h, w, c = arr.shape

# Threshold to binary: 1 if dark, 0 if white/light
# Background and box interiors are white. Borders and text are dark.
binary = np.zeros((h, w), dtype=int)
for y in range(h):
    for x in range(w):
        # If any channel is < 200, it's dark
        if np.any(arr[y, x] < 200):
            binary[y, x] = 1

# Flood fill from (0, 0) to find background
# Background will be marked as 2.
queue = deque([(0, 0)])
binary[0, 0] = 2

while queue:
    cy, cx = queue.popleft()
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = cy + dy, cx + dx
        if 0 <= ny < h and 0 <= nx < w:
            if binary[ny, nx] == 0:
                binary[ny, nx] = 2
                queue.append((ny, nx))

# Now, pixels with value 0 are inside the boxes.
# Let's find connected components of these 0s.
visited = np.zeros((h, w), dtype=bool)
components = []

for y in range(h):
    for x in range(w):
        if binary[y, x] == 0 and not visited[y, x]:
            # Found a new component (inside of a box)
            comp_pixels = []
            q = deque([(y, x)])
            visited[y, x] = True
            while q:
                cy, cx = q.popleft()
                comp_pixels.append((cy, cx))
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if binary[ny, nx] == 0 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
            components.append(comp_pixels)

print(f"Found {len(components)} boxes:")
for idx, comp in enumerate(components):
    ys = [p[0] for p in comp]
    xs = [p[1] for p in comp]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    print(f"Box {idx+1}: X=[{min_x}, {max_x}] (W={width}), Y=[{min_y}, {max_y}] (H={height})")
