import os
from PIL import Image, ImageDraw, ImageFont

def get_font(font_name, size):
    paths = [
        f"C:\\Windows\\Fonts\\{font_name}.ttf",
        f"C:\\Windows\\Fonts\\{font_name.lower()}.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def draw_centered_text(draw, box_coords, text, font, fill=(50, 50, 50)):
    # box_coords is (x1, y1, x2, y2)
    x1, y1, x2, y2 = box_coords
    w_box = x2 - x1 + 1
    h_box = y2 - y1 + 1
    
    # Split text into lines
    lines = text.split('\n')
    
    # Calculate line heights and widths
    line_widths = []
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    
    spacing = 5
    total_height = sum(line_heights) + spacing * (len(lines) - 1)
    
    # Start y to center vertically
    curr_y = y1 + (h_box - total_height) / 2
    
    # Draw each line
    for i, line in enumerate(lines):
        lw = line_widths[i]
        curr_x = x1 + (w_box - lw) / 2
        draw.text((curr_x, curr_y), line, font=font, fill=fill)
        curr_y += line_heights[i] + spacing

# Load original image
img_path = 'produc_vers/online_routing_flowchart.png'
img = Image.open(img_path).convert('RGB')
draw = ImageDraw.Draw(img)

# Clear top right box interior (Box 1: X=[850, 1006], Y=[410, 491])
# Fill with pure white, leaving the 1-pixel border intact
draw.rectangle([851, 411, 1005, 490], fill=(255, 255, 255))

# Clear bottom right box interior (Box 28: X=[850, 1006], Y=[526, 606])
draw.rectangle([851, 527, 1005, 605], fill=(255, 255, 255))

# Use Arial Bold to match the bold text in other boxes
font = get_font("arialbd", 15)

# Text definitions
top_text = "Free Models via\nOpenRouter\n(Tier 1)"
bottom_text = "Claude Haiku\n(Tier 2)"

# Draw text (using dark gray/black color to match the original text style)
draw_centered_text(draw, (850, 410, 1006, 491), top_text, font, fill=(0, 0, 0))
draw_centered_text(draw, (850, 526, 1006, 606), bottom_text, font, fill=(0, 0, 0))

# Save the updated image over the original file directly to apply the change
output_path = 'produc_vers/online_routing_flowchart.png'
img.save(output_path)
print(f"Updated flowchart saved to {output_path}")
