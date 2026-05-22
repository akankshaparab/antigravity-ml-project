import os
from PIL import Image, ImageDraw, ImageFont

def get_font(font_name, size):
    paths = [
        f"C:\\Windows\\Fonts\\{font_name}.ttf",
        f"C:\\Windows\\Fonts\\{font_name.lower()}.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

# Create dummy image to draw on
img = Image.new('RGB', (200, 200), (255, 255, 255))
draw = ImageDraw.Draw(img)

# Let's test arialbd (Arial Bold)
font_bold = get_font("arialbd", 16)
font_regular = get_font("arial", 16)

# Test strings
strings = [
    "Free Models via",
    "OpenRouter",
    "Free Models",
    "via OpenRouter",
    "Claude Haiku",
    "(Tier 1)",
    "(Tier 2)"
]

print("Bold Font (Size 16):")
for s in strings:
    bbox = draw.textbbox((0, 0), s, font=font_bold)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    print(f"'{s}': W={w}, H={h}")

print("\nRegular Font (Size 16):")
for s in strings:
    bbox = draw.textbbox((0, 0), s, font=font_regular)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    print(f"'{s}': W={w}, H={h}")

font_bold_15 = get_font("arialbd", 15)
print("\nBold Font (Size 15):")
for s in strings:
    bbox = draw.textbbox((0, 0), s, font=font_bold_15)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    print(f"'{s}': W={w}, H={h}")
