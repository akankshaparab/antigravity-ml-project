import sys

try:
    from PIL import Image, ImageDraw, ImageFont
    print("PIL is installed!")
except ImportError:
    print("PIL is NOT installed.")

try:
    import matplotlib
    print("matplotlib is installed!")
except ImportError:
    print("matplotlib is NOT installed.")

try:
    import cv2
    print("OpenCV (cv2) is installed!")
except ImportError:
    print("OpenCV is NOT installed.")
