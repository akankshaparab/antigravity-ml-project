import os

diff_path = r"c:\Users\Dell1234\.gemini\antigravity\scratch\ml_project\scratch\full_diff.txt"
if os.path.exists(diff_path):
    # Try reading with utf-16-le first, then fallback to utf-8
    content = ""
    try:
        with open(diff_path, "r", encoding="utf-16") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(diff_path, "r", encoding="utf-8") as f:
            content = f.read()
            
    print(content)
else:
    print("Diff file not found.")
