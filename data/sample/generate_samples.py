"""
generate_samples.py
-------------------
Creates coloured placeholder images in data/sample/ so the project
can be demoed without the full Kaggle dataset.

Run: python data/sample/generate_samples.py
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

OUTPUT_DIR = Path(__file__).parent
CATEGORIES = {
    "Tshirts":    (100, 160, 230),
    "Shirts":     (80,  180, 120),
    "Casual_Shoes":(210, 120, 80),
    "Watches":    (180, 100, 200),
    "Handbags":   (220, 180, 60),
}
N_PER_CLASS = 5


def make_placeholder(category: str, idx: int, color: tuple) -> Image.Image:
    img  = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    # Add a simple shape to differentiate items
    r = random.Random(idx)
    shape_x = r.randint(40, 140)
    shape_y = r.randint(40, 140)
    draw.rectangle([shape_x, shape_y, shape_x + 80, shape_y + 80],
                   fill=tuple(max(0, c - 60) for c in color), outline="white", width=2)
    draw.text((10, 200), f"{category} #{idx}", fill="white")
    return img


def main():
    for cat, color in CATEGORIES.items():
        cat_dir = OUTPUT_DIR / cat
        cat_dir.mkdir(exist_ok=True)
        for i in range(N_PER_CLASS):
            img  = make_placeholder(cat, i, color)
            path = cat_dir / f"{cat}_{i:03d}.jpg"
            img.save(path)
        print(f"Generated {N_PER_CLASS} samples → {cat_dir}")
    print("\n✅ Sample images ready in data/sample/")


if __name__ == "__main__":
    main()
