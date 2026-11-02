"""Assemble the silent demo reel from sim clips + title cards.

    python demo_assets/make_demo.py

Reads the protein-assay and cell-culture clips from the simulator reports,
trims them, and sandwiches them between caption cards. Output:
demo_assets/demo.mp4 (~40 s, 640x480, 20 fps, no audio). The narrated
~3-minute cut follows demo_assets/SHOTS.md.
"""

import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "Code-as-a-Policy-Evolving-Skill" / "reports" / "videos"
OUT = Path(__file__).resolve().parent / "demo.mp4"
W, H, FPS = 640, 480, 20
BG = (26, 26, 46)
TITLE = (255, 255, 255)
SUB = (160, 220, 255)


def card(title, subtitle="", seconds=3):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    big = ImageFont.load_default(size=34)
    small = ImageFont.load_default(size=20)
    tb = draw.textbbox((0, 0), title, font=big)
    draw.text(((W - (tb[2] - tb[0])) / 2, H / 2 - 50), title, font=big,
              fill=TITLE)
    if subtitle:
        sb = draw.textbbox((0, 0), subtitle, font=small)
        draw.text(((W - (sb[2] - sb[0])) / 2, H / 2 + 10), subtitle,
                  font=small, fill=SUB)
    return [np.asarray(img)] * int(seconds * FPS)


def clip(name, start, count):
    reader = iio.get_reader(str(CLIPS / f"{name}.mp4"))
    frames = []
    for i, frame in enumerate(reader):
        if start <= i < start + count:
            frames.append(np.asarray(frame))
    reader.close()
    return frames


def main():
    parts = [
        card("lab2rob", "published protocol in, certified robot task out", 4),
        card("1. Compile", "ground steps, plan program + judge, build scene", 3),
        clip("protein_assay", 50, 300),
        card("2. Run long horizons", "cell culture: one simulated week", 3),
        clip("cell_culture", 100, 300),
        card("3. Certify", "predicates + VLM check + safety twin", 3),
        card("lab2rob", "257 tests green · $4.20 of $150 budget", 4),
    ]
    frames = [f for part in parts for f in part]
    print(f"writing {len(frames)} frames ({len(frames) / FPS:.0f}s) to {OUT}")
    writer = iio.get_writer(str(OUT), fps=FPS, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    sys.exit(main())
