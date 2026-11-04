#!/usr/bin/env python3
"""Build demo_assets/reel_tasks.mp4: protocol step -> compiled task -> robot demo, twice.

    python demo_assets/make_task_reel.py

Needs Pillow and ffmpeg. Uses the task demos in demo_assets/ (task_*_demo.mp4).
Title cards are drawn here; the demos are sped up to keep the reel short.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 20
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, FG, DIM, ACCENT, OK = (18, 20, 24), (236, 236, 236), (150, 156, 166), (110, 190, 255), (120, 210, 140)

CARDS = {
    "title": (4, [
        ("lab2rob", BOLD, 64, FG),
        ("published wet-lab protocols  ->  validated robot benchmark tasks", FONT, 30, DIM),
        ("", FONT, 20, FG),
        ("10,149 protocols compiled offline in about 100 s", FONT, 28, FG),
        ("every task frozen only if its demo passes an independent judge,", FONT, 26, FG),
        ("replays identically twice, and doing nothing fails", FONT, 26, FG),
    ]),
    "weigh": (7, [
        ("1. A protocol step", BOLD, 34, ACCENT),
        ('"Using an analytical balance, weigh out 0.9 g sodium chloride (NaCl)..."', FONT, 24, FG),
        ("SARS-CoV-2 RNA from wastewater (protocols.io)", FONT, 22, DIM),
        ("", FONT, 14, FG),
        ("compiled:  weigh_sample(mass_g=0.9, reagent=NaCl)", MONO, 26, FG),
        ("judge:     mass_on_balance(0.90 g +- 0.15 g)   (written without the program)", MONO, 22, FG),
        ("scene:     weighing bench, 68 salt grains, spring balance   accepted", MONO, 22, OK),
        ("", FONT, 14, FG),
        ("Two arms: hand the vial over, then pour in small tilts reading the balance.", FONT, 24, FG),
        ("Scripted demo, 4x speed.", FONT, 22, DIM),
    ]),
    "centrifuge": (7, [
        ("2. Another protocol step", BOLD, 34, ACCENT),
        ('"Centrifuge the tubes at 12,000xg for 3 min at room temperature"', FONT, 24, FG),
        ("DNA extraction for low-biomass samples (Protocol Exchange)", FONT, 22, DIM),
        ("", FONT, 14, FG),
        ("compiled:  centrifuge(tubes=2*, speed=12,000xg, time_s=180)", MONO, 26, FG),
        ("judge:     rotor_balanced(2), lid_closed", MONO, 22, FG),
        ("scene:     microcentrifuge (LabUtopia model), hinged latching lid   accepted", MONO, 22, OK),
        ("* the text gives no count: 2 assumed and recorded", FONT, 20, DIM),
        ("Two arms: drop tubes into opposite slots, tip the lid shut. Unbalanced + shut = 0.", FONT, 24, FG),
        ("Scripted demo, 3x speed.", FONT, 22, DIM),
    ]),
    "end": (6, [
        ("Benchmark-ready tasks", BOLD, 40, FG),
        ("20 Hz control, the arms' 14 joint and gripper targets", FONT, 26, FG),
        ("scored with a balance or a photo, partial credit for every outcome", FONT, 26, FG),
        ("Robocurve task format: task.json, scorer, frozen scene, demo, screenshot", FONT, 26, FG),
        ("", FONT, 20, FG),
        ("Across the corpus: 307 weighings -> 2 accepted, 15,519 spins -> 10,466 accepted;", FONT, 24, DIM),
        ("every rejection says what the scene is missing.", FONT, 24, DIM),
    ]),
}
CLIPS = {"weigh": (HERE / "task_weigh_demo.mp4", 4.0), "centrifuge": (HERE / "task_centrifuge_demo.mp4", 3.0)}


def card(lines) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    fonts = [ImageFont.truetype(path, size) for _, path, size, _ in lines]
    heights = [f.getbbox("Ag")[3] + 14 for f in fonts]
    y = (H - sum(heights)) // 2
    for (text, _, _, color), font, h in zip(lines, fonts, heights):
        x = 80
        draw.text((x, y), text, font=font, fill=color)
        y += h
    return img


def run(*args) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def main() -> None:
    work = Path(tempfile.mkdtemp())
    parts = []
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS)]
    scale = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black"
    for name in ("title", "weigh", "centrifuge", "end"):
        seconds, lines = CARDS[name]
        png = work / f"{name}.png"
        card(lines).save(png)
        clip = work / f"{name}_card.mp4"
        run("-loop", "1", "-t", str(seconds), "-i", str(png), "-vf", scale, *enc, str(clip))
        parts.append(clip)
        if name in CLIPS:
            src, speed = CLIPS[name]
            fast = work / f"{name}_demo.mp4"
            run("-i", str(src), "-an", "-vf", f"setpts=PTS/{speed},{scale}", *enc, str(fast))
            parts.append(fast)
    listing = work / "parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    run("-f", "concat", "-safe", "0", "-i", str(listing), *enc, "-movflags", "+faststart",
        str(HERE / "reel_tasks.mp4"))
    print(HERE / "reel_tasks.mp4")


if __name__ == "__main__":
    main()
