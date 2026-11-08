"""Encode rendered README frames with a shared palette per animation.

Usage: python docs/tools/encode_readme_media.py /path/to/frames
Requires Pillow. See export-readme-media.mjs for the renderer.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

frames_root = Path(sys.argv[1]).resolve()
assets = Path(__file__).resolve().parents[1] / "source" / "_static"
manifest = json.loads((frames_root / "manifest.json").read_text(encoding="utf-8"))
frame_duration = manifest["frameDurationMs"]
assert frame_duration % 10 == 0, "GIF durations must be whole centiseconds"
for scene in manifest["scenes"]:
    folder = frames_root / scene["name"]
    indices = list(range(0, scene["frames"], 20))
    width, height = scene["width"], scene["height"]
    samples = Image.new("RGB", (width, height * len(indices)))
    for row, index in enumerate(indices):
        with Image.open(folder / f"{index:04}.png") as image:
            samples.paste(image.convert("RGB"), (0, row * height))
    palette = samples.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    # Keep common flat surfaces exact instead of averaging them with antialiasing colors.
    frequent = sorted(samples.getcolors(samples.width * samples.height), reverse=True)[:8]
    colors = palette.getpalette()[: 128 * 3] + [value for _, rgb in frequent for value in rgb]
    palette.putpalette(colors + [0] * (768 - len(colors)))
    frames = []
    for index in range(scene["frames"]):
        with Image.open(folder / f"{index:04}.png") as image:
            rgb = image.convert("RGB")
            indexed = rgb.quantize(palette=palette, dither=Image.Dither.NONE)
            # Pillow's palette lookup cache groups nearby colors; restore exact flat fills.
            for offset, (_, color) in enumerate(frequent[:2]):
                red, green, blue = ImageChops.difference(rgb, Image.new("RGB", rgb.size, color)).split()
                mask = ImageChops.lighter(ImageChops.lighter(red, green), blue).point(lambda value: 255 if value == 0 else 0)
                indexed.paste(128 + offset, mask=mask)
            frames.append(indexed)
    destination = assets / (scene["name"] + ".gif")
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        # Emit one global table instead of repeating a local palette in every frame.
        palette=palette.getpalette(),
        optimize=True,
        disposal=1,
    )
    with Image.open(destination) as animation:
        duration = 0
        for index in range(animation.n_frames):
            animation.seek(index)
            duration += animation.info["duration"]
        assert duration == scene["durationMs"]
        assert animation.size == (width, height)
        assert animation.info["loop"] == 0
        scene.update(encodedFrames=animation.n_frames, bytes=destination.stat().st_size)
    print(f"{scene['name']}: {scene['bytes']:,} bytes", flush=True)
(assets / "readme-motion-source.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
