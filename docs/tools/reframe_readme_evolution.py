"""Start README evolution GIFs at a complete population without re-encoding motion.

Run after export-readme-evolution.mjs and encode_readme_media.py.
All original frames and their durations remain in the loop. Only the first
delta frame is replaced by its fully composited equivalent, using the exact
original palette. Requires Pillow.
"""

import io
import json
from pathlib import Path

from PIL import Image


def split_gif(data):
    """Return global metadata and complete extension/image frame packets."""
    assert data[:6] in (b"GIF89a", b"GIF87a")
    position = 13
    if data[10] & 128:
        position += 3 * (2 ** ((data[10] & 7) + 1))
    prefix = data[:position]
    frames, pending = [], b""

    def subblocks(offset):
        while data[offset]:
            offset += data[offset] + 1
        return offset + 1

    while data[position] != 59:
        start = position
        if data[position] == 33:
            label = data[position + 1]
            position = subblocks(position + 2)
            packet = data[start:position]
            if not frames and not pending and label != 249:
                prefix += packet
            else:
                pending += packet
        elif data[position] == 44:
            flags = data[position + 9]
            position += 10
            if flags & 128:
                position += 3 * (2 ** ((flags & 7) + 1))
            position = subblocks(position + 1)
            control = pending.index(b"!\xf9\x04")
            duration = int.from_bytes(pending[control + 4 : control + 6], "little") * 10
            frames.append((pending + data[start:position], duration))
            pending = b""
        else:
            raise ValueError(f"Unexpected GIF block at {position}")
    assert not pending
    return prefix, frames


def rotate_animation(path, offset_ms):
    prefix, frames = split_gif(path.read_bytes())
    elapsed, index = 0, 0
    while elapsed < offset_ms:
        elapsed += frames[index][1]
        index += 1
    assert elapsed == offset_ms and index < len(frames)
    with Image.open(path) as animation:
        palette = animation.getpalette()
        animation.seek(index)
        first = animation.convert("RGB")
    first.save(path.with_suffix(".png"))

    # Match colors exactly; generic quantization can change nearby flat colors.
    lookup = {bytes(palette[i : i + 3]): i // 3 for i in range(0, len(palette), 3)}
    rgb = first.tobytes()
    pixels = bytes(lookup[rgb[i : i + 3]] for i in range(0, len(rgb), 3))
    indexed = Image.frombytes("P", first.size, pixels)
    indexed.putpalette(palette)
    encoded = io.BytesIO()
    indexed.save(encoded, format="GIF", duration=frames[index][1], disposal=1, optimize=False)
    _, replacement = split_gif(encoded.getvalue())
    reordered = [replacement[0]] + frames[index + 1 :] + frames[:index]
    output = prefix + b"".join(packet for packet, _ in reordered) + b";"
    assert sum(duration for _, duration in reordered) == sum(duration for _, duration in frames)
    with Image.open(io.BytesIO(output)) as animation:
        assert animation.n_frames == len(frames)
        assert animation.convert("RGB").tobytes() == first.tobytes()
    path.write_bytes(output)
    return len(frames), len(output)


if __name__ == "__main__":
    assets = Path(__file__).resolve().parents[1] / "source" / "_static"
    manifest_path = assets / "readme-motion-source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for scene in manifest["scenes"]:
        if scene["kind"] != "evolution" or scene.get("presentationStartOffsetMs") == 3600:
            continue
        frames, size = rotate_animation(assets / (scene["name"] + ".gif"), 3600)
        scene.update(presentationStartOffsetMs=3600, encodedFrames=frames, bytes=size)
        print(f"{scene['name']}: {frames} original frames, complete population on entry")
    manifest["evolutionPresentation"] = (
        "The complete original loop starts at 3600 ms so the initial frame and reduced-motion still show a complete population. "
        "Original frame packets, palette, loop duration, and captions are retained without cropping or resizing."
    )
    manifest["evolutionAdaptation"] = manifest.get("evolutionAdaptation", "Direct capture of the production component; the static view shows inheritance.").replace(
        "the static view shows inheritance.", "the reduced-motion still shows a complete population."
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
