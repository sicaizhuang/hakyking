from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUT = ASSETS / "hakyking.ico"
PNG_OUTPUT = ASSETS / "hakyking_256.png"


def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _rounded_rect_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def build_icon(size: int = 256) -> Image.Image:
    scale = size / 256.0
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        tuple(int(v * scale) for v in (16, 18, 240, 242)),
        radius=int(44 * scale),
        fill=(0, 0, 0, 135),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(int(8 * scale))))

    body = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = _rounded_rect_mask(size, int(44 * scale))
    gradient = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gradient)
    for y in range(size):
        t = y / max(1, size - 1)
        r = int(24 + 12 * t)
        g = int(31 + 15 * t)
        b = int(43 + 24 * t)
        gd.line((0, y, size, y), fill=(r, g, b, 255))
    body.alpha_composite(gradient)
    body.putalpha(mask)
    canvas.alpha_composite(body)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        tuple(int(v * scale) for v in (18, 18, 238, 238)),
        radius=int(42 * scale),
        outline=(113, 220, 255, 130),
        width=max(1, int(3 * scale)),
    )

    # Piano strip.
    key_y = int(172 * scale)
    key_h = int(42 * scale)
    key_x = int(42 * scale)
    key_w = int(172 * scale)
    white_count = 7
    single_w = key_w / white_count
    for i in range(white_count):
        x0 = key_x + int(i * single_w)
        x1 = key_x + int((i + 1) * single_w) - 2
        draw.rounded_rectangle(
            (x0, key_y, x1, key_y + key_h),
            radius=int(5 * scale),
            fill=(235, 242, 247, 238),
        )
    for i in (1, 2, 4, 5, 6):
        x = key_x + int(i * single_w - single_w * 0.28)
        draw.rounded_rectangle(
            (x, key_y, x + int(single_w * 0.46), key_y + int(key_h * 0.62)),
            radius=int(3 * scale),
            fill=(13, 16, 22, 240),
        )

    # Neon waveform.
    points = []
    import math

    for i in range(0, 160):
        x = int((48 + i) * scale)
        y = int(
            (
                116
                + 22 * math.sin(i / 9.0)
                + 9 * math.sin(i / 3.7)
            )
            * scale
        )
        points.append((x, y))
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.line(points, fill=(69, 196, 255, 150), width=max(2, int(9 * scale)), joint="curve")
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(int(4 * scale))))
    draw = ImageDraw.Draw(canvas)
    draw.line(points, fill=(117, 226, 255, 255), width=max(1, int(4 * scale)), joint="curve")
    for x, y in points[::22]:
        draw.ellipse(
            (x - int(4 * scale), y - int(4 * scale), x + int(4 * scale), y + int(4 * scale)),
            fill=(255, 123, 203, 255),
        )

    return canvas


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    base = build_icon(256)
    base.save(PNG_OUTPUT)
    base.save(
        OUTPUT,
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
