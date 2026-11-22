"""Build OS Studio's full-bleed lock-screen artwork; no rounded tile/padding."""
from pathlib import Path
import argparse
from PIL import Image, ImageDraw


def build_artwork(output: Path) -> None:
    size, scale = 1024, 3
    background = Image.new('RGB', (size, size))
    gradient = ImageDraw.Draw(background)
    amber, mint = (254, 189, 73), (154, 221, 216)
    for y in range(size):
        fraction = y / (size - 1)
        color = tuple(round(a + (b - a) * fraction) for a, b in zip(amber, mint))
        gradient.line((0, y, size - 1, y), fill=color)

    # Preserve the original music-note silhouette, scaled out of its old inset tile.
    mask = Image.new('L', (size * scale, size * scale), 0)
    note = ImageDraw.Draw(mask)

    def point(x: int, y: int) -> tuple[int, int]:
        return round((x - 88) / 336 * size * scale), round((y - 84) / 346 * size * scale)

    def box(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        return (*point(x1, y1), *point(x2, y2))

    note.polygon([point(215, 187), point(348, 160), point(348, 207), point(215, 235)], fill=255)
    note.rectangle(box(202, 192, 224, 341), fill=255)
    note.rectangle(box(327, 177, 349, 314), fill=255)
    note.ellipse(box(155, 309, 225, 363), fill=255)
    note.ellipse(box(280, 282, 350, 336), fill=255)
    mask = mask.resize((size, size), Image.Resampling.LANCZOS)
    artwork = Image.composite(Image.new('RGB', (size, size), '#15130c'), background, mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    artwork.save(output, optimize=True)
    print(f'Built {output}: {size}x{size}, opaque RGB, edge-to-edge amber/mint')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'app/web/player-artwork.png')
    build_artwork(parser.parse_args().output)
