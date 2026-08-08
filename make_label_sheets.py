"""Create pre-prediction contact sheets for blinded scene-category annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from segmentation_common import DATASETS, load_grouped_split


PROJECT = Path(__file__).resolve().parent


def font(size: int):
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/calibri.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--thumb", type=int, default=300)
    args = parser.parse_args()
    split = Path(args.split_file).resolve() if args.split_file else PROJECT / "splits" / f"{args.dataset}_grouped_v2.json"
    _train, _validation, test, info = load_grouped_split(split)
    data_root = Path(args.data_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_sheet = args.columns * args.rows
    title_height, label_height = 56, 48
    sheet_width = args.columns * args.thumb
    sheet_height = title_height + args.rows * (args.thumb + label_height)
    mapping = []
    for sheet_index, start in enumerate(range(0, len(test), per_sheet), 1):
        canvas = Image.new("RGB", (sheet_width, sheet_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 12), f"{args.dataset} blinded test scene audit — sheet {sheet_index}", fill="black", font=font(26))
        for local_index, pair in enumerate(test[start : start + per_sheet]):
            global_index = start + local_index + 1
            row, column = divmod(local_index, args.columns)
            x = column * args.thumb
            y = title_height + row * (args.thumb + label_height)
            image = Image.open(data_root / pair.image).convert("RGB")
            image.thumbnail((args.thumb, args.thumb), Image.Resampling.LANCZOS)
            left = x + (args.thumb - image.width) // 2
            top = y + (args.thumb - image.height) // 2
            canvas.paste(image, (left, top))
            draw.rectangle((x, y, x + args.thumb - 1, y + args.thumb + label_height - 1), outline="#555555", width=2)
            label = f"{global_index:02d}  {Path(pair.image).stem[:37]}"
            draw.text((x + 6, y + args.thumb + 7), label, fill="black", font=font(16))
            mapping.append({"index": global_index, "image": pair.image, "mask": pair.mask, "group": pair.group})
        canvas.save(output / f"{args.dataset}_test_sheet_{sheet_index:02d}.png")
    (output / f"{args.dataset}_test_mapping.json").write_text(
        json.dumps({"dataset": args.dataset, "split_sha256": info["sha256"], "images": mapping}, indent=2),
        encoding="utf-8",
    )
    print(f"created {len(list(output.glob(args.dataset + '_test_sheet_*.png')))} sheets for {len(test)} images")


if __name__ == "__main__":
    main()
