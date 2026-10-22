# python -m app.highlight dt/Test/pituitary/1322.jpg
from __future__ import annotations

import argparse
from pathlib import Path

from app.train import classify_image, is_tumor_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Срез, на котором искать опухоль.")
    parser.add_argument("--output", type=Path, default=Path("runs/highlighted.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, class_name = classify_image(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Класс: {class_name}")
    if is_tumor_class(class_name):
        print(f"Подсвеченный снимок: {args.output}")
    else:
        print(f"Опухоли нет, снимок без подсветки: {args.output}")


if __name__ == "__main__":
    main()
