import argparse
import ast
import json
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
OUTPUT_IMAGES_DIR = DATA_DIR / "images"
SOURCE_DATA_DIR = Path("archive")
SOURCE_IMAGES_DIR = SOURCE_DATA_DIR / "Food Images" / "Food Images"
RECIPES_CSV = SOURCE_DATA_DIR / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_recipes_df(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    df = df.rename({col: col.lower() for col in df.columns}).rename({"": "id"})
    df = df.with_columns(
        pl.col("cleaned_ingredients").map_elements(ast.literal_eval, return_dtype=pl.List(pl.Utf8)),
    )
    return df


def resolve_image_path(image_name: str) -> Path:
    candidate = SOURCE_IMAGES_DIR / image_name
    if candidate.exists():
        return candidate
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = SOURCE_IMAGES_DIR / f"{image_name}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(SOURCE_IMAGES_DIR.glob(f"{image_name}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No image found for {image_name!r} in {SOURCE_IMAGES_DIR}")


def build_recipes(df: pl.DataFrame, start: int, end: int) -> list[dict]:
    if start < 0 or end <= start:
        raise ValueError("start must be >= 0 and end must be greater than start")
    if end > df.height:
        raise ValueError(f"end must be <= {df.height}")

    subset = df.slice(start, end - start)
    recipes = []
    for row in subset.to_dicts():
        image_name = row.get("image_name") or ""
        image_path = resolve_image_path(image_name)
        output_image_path = OUTPUT_IMAGES_DIR / image_path.name
        recipes.append(
            {
                "id": int(row["id"]),
                "title": row["title"],
                "ingredients": row["cleaned_ingredients"],
                "instructions": row["instructions"],
                "image_name": image_name,
                "image_path": output_image_path.as_posix(),
            }
        )
    return recipes


def copy_images(recipes: list[dict]) -> None:
    for recipe in recipes:
        source = resolve_image_path(recipe["image_name"])
        destination = Path(recipe["image_path"])
        if not destination.exists():
            destination.write_bytes(source.read_bytes())


def write_json(recipes: list[dict]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DATA_DIR / f"recipes_{timestamp}.json"
    output_path.write_text(json.dumps(recipes, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a subset of recipe data.")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive).")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete the data directory before generating new output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    ensure_output_dirs()
    df = load_recipes_df(RECIPES_CSV)
    recipes = build_recipes(df, args.start, args.end)
    copy_images(recipes)
    output_path = write_json(recipes)
    print(f"Wrote {len(recipes)} recipes to {output_path}")


if __name__ == "__main__":
    main()
