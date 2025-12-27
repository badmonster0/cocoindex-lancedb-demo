import argparse
import glob
import io
from pathlib import Path

import lancedb
import polars as pl
import torch
from PIL import Image
from lancedb.pydantic import LanceModel, Vector
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
LANCEDB_URI = "./recipe_lancedb"
TABLE_NAME = "recipes"
TEXT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"


def build_recipe_schema(text_dim: int, image_dim: int) -> type[LanceModel]:
    class Recipe(LanceModel):
        id: int
        title: str
        ingredients: list[str] | None
        instructions: str
        image_name: str | None
        image_path: str | None
        image: bytes | None
        instructions_vector: Vector(text_dim)  # type: ignore
        image_vector: Vector(image_dim, nullable=True)  # type: ignore

    return Recipe


def load_image_bytes(image_name: str) -> bytes | None:
    """
    Load image bytes from the images directory given an image name
    - Assumes *.jpg extension
    """
    candidate = IMAGES_DIR / f"{image_name}.jpg"
    if candidate.exists():
        return candidate.read_bytes()
    return None


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
) -> list[float]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)
        embeddings = mean_pooling(output.last_hidden_state, inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().tolist()


def embed_image(
    image_bytes: bytes,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
) -> list[float]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")  # type: ignore[call-arg]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features[0].cpu().tolist()


def load_models() -> tuple[
    PreTrainedTokenizerBase, PreTrainedModel, CLIPProcessor, CLIPModel, torch.device
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        TEXT_MODEL_NAME
    )
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)
    text_model.to(device)
    text_model.eval()

    image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)
    image_model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME, trust_remote_code=True)
    image_model.to(device)    # type: ignore
    image_model.eval()

    return text_tokenizer, text_model, image_processor, image_model, device


def main(overwrite: bool) -> None:
    """
    Upsert data into LanceDB from JSON files in the data directory.
    """
    text_tokenizer, text_model, image_processor, image_model, device = load_models()
    text_dim = text_model.config.hidden_size
    image_dim = image_model.config.projection_dim
    recipe_schema = build_recipe_schema(text_dim, image_dim)

    db = lancedb.connect(LANCEDB_URI)
    if overwrite:
        # Sometimes, we may want to do a fresh start and overwrite existing table
        table = db.create_table(TABLE_NAME, schema=recipe_schema, mode="overwrite")
    else:
        # By default, we append to existing table
        table = db.open_table(TABLE_NAME)

    # Find all JSON files in the data directory
    files = glob.glob(str(DATA_DIR / "recipes_*.json"))
    for file in files:
        print(f"Loading {file}...")
        payload = pl.read_json(file).to_dicts()
        # Gather the image bytes for each item and add to payload to ingest into LanceDB
        for item in payload:
            image_name = item.get("image_name") or ""
            image_bytes = load_image_bytes(image_name)
            item["image"] = image_bytes
            instructions = item.get("instructions")
            if not isinstance(instructions, str):
                instructions = ""
            item["instructions_vector"] = embed_text(
                instructions, text_tokenizer, text_model, device
            )
            if image_bytes is None:
                item["image_vector"] = None
            else:
                item["image_vector"] = embed_image(
                    image_bytes, image_processor, image_model, device
                )
        print(payload)
        # Upsert payload (insert or update if it exists)
        # https://docs.lancedb.com/tables/update
        (
            # Upsert by id
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(payload)
        )
    print(f"LanceDB table now has {table.count_rows()} rows.")

    # Search for recipes particular strings in the Ingredients
    q1 = table.search().limit(10).to_polars()
    print(q1.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ingest recipe data into LanceDB")
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite an existing table instead of appending to it.",
    )

    args = parser.parse_args()

    main(overwrite=args.overwrite)
