# Keeping your data in LanceDB fresh with CocoIndex

## Background

This repo contains a demo of using [CocoIndex](https://cocoindex.io/), a data transformation framework
the provides incremental processing and data lineage out-of-the-box, with [LanceDB](https://lancedb.com),
a multimodal lakehouse for AI.

The goal is to store a multimodal dataset (images + text) in LanceDB and keep it fresh with CocoIndex.

### Why use incremental processing?

Not all vector processing workloads are large batch workloads. Consider this scenario: you have a
user-facing application where users enter their recipes (along with images of the food/drink item that
they prepared), and you want to persist the data to a multimodal storage engine.
In this scenario, you typically don't begin with huge amounts of data. You accumulate
data over time, as users add their creations. And the volume/velocity of the data aren't staggeringly
high -- at times, there's ony a trickle of data coming in, and at times, there are minor spikes in
velocity.

For scenarios like this, incremental processing is an efficient technique that processes only new
or changed data (deltas) since the last update, rather than reprocessing entire large datasets. This
tends to reduce computation while lowering costs for near real-time
analytics. It's ideal for constantly evolving data sources, handling small batches of updates to
keep data fresh with less overhead than full batch loads.

## Dataset

We'll be using the [food ingredients and recipes](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)
dataset from Kaggle. The data contains 13k+ recipes and images of food/drinks scraped from the
Epicurious website. The dataset is multimodal, containing images, arrays and text.

Download the dataset from Kaggle to the local directory (it will be in a file named `archive.zip`).
Unzip the file at the root level of this repository.

## Setup

We'll use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the dependencies for
this project. Run the following command to install the required Python libraries to get started.

```bash
uv sync
```

## Generate data

To simulate a scenario where we have data intermittently coming in from a source, we'll use the
script `data_generator.py`. This script looks at the source data in the `archive` directory
and writes JSON records of the source data. The JSON records also contain a path to the image
file for the corresponding recipe ID, so that it can be easily located for ingestion into LanceDB.

```bash
uv run data_generator.py --start 0 --end 5
```

This writes out the first 5 recipe records to a JSON file in the path `data/*.json`. Simultaneously,
it also copies the image file into the `data/images/*.jpg` path.

To generate the data for the next 5 records, the corresponding start and end indices can be entered.

```bash
uv run data_generator.py --start 5 --end 10
```

To delete existing records and start afresh, use the `--refresh` flag.

```bash
uv run data_generator.py --start 0 --end 10 --refresh
```

Running the script multiple times will generate multiple JSON files, one for each run. This mimics
"real" data that may be coming from a push API in an application.

## Getting data into LanceDB

The `ingest.py` script contains code that ingests the recipe data in to a LanceDB table called
`recipes` that collocates the images and the recipe metadata in **one single table**. This is
one of the biggest benefits of LanceDB -- data that would otherwise be managed separately (as
in Parquet, where tables tend to store pointer URLs to the actual files in a separate directory).

The ingestion script also generates two kinds of embeddings:
- Text embeddings on the `instructions` column (TODO: concatenate the `title` and `instructions` and embed _that_ instead)
- Image embeddings on the `image` binary column

The text embeddings use the `nomic-ai/nomic-embed-text-v1.5` model, and the image embeddings use
the `openai/clip-vit-base-patch32` model, both accessed via the Hugging Face transformers library.

Run the script as follows:

```bash
# Overwrite the existing database
uv run ingest.py -o
# Or, append to an existing database (default mode)
uv run ingest.py
```

An upsert pipeline is used during ingestion, so that duplicate data isn't written to the table.
This means that as the script is run multiple times (as new data comes in), only records that have
a new unique `id` field for the recipe are written to the table.

## Querying the database

The `query.py` script contains sample code to query the data once it's persisted to LanceDB.

```bash
uv run query.py
```

Two kinds of queries are run:
- Query via a text embedding on the `instruction_vector` column
- Query via a text-to-image embedding on the `image_vector` column

Each should return relevant `top-k` results based on the query.