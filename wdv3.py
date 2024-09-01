import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, List, Dict

import flax
import jax
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import ArgumentParser, field
from tqdm import tqdm

import Models

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}

@flax.struct.dataclass
class PredModel:
    apply_fun: Callable = flax.struct.field(pytree_node=False)
    params: Any = flax.struct.field(pytree_node=True)

    @jax.jit
    def predict(self, x):
        x = x / 127.5 - 1
        x = self.apply_fun(self.params, x, train=False)
        return jax.nn.sigmoid(x)

@dataclass
class LabelData:
    names: list[str]
    rating: list[int]
    general: list[int]
    character: list[int]

@dataclass
class ScriptOptions:
    input_path: Path = field(positional=True)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=1.00)
    batch_size: int = field(default=4)

def load_labels_hf(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> LabelData:
    try:
        csv_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token)
        df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
        return LabelData(
            names=df["name"].tolist(),
            rating=list(np.where(df["category"] == 9)[0]),
            general=list(np.where(df["category"] == 0)[0]),
            character=list(np.where(df["category"] == 4)[0]),
        )
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

def load_model_hf(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> PredModel:
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.msgpack", revision=revision, token=token)
    model_config = hf_hub_download(repo_id=repo_id, filename="sw_jax_cv_config.json", revision=revision, token=token)

    with open(weights_path, "rb") as f:
        restored = flax.serialization.msgpack_restore(f.read())["model"]
    variables = {"params": restored["params"], **restored["constants"]}

    with open(model_config) as f:
        model_config = json.load(f)

    model_name = model_config["model_name"]
    model_builder = Models.model_registry[model_name]()
    model = model_builder.build(config=model_builder, **model_config["model_args"])
    return PredModel(model.apply, params=variables), model_config["image_size"]

def preprocess_image(image_path: Path, target_size: int) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((target_size, target_size), Image.LANCZOS)
        return np.array(img)[..., ::-1]  # Convert to BGR

def get_tags(probs: np.ndarray, labels: LabelData, gen_threshold: float, char_threshold: float) -> List[str]:
    indices = np.where(probs > gen_threshold)[0]
    char_indices = np.intersect1d(indices, labels.character)
    gen_indices = np.intersect1d(indices, labels.general)
    
    char_tags = [labels.names[i] for i in char_indices if probs[i] > char_threshold]
    gen_tags = [labels.names[i] for i in gen_indices]
    
    all_tags = sorted(set(char_tags + gen_tags))
    
    # Remove banned words
    banned_words = {"1girl", "realistic", "cosplay"}
    return [tag for tag in all_tags if tag not in banned_words]

def process_batch(batch: List[Dict], model: PredModel, target_size: int, labels: LabelData, opts: ScriptOptions, input_path: Path):
    images = [preprocess_image(input_path / item['path'].lstrip('/'), target_size) for item in batch]
    inputs = np.stack(images)
    
    outputs = model.predict(inputs)
    outputs = np.array(outputs)  # Move from device to host memory
    
    for item, output in zip(batch, outputs):
        image_path = input_path / item['path'].lstrip('/')
        ai_tags = get_tags(output, labels, opts.gen_threshold, opts.char_threshold)
        
        subject = ",".join(item["subject"])
        json_tags = ",".join(item["tag"])
        
        unique_tags = f"{subject}, {json_tags}, 1girl, |||, "
        final_caption = unique_tags + ", ".join(ai_tags)
        
        txt_filename = f"{image_path.stem}.txt"
        with open(image_path.parent / txt_filename, "w") as f:
            f.write(final_caption)

def load_index_json(json_path: Path) -> List[Dict]:
    with open(json_path, 'r') as f:
        return json.load(f)

def main(opts: ScriptOptions):
    input_path = Path(opts.input_path).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {input_path}")

    index_json_path = input_path / "index.json"
    if not index_json_path.exists():
        raise FileNotFoundError(f"index.json not found in {input_path}")

    image_data = load_index_json(index_json_path)
    
    repo_id = MODEL_REPO_MAP.get(opts.model)
    print(f"Loading model '{opts.model}' from '{repo_id}'...")
    model, target_size = load_model_hf(repo_id=repo_id)

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Processing images...")
    for i in tqdm(range(0, len(image_data), opts.batch_size), desc="Batches", unit="batch"):
        batch = image_data[i:i+opts.batch_size]
        process_batch(batch, model, target_size, labels, opts, input_path)

    print("Processing complete!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ScriptOptions, dest="opts")
    args = parser.parse_args()
    opts = args.opts
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
