import json
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Callable, Optional

import flax
import jax
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import ArgumentParser, field

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

    def jit_predict(self, x):
        x = x / 127.5 - 1
        x = self.apply_fun(self.params, x, train=False)
        x = flax.linen.sigmoid(x)
        x = jax.numpy.float32(x)
        return x

    def predict(self, x):
        preds = self.jit_predict(x)
        preds = jax.device_get(preds)
        preds = preds[0]
        return preds

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas

def pil_resize(image: Image.Image, target_size: int) -> Image.Image:
    max_dim = max(image.size)
    if max_dim != target_size:
        image = image.resize(
            (target_size, target_size),
            Image.BICUBIC,
        )
    return image

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]

def load_labels_hf(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> LabelData:
    try:
        csv_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token)
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data

def load_model_hf(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> PredModel:
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.msgpack", revision=revision, token=token)

    model_config = hf_hub_download(repo_id=repo_id, filename="sw_jax_cv_config.json", revision=revision, token=token)

    with open(weights_path, "rb") as f:
        data = f.read()

    restored = flax.serialization.msgpack_restore(data)["model"]
    variables = {"params": restored["params"], **restored["constants"]}

    with open(model_config) as f:
        model_config = json.loads(f.read())

    model_name = model_config["model_name"]
    model_builder = Models.model_registry[model_name]()
    model = model_builder.build(
        config=model_builder,
        **model_config["model_args"],
    )
    model = PredModel(model.apply, params=variables)
    return model, model_config["image_size"]

def get_tags(probs: Any, labels: LabelData, gen_threshold: float, char_threshold: float):
    probs = list(zip(labels.names, probs))
    rating_labels = dict([probs[i] for i in labels.rating])
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")
    return caption, taglist, rating_labels, char_labels, gen_labels

@dataclass
class ScriptOptions:
    input_path: Path = field(positional=True)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)
    recursive: bool = field(default=False)

def process_image(image_path: Path, model, target_size, labels, opts: ScriptOptions):
    print(f"Processing image: {image_path}")
    img_input: Image.Image = Image.open(image_path)
    img_input = pil_ensure_rgb(img_input)
    img_input = pil_pad_square(img_input)
    img_input = pil_resize(img_input, target_size)
    inputs = np.array(img_input)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = inputs[..., ::-1]
    outputs = model.predict(inputs)
    caption, taglist, _, character_tags, _ = get_tags(
        probs=outputs,
        labels=labels,
        gen_threshold=opts.gen_threshold,
        char_threshold=opts.char_threshold,
    )

    character_tags_str = ""

    if character_tags:
        character_tags_str = ", ".join(character_tags.keys())
        merged = f"{character_tags_str}, {caption}"
    else:
        merged = caption

    merged_list = list(set(merged.split(", ")))
    final_caption = ", ".join(filter(None, merged_list))
    
    txt_filename = f"{image_path.stem}.txt"
    with open(image_path.parent / txt_filename, "w") as f:
        f.write(final_caption)
    print(f"Saved tags to {txt_filename}")

def find_images(directory: Path, recursive: bool) -> list:
    patterns = ["*.jpg", "*.png", "*.jpeg"]  # Extend with more formats as needed
    images = []
    if recursive:
        for pattern in patterns:
            images.extend(directory.rglob(pattern))
    else:
        for pattern in patterns:
            images.extend(directory.glob(pattern))
    return images

def main(opts: ScriptOptions):
    input_path = Path(opts.input_path).resolve()
    if input_path.is_dir():
        image_paths = find_images(input_path, opts.recursive)
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    repo_id = MODEL_REPO_MAP.get(opts.model)
    print(f"Loading model '{opts.model}' from '{repo_id}'...")
    model, target_size = load_model_hf(repo_id=repo_id)

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    for image_path in image_paths:
        process_image(image_path, model, target_size, labels, opts)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ScriptOptions, dest="opts")
    args = parser.parse_args()
    opts = args.opts
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
