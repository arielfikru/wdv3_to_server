import json
import os
import time
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import flax
import jax
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, upload_file
from PIL import Image
import Models
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__, static_folder='static', static_url_path='')

# Ensure dataset directory exists
dataset_dir = os.path.join(os.getcwd(), 'Dataset')
os.makedirs(dataset_dir, exist_ok=True)

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
    input_path: Path
    model: str = "vit"
    gen_threshold: float = 0.35
    char_threshold: float = 0.75
    recursive: bool = False

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
    return final_caption, character_tags_str

def upload_to_huggingface(image_path, txt_path, target_caption_append):
    target_path = f"{target_caption_append}/{os.path.basename(image_path)}"
    token = ""
    repo_id = "Alterneko/dataset_fav"
    
    upload_file(
        path_or_fileobj=image_path,
        path_in_repo=target_path,
        repo_id=repo_id,
        token=token,
        repo_type="dataset"
    )

    upload_file(
        path_or_fileobj=txt_path,
        path_in_repo=target_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'),
        repo_id=repo_id,
        token=token,
        repo_type="dataset"
    )

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    target_filename = request.form.get('target_filename', '')
    target_caption = request.form.get('target_caption', '')

    if file and target_filename:
        # Secure the filename
        filename = secure_filename(file.filename)
        ext = filename.split('.')[-1]
        timestamp = int(time.time())
        random_number = ''.join(random.choices(string.digits, k=8))
        saved_filename = f"{timestamp}_-_{target_filename}_-_{random_number}.{ext}"
        save_path = os.path.join(dataset_dir, saved_filename)
        
        file.save(save_path)

        # Process image to generate caption
        def process_and_upload():
            try:
                opts = ScriptOptions(
                    input_path=Path(save_path),
                    model="convnext",
                    gen_threshold=0.35,
                    char_threshold=0.5,
                    recursive=False
                )
                repo_id = MODEL_REPO_MAP.get(opts.model)
                print(f"Loading model '{opts.model}' from '{repo_id}'...")
                model, target_size = load_model_hf(repo_id=repo_id)

                print("Loading tag list...")
                labels: LabelData = load_labels_hf(repo_id=repo_id)

                caption, character_tags_str = process_image(Path(save_path), model, target_size, labels, opts)

                # Modify the caption file to include target-caption-append
                txt_filename = save_path.replace(f".{ext}", ".txt")
                if os.path.exists(txt_filename):
                    with open(txt_filename, 'r') as f:
                        tags = f.read().split(", ")
                    tags.insert(0, target_caption)
                    with open(txt_filename, 'w') as f:
                        f.write(", ".join(tags))
                    caption = ", ".join(tags)

                    # Generate new filename with character tags
                    if character_tags_str:
                        new_filename = f"{timestamp}_-_{target_filename}_-_{character_tags_str}_-_{random_number}.{ext}"
                    else:
                        new_filename = f"{timestamp}_-_{target_filename}_-_{random_number}.{ext}"

                    new_save_path = os.path.join(dataset_dir, new_filename)
                    os.rename(save_path, new_save_path)
                    os.rename(txt_filename, new_save_path.replace(f".{ext}", ".txt"))

                    # Upload files to Hugging Face
                    upload_to_huggingface(new_save_path, new_save_path.replace(f".{ext}", ".txt"), target_caption)

                    return {'caption': caption}, 200
                else:
                    return {'error': 'Caption file not found'}, 500
            except Exception as e:
                return {'error': str(e)}, 500

        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_and_upload)
            result = future.result()
            return jsonify(result), result[1]

    return jsonify({'error': 'Failed to save file'}), 500

if __name__ == '__main__':
    app.run(debug=True)
