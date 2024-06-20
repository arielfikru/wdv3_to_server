# wdv3-jax

A small example illustrating how to use JAX/FLAX for running the WD Tagger V3 models with added support for bulk processing of images in directories, including recursively through subdirectories.

Base code adapted from https://github.com/neggles/wdv3-timm

The Models directory has been incorporated from https://github.com/SmilingWolf/JAX-CV  

## How To Use

1. Clone the repository and navigate to the directory:
```
git clone https://github.com/arielfikru/wdv3-jax.git
cd wd3-jax
```

2. Create a virtual environment and install the required Python packages.

For Linux users, a script is provided for convenience:
```
bash setup.sh
```

Windows users (or those preferring manual setup) can follow these steps:
```
# Create the virtual environment
python -m venv .venv
# Activate the environment
source .venv/bin/activate
# Update pip, setuptools, and wheel
python -m pip install -U pip setuptools wheel
# Optionally, manually install JAX (e.g., for nVidia GPU users)
python -m pip install -U "jax[cpu]"
# Install the remaining requirements
python -m pip install -r requirements.txt
```

For additional details on JAX installation with GPU/TPU/ROCm/Metal support, see https://github.com/google/jax?tab=readme-ov-file#installation

3. To run the script, choose one of the 3 models and provide an image file or a directory path as input. The script now supports processing multiple images in a directory and can optionally process images in subdirectories recursively.
```
python wdv3_jax.py --model <swinv2|convnext|vit> --gen_threshold <between 0.0 to 1.0> --char_threshold <between 0.0 to 1.0> <path/to/input, can be image or dir path> --recursive=True
```

Example output for `python wdv3_jax.py --model "vit" --gen_threshold=0.3 --char_threshold 0.6 "./animuData" --recursive=True` might look like this:
```
Processing image: path/to/directory/image1.png
Saved tags to image1.txt
Processing image: path/to/directory/subdirectory/image2.jpg
Saved tags to image2.txt
...
Done!
```

The output files (`image1.txt`, `image2.txt`, etc.) will contain a unique set of tags merged from the "Character tags" and general tags, ensuring no duplicates and reflecting the enhanced capabilities of the script.
