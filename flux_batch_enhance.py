import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image

# Load Flux Kontext model
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Set input and output directories
input_base_dir = "../cell_datasets/confocal_new"
output_base_dir = "../cell_datasets_Flux/confocal_new"

# Set prompt
# Modified prompt with stronger emphasis on noise control
prompt = "enhance microscopy image clarity and contrast selectively on cellular structures only, sharpen cell boundaries and organelles while preserving smooth background areas, improve definition of membranes and internal structures, increase visibility of fine cellular details, maintain clean background regions, preserve original colors exactly, selective enhancement of biological structures only, professional microscopy enhancement with noise reduction"
# Enhanced negative prompt for noise control
negative_prompt = "background noise amplification, grain enhancement, speckle artifacts, noise boost, texture noise, random pixel variation, color distortion, color shift, oversaturation, artificial coloring, excessive noise, over-processing artifacts, loss of cellular structure, blurred details, plastic appearance, unnatural smoothing, halo effects, ringing artifacts, background texture enhancement"

os.makedirs(output_base_dir, exist_ok=True)

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def process_directory(input_dir, output_dir):
    """Recursively process all images in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Iterate through all files and subdirectories in input directory
    for item in input_path.iterdir():
        if item.is_file():
            # Check if it's an image file
            if item.suffix.lower() in image_extensions:
                try:
                    print(f"Processing: {item}")

                    # Load image
                    input_image = load_image(str(item))

                    # Run enhancement
                    enhanced_image = pipe(
                        image=input_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=2.5,
                        num_inference_steps=28,
                    ).images[0]

                    # Save enhanced image
                    output_file = output_path / item.name
                    enhanced_image.save(str(output_file))
                    print(f"Saved: {output_file}")

                except Exception as e:
                    print(f"Error processing file {item}: {e}")

        elif item.is_dir():
            # Recursively process subdirectories
            sub_output_dir = output_path / item.name
            process_directory(str(item), str(sub_output_dir))


# Start batch processing
print(f"Starting batch processing for directory: {input_base_dir}")
print(f"Output directory: {output_base_dir}")

process_directory(input_base_dir, output_base_dir)

print("Batch processing completed!")
