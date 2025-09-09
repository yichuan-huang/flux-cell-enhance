import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
from PIL import Image

# Load Flux Kontext model
print("Loading Flux Kontext model...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print("✅ Model loaded successfully")

# Set input and output directories
input_base_dir = "../cell_datasets/confocal_new"
output_base_dir = "../cell_datasets_Flux/confocal_new"


def is_brightfield_folder(folder_name):
    """Check if folder contains bright-field images (A folders)"""
    pattern = r"(train|test)A(_|\d|$)"
    return bool(re.search(pattern, folder_name, re.IGNORECASE))


def is_stained_folder(folder_name):
    """Check if folder contains stained images (B folders)"""
    pattern = r"(train|test)B(_|\d|$)"
    return bool(re.search(pattern, folder_name, re.IGNORECASE))


# Define optimized prompts with stronger emphasis on structural preservation and minimal distortion
# For bright-field microscopy images (A folders) - improved edge definition while preserving structure
brightfield_prompt = "enhance microscopy image with subtle improvements, gently increase cellular boundary clarity, preserve original morphological structure, maintain authentic texture patterns, minimal noise reduction while keeping fine details intact"
brightfield_negative = "dramatic enhancement, structural deformation, artificial edge effects, over-sharpening, contrast artifacts, morphology alteration, heavy noise filtering, unnatural textures, processing artifacts"

# For stained cell images (B folders) - ultra-gentle enhancement to prevent distortion
stained_prompt = "very gently enhance stained microscopy image, preserve exact cellular morphology and structure, maintain original color fidelity, minimal contrast adjustment, keep authentic staining patterns, subtle detail clarification without alteration"
stained_negative = "color modification, structural changes, morphological distortion, contrast manipulation, saturation changes, artificial enhancement, processing distortion, staining pattern alteration, cellular deformation"

# Default prompts for unknown folder types
default_prompt = "very gently enhance microscopy image quality, minimal processing, preserve original structure"
default_negative = (
    "over-processing, artifacts, distortion, structural changes, unnatural enhancement"
)

os.makedirs(output_base_dir, exist_ok=True)

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def process_directory(input_dir, output_dir):
    """Recursively process all images in directory with specialized enhancement"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get folder name for image type detection
    folder_name = input_path.name

    # Iterate through all files and subdirectories in input directory
    for item in input_path.iterdir():
        if item.is_file():
            # Check if it's an image file
            if item.suffix.lower() in image_extensions:
                try:
                    # Determine image type and select appropriate prompts and parameters
                    if is_brightfield_folder(
                        folder_name
                    ):  # A folders - Bright-field images
                        current_prompt = brightfield_prompt
                        current_negative = brightfield_negative
                        guidance_scale = (
                            2.0  # Further reduced for more natural enhancement
                        )
                        num_steps = 30  # Reduced steps for gentler processing
                        image_type = "Bright-field"
                    elif is_stained_folder(folder_name):  # B folders - Stained images
                        current_prompt = stained_prompt
                        current_negative = stained_negative
                        guidance_scale = (
                            1.5  # Significantly reduced to prevent distortion
                        )
                        num_steps = 25  # Minimal steps to avoid over-processing
                        image_type = "Stained"
                    else:
                        # Default fallback for unknown folder types
                        current_prompt = default_prompt
                        current_negative = default_negative
                        guidance_scale = 1.8
                        num_steps = 28
                        image_type = "Unknown"

                    print(f"Processing {image_type}: {item}")
                    print(
                        f"  Conservative parameters: guidance_scale={guidance_scale}, steps={num_steps}"
                    )

                    # Load image
                    input_image = load_image(str(item))

                    # Run enhancement with specialized parameters
                    enhanced_image = pipe(
                        image=input_image,
                        prompt=current_prompt,
                        negative_prompt=current_negative,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                    ).images[0]

                    # Save enhanced image
                    output_file = output_path / item.name
                    enhanced_image.save(str(output_file))
                    print(f"✅ Saved: {output_file}")

                    # Update counters
                    update_counters(image_type)

                except Exception as e:
                    global total_errors
                    total_errors += 1
                    print(f"❌ Error processing file {item}: {e}")

        elif item.is_dir():
            # Recursively process subdirectories
            sub_output_dir = output_path / item.name
            process_directory(str(item), str(sub_output_dir))


# Start batch processing
print(f"🚀 Starting batch processing for directory: {input_base_dir}")
print(f"📁 Output directory: {output_base_dir}")
print("=" * 60)

# Initialize counters
brightfield_count = 0
stained_count = 0
unknown_count = 0
total_processed = 0
total_errors = 0


def update_counters(image_type):
    """Update processing counters"""
    global brightfield_count, stained_count, unknown_count, total_processed
    total_processed += 1
    if image_type == "Bright-field":
        brightfield_count += 1
    elif image_type == "Stained":
        stained_count += 1
    else:
        unknown_count += 1


process_directory(input_base_dir, output_base_dir)

print("=" * 60)
print("🎉 Batch processing completed!")
print(f"📊 Processing summary:")
print(f"  ✅ Total processed: {total_processed} images")
print(f"  🔬 Bright-field images: {brightfield_count}")
print(f"  🧪 Stained images: {stained_count}")
print(f"  ❓ Unknown type images: {unknown_count}")
print(f"  ❌ Errors: {total_errors}")
print("=" * 60)
