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


# Define specialized prompts for different image types
# For bright-field microscopy images (A folders) - enhanced for better edge clarity
brightfield_prompt = "enhance bright-field microscopy, high cell-background contrast, sharp crisp cell boundaries, clear defined edges, enhanced cell outline"
brightfield_negative = "noise amplification, grain, speckle, background texture, blurry edges, soft boundaries"

# For stained cell images (B folders) - gentle enhancement with preserved cellular structures
stained_prompt = "enhance stained microscopy, preserve cellular structures, improve clarity, maintain natural staining patterns, enhance cell visibility"
stained_negative = "distortion, color shift, over-enhancement, artificial appearance, noise amplification, artifacts, excessive contrast, unnatural colors"

# Default prompts for unknown folder types
default_prompt = "enhance microscopy image, improve clarity, noise reduction"
default_negative = "noise amplification, artifacts, over-processing"

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
                        guidance_scale = 4.0
                        num_steps = 55
                        image_type = "Bright-field"
                    elif is_stained_folder(folder_name):  # B folders - Stained images
                        current_prompt = stained_prompt
                        current_negative = stained_negative
                        guidance_scale = 3.0  # Reduced to prevent distortion
                        num_steps = 45  # Fewer steps for gentler processing
                        image_type = "Stained"
                    else:
                        # Default fallback for unknown folder types
                        current_prompt = default_prompt
                        current_negative = default_negative
                        guidance_scale = 3.5
                        num_steps = 50
                        image_type = "Unknown"

                    print(f"Processing {image_type}: {item}")
                    print(
                        f"  Parameters: guidance_scale={guidance_scale}, steps={num_steps}"
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
