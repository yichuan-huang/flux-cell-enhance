# Flux Cell Enhancement

A Python toolkit for enhancing microscopy cell images using the Flux Kontext diffusion model. This project provides both batch processing capabilities and interactive Jupyter notebook demonstrations for improving cellular microscopy image quality.

## Features

- **Intelligent Enhancement**: Uses Flux Kontext model for selective enhancement of cellular structures
- **Batch Processing**: Process entire directories of microscopy images automatically
- **Interactive Demo**: Jupyter notebook for experimenting with different parameters
- **Noise Control**: Advanced prompting for noise reduction while preserving cellular details
- **Format Support**: Supports multiple image formats (JPG, PNG, TIFF, BMP)

## Requirements

- Python 3.8+
- CUDA-compatible GPU with sufficient VRAM
- PyTorch with CUDA support

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/flux-cell-enhance.git
cd flux-cell-enhance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Batch Processing

Use `flux_batch_enhance.py` for processing multiple images:

```python
python flux_batch_enhance.py
```

The script will:
- Load the Flux Kontext model
- Process all images in the specified input directory
- Save enhanced images to the output directory
- Maintain the original directory structure

### Interactive Demo

Open the Jupyter notebook for interactive experimentation:

```bash
jupyter notebook flux_enhance_demo.ipynb
```

The notebook allows you to:
- Select random images from your dataset
- Experiment with different enhancement parameters
- Compare original and enhanced images side-by-side
- Visualize results with matplotlib

## Configuration

### Input/Output Directories

Modify the paths in `flux_batch_enhance.py`:

```python
input_base_dir = "../cell_datasets/confocal_new"
output_base_dir = "../cell_datasets_Flux/confocal_new"
```

### Enhancement Parameters

Adjust the enhancement settings:

```python
guidance_scale = 2.5          # Controls enhancement strength
num_inference_steps = 28      # Quality vs speed trade-off
```

### Custom Prompts

Modify the enhancement prompts for different results:

```python
prompt = "your custom enhancement prompt"
negative_prompt = "aspects to avoid during enhancement"
```

## Model Information

This project uses the **FLUX.1-Kontext-dev** model from Black Forest Labs:
- Model: `black-forest-labs/FLUX.1-Kontext-dev`
- Type: Diffusion model for image-to-image enhancement
- Precision: bfloat16 for optimal VRAM usage

## File Structure

```
flux-cell-enhance/
├── flux_batch_enhance.py      # Batch processing script
├── flux_enhance_demo.ipynb    # Interactive demo notebook
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # This file
└── LICENSE                   # License information
```

## Tips for Best Results

1. **Image Quality**: Use high-quality microscopy images for best enhancement results
2. **GPU Memory**: Ensure sufficient VRAM (8GB+ recommended)
3. **Batch Size**: Process images individually to avoid memory issues
4. **Parameter Tuning**: Experiment with different guidance scales for your specific use case

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Black Forest Labs for the FLUX.1-Kontext model
- Hugging Face Diffusers library
- The scientific microscopy community

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{flux_cell_enhance,
  title={Flux Cell Enhancement: Microscopy Image Enhancement using Diffusion Models},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/flux-cell-enhance}
}
```
