# Flux Cell Enhancement

A Python toolkit for enhancing microscopy cell images using the Flux Kontext diffusion model. This project provides both batch processing capabilities and interactive Jupyter notebook demonstrations for improving cellular microscopy image quality.

## Features

- **Intelligent Enhancement**: Uses Flux Kontext model for selective enhancement of cellular structures
- **Adaptive Image Processing**: Automatically detects and applies specialized enhancement for bright-field (A folders) vs stained (B folders) microscopy images
- **Batch Processing**: Process entire directories of microscopy images automatically with progress tracking
- **Interactive Demo**: Jupyter notebook for experimenting with different parameters and image types
- **Optimized Parameters**: Specialized prompts and settings for different microscopy image types
- **Noise Control**: Advanced prompting for noise reduction while preserving cellular details
- **Error Handling**: Robust error handling with detailed processing statistics

## Requirements

- Python 3.8+
- CUDA-compatible GPU with sufficient VRAM (48GB+ recommended)
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

## Dataset Structure
```
confocal_new/
├── trainA/
├── trainB/
├── testA/
├── testB/      
```
**Note**: A and B can also have suffixes such as trainB1 and trainB_actin

### Batch Processing

Use `flux_batch_enhance.py` for processing multiple images:

```python
python flux_batch_enhance.py
```

The script will:
- Load the Flux Kontext model
- Automatically detect image types (bright-field vs stained) based on folder naming patterns
- Apply specialized enhancement parameters for each image type:
  - **Bright-field images** (A folders): Enhanced edge clarity and cell-background contrast
  - **Stained images** (B folders): Gentle enhancement preserving cellular structures and staining patterns
- Process all images in the specified input directory recursively
- Save enhanced images to the output directory maintaining original structure
- Provide detailed processing statistics and error reporting

### Interactive Demo

Open the Jupyter notebook for interactive experimentation:

```bash
jupyter notebook flux_enhance_demo.ipynb
```

The notebook allows you to:
- Select random images from your dataset (both bright-field and stained types)
- Automatically apply optimized enhancement parameters based on image type
- Experiment with different enhancement parameters
- Compare original and enhanced images side-by-side with type identification
- Visualize results with matplotlib and processing statistics

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

1. **Image Organization**: Organize images using the naming convention:
   - `trainA`, `testA` folders for bright-field microscopy images
   - `trainB`, `testB` folders for stained microscopy images
   - The system automatically detects and applies appropriate enhancement
2. **Image Quality**: Use high-quality microscopy images for best enhancement results
3. **GPU Memory**: Ensure sufficient VRAM (48GB+ recommended)
4. **Batch Processing**: Images are processed individually to avoid memory issues
5. **Parameter Tuning**: The system auto-optimizes parameters, but manual adjustment is possible
6. **Processing Statistics**: Monitor the detailed statistics output for processing insights

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
  title={Flux Cell Enhancement: Adaptive Microscopy Image Enhancement using Diffusion Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yichuan-huang/flux-cell-enhance},
  note={Intelligent enhancement system with automatic image type detection and specialized processing for bright-field and stained microscopy images}
}
```