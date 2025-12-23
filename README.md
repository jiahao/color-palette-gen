# Color Palette Generator

Generate WCAG AAA-compliant color palettes with maximal perceptual distance using CIEDE2000.

## Features

- **WCAG AAA Compliant**: All colors maintain 7:1 contrast ratio on white background
- **Maximally Distant Colors**: Uses CIEDE2000 delta-E metric for perceptual spacing
- **Black & White Optimization**: Maximizes distance from both black and white
- **HTML Color Names**: Maps generated colors to nearest standard CSS color names
- **Visual Comparison**: Generates distance matrix heatmap and color swatches

## Requirements

- Python 3.14+
- numpy
- scipy
- matplotlib

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install numpy scipy matplotlib
```

## Usage

```bash
python color_palette_generator.py
```

Generates 6 WCAG AAA-compliant colors and saves visualization to `/tmp/color_palette.png`

## Output

The script outputs:
- Color hex codes and RGB values
- Contrast ratios against white background
- CIEDE2000 distance from black
- Pairwise distance matrix with heatmap visualization
- Color patches with HTML/CSS color name mappings
