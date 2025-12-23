#!/usr/bin/env python3
"""
Generate WCAG-compliant color palettes with maximal CIEDE2000 distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import differential_evolution
import itertools


def rgb_to_xyz(rgb):
    """Convert RGB (0-255) to XYZ color space."""
    rgb_normalized = rgb / 255.0
    
    def adjust(channel):
        if channel <= 0.04045:
            return channel / 12.92
        else:
            return ((channel + 0.055) / 1.055) ** 2.4
    
    r, g, b = [adjust(c) for c in rgb_normalized]
    
    # RGB to XYZ conversion matrix (sRGB D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    return np.array([x * 100, y * 100, z * 100])


def xyz_to_lab(xyz):
    """Convert XYZ to LAB color space."""
    # D65 reference white
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    
    x = xyz[0] / ref_x
    y = xyz[1] / ref_y
    z = xyz[2] / ref_z
    
    def f(t):
        delta = 6/29
        if t > delta**3:
            return t**(1/3)
        else:
            return t / (3 * delta**2) + 4/29
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.array([L, a, b])


def rgb_to_lab(rgb):
    """Convert RGB (0-255) to LAB color space."""
    xyz = rgb_to_xyz(rgb)
    return xyz_to_lab(xyz)


def delta_e_cie2000(lab1, lab2):
    """Calculate CIEDE2000 color difference."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Calculate C and h
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    
    C_bar = (C1 + C2) / 2
    
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.arctan2(b1, a1_prime) % (2 * np.pi)
    h2_prime = np.arctan2(b2, a2_prime) % (2 * np.pi)
    
    # Calculate differences
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        delta_h = h2_prime - h1_prime
        if abs(delta_h) <= np.pi:
            delta_h_prime = delta_h
        elif delta_h > np.pi:
            delta_h_prime = delta_h - 2 * np.pi
        else:
            delta_h_prime = delta_h + 2 * np.pi
    
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(delta_h_prime / 2)
    
    # Calculate mean values
    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    
    if C1_prime * C2_prime == 0:
        h_bar_prime = h1_prime + h2_prime
    else:
        h_sum = h1_prime + h2_prime
        h_diff = abs(h1_prime - h2_prime)
        if h_diff <= np.pi:
            h_bar_prime = h_sum / 2
        elif h_sum < 2 * np.pi:
            h_bar_prime = (h_sum + 2 * np.pi) / 2
        else:
            h_bar_prime = (h_sum - 2 * np.pi) / 2
    
    T = (1 - 0.17 * np.cos(h_bar_prime - np.pi/6) +
         0.24 * np.cos(2 * h_bar_prime) +
         0.32 * np.cos(3 * h_bar_prime + np.pi/30) -
         0.20 * np.cos(4 * h_bar_prime - 63*np.pi/180))
    
    delta_theta = (30 * np.pi / 180) * np.exp(-((h_bar_prime - 275*np.pi/180) / (25*np.pi/180))**2)
    
    R_C = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    
    S_L = 1 + (0.015 * (L_bar_prime - 50)**2) / np.sqrt(20 + (L_bar_prime - 50)**2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    
    R_T = -np.sin(2 * delta_theta) * R_C
    
    # Calculate final delta E
    delta_E = np.sqrt(
        (delta_L_prime / S_L)**2 +
        (delta_C_prime / S_C)**2 +
        (delta_H_prime / S_H)**2 +
        R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )
    
    return delta_E


def calculate_luminance(rgb):
    """Calculate relative luminance for contrast ratio."""
    rgb_normalized = rgb / 255.0
    
    def adjust(channel):
        if channel <= 0.03928:
            return channel / 12.92
        else:
            return ((channel + 0.055) / 1.055) ** 2.4
    
    r, g, b = [adjust(c) for c in rgb_normalized]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(rgb1, rgb2):
    """Calculate contrast ratio between two RGB colors."""
    l1 = calculate_luminance(rgb1)
    l2 = calculate_luminance(rgb2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def meets_wcag_aa_large(rgb, background=np.array([255, 255, 255])):
    """Check if color meets WCAG AAA (7:1) against background."""
    ratio = contrast_ratio(rgb, background)
    return ratio >= 7.0


def generate_candidate_colors(n_samples=1000):
    """Generate candidate colors that meet WCAG contrast requirements."""
    candidates = []
    white_bg = np.array([255, 255, 255])
    
    # Sample from RGB space and filter by contrast
    for _ in range(n_samples):
        rgb = np.random.randint(0, 256, 3)
        if meets_wcag_aa_large(rgb, white_bg):
            candidates.append(rgb)
    
    return np.array(candidates)


# HTML/CSS color names with their RGB values
HTML_COLORS = {
    'AliceBlue': (240, 248, 255), 'AntiqueWhite': (250, 235, 215), 'Aqua': (0, 255, 255),
    'Aquamarine': (127, 255, 212), 'Azure': (240, 255, 255), 'Beige': (245, 245, 220),
    'Bisque': (255, 228, 196), 'Black': (0, 0, 0), 'BlanchedAlmond': (255, 235, 205),
    'Blue': (0, 0, 255), 'BlueViolet': (138, 43, 226), 'Brown': (165, 42, 42),
    'BurlyWood': (222, 184, 135), 'CadetBlue': (95, 158, 160), 'Chartreuse': (127, 255, 0),
    'Chocolate': (210, 105, 30), 'Coral': (255, 127, 80), 'CornflowerBlue': (100, 149, 237),
    'Cornsilk': (255, 248, 220), 'Crimson': (220, 20, 60), 'Cyan': (0, 255, 255),
    'DarkBlue': (0, 0, 139), 'DarkCyan': (0, 139, 139), 'DarkGoldenRod': (184, 134, 11),
    'DarkGray': (169, 169, 169), 'DarkGreen': (0, 100, 0), 'DarkKhaki': (189, 183, 107),
    'DarkMagenta': (139, 0, 139), 'DarkOliveGreen': (85, 107, 47), 'DarkOrange': (255, 140, 0),
    'DarkOrchid': (153, 50, 204), 'DarkRed': (139, 0, 0), 'DarkSalmon': (233, 150, 122),
    'DarkSeaGreen': (143, 188, 143), 'DarkSlateBlue': (72, 61, 139), 'DarkSlateGray': (47, 79, 79),
    'DarkTurquoise': (0, 206, 209), 'DarkViolet': (148, 0, 211), 'DeepPink': (255, 20, 147),
    'DeepSkyBlue': (0, 191, 255), 'DimGray': (105, 105, 105), 'DodgerBlue': (30, 144, 255),
    'FireBrick': (178, 34, 34), 'FloralWhite': (255, 250, 240), 'ForestGreen': (34, 139, 34),
    'Fuchsia': (255, 0, 255), 'Gainsboro': (220, 220, 220), 'GhostWhite': (248, 248, 255),
    'Gold': (255, 215, 0), 'GoldenRod': (218, 165, 32), 'Gray': (128, 128, 128),
    'Green': (0, 128, 0), 'GreenYellow': (173, 255, 47), 'HoneyDew': (240, 255, 240),
    'HotPink': (255, 105, 180), 'IndianRed': (205, 92, 92), 'Indigo': (75, 0, 130),
    'Ivory': (255, 255, 240), 'Khaki': (240, 230, 140), 'Lavender': (230, 230, 250),
    'LavenderBlush': (255, 240, 245), 'LawnGreen': (124, 252, 0), 'LemonChiffon': (255, 250, 205),
    'LightBlue': (173, 216, 230), 'LightCoral': (240, 128, 128), 'LightCyan': (224, 255, 255),
    'LightGoldenRodYellow': (250, 250, 210), 'LightGray': (211, 211, 211), 'LightGreen': (144, 238, 144),
    'LightPink': (255, 182, 193), 'LightSalmon': (255, 160, 122), 'LightSeaGreen': (32, 178, 170),
    'LightSkyBlue': (135, 206, 250), 'LightSlateGray': (119, 136, 153), 'LightSteelBlue': (176, 196, 222),
    'LightYellow': (255, 255, 224), 'Lime': (0, 255, 0), 'LimeGreen': (50, 205, 50),
    'Linen': (250, 240, 230), 'Magenta': (255, 0, 255), 'Maroon': (128, 0, 0),
    'MediumAquaMarine': (102, 205, 170), 'MediumBlue': (0, 0, 205), 'MediumOrchid': (186, 85, 211),
    'MediumPurple': (147, 112, 219), 'MediumSeaGreen': (60, 179, 113), 'MediumSlateBlue': (123, 104, 238),
    'MediumSpringGreen': (0, 250, 154), 'MediumTurquoise': (72, 209, 204), 'MediumVioletRed': (199, 21, 133),
    'MidnightBlue': (25, 25, 112), 'MintCream': (245, 255, 250), 'MistyRose': (255, 228, 225),
    'Moccasin': (255, 228, 181), 'NavajoWhite': (255, 222, 173), 'Navy': (0, 0, 128),
    'OldLace': (253, 245, 230), 'Olive': (128, 128, 0), 'OliveDrab': (107, 142, 35),
    'Orange': (255, 165, 0), 'OrangeRed': (255, 69, 0), 'Orchid': (218, 112, 214),
    'PaleGoldenRod': (238, 232, 170), 'PaleGreen': (152, 251, 152), 'PaleTurquoise': (175, 238, 238),
    'PaleVioletRed': (219, 112, 147), 'PapayaWhip': (255, 239, 213), 'PeachPuff': (255, 218, 185),
    'Peru': (205, 133, 63), 'Pink': (255, 192, 203), 'Plum': (221, 160, 221),
    'PowderBlue': (176, 224, 230), 'Purple': (128, 0, 128), 'RebeccaPurple': (102, 51, 153),
    'Red': (255, 0, 0), 'RosyBrown': (188, 143, 143), 'RoyalBlue': (65, 105, 225),
    'SaddleBrown': (139, 69, 19), 'Salmon': (250, 128, 114), 'SandyBrown': (244, 164, 96),
    'SeaGreen': (46, 139, 87), 'SeaShell': (255, 245, 238), 'Sienna': (160, 82, 45),
    'Silver': (192, 192, 192), 'SkyBlue': (135, 206, 235), 'SlateBlue': (106, 90, 205),
    'SlateGray': (112, 128, 144), 'Snow': (255, 250, 250), 'SpringGreen': (0, 255, 127),
    'SteelBlue': (70, 130, 180), 'Tan': (210, 180, 140), 'Teal': (0, 128, 128),
    'Thistle': (216, 191, 216), 'Tomato': (255, 99, 71), 'Turquoise': (64, 224, 208),
    'Violet': (238, 130, 238), 'Wheat': (245, 222, 179), 'White': (255, 255, 255),
    'WhiteSmoke': (245, 245, 245), 'Yellow': (255, 255, 0), 'YellowGreen': (154, 205, 50)
}


def get_ncs_name(rgb):
    """Generate NCS (Natural Color System) color name."""
    r, g, b = rgb / 255.0

    # Convert to HSL
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    l = (max_val + min_val) / 2

    # For very dark colors, hue is unreliable
    if l < 0.15:
        if l < 0.05:
            return "NCS 9000 Black"
        else:
            return "NCS 8000 Very Dark Gray"

    if max_val == min_val:
        h_deg = 0
        s = 0
    else:
        d = max_val - min_val
        s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        if max_val == r:
            h_deg = ((g - b) / d + (6 if g < b else 0)) * 60
        elif max_val == g:
            h_deg = ((b - r) / d + 2) * 60
        else:
            h_deg = ((r - g) / d + 4) * 60

    if s < 0.05:
        if l < 0.5:
            return "NCS 5000 Dark Gray"
        else:
            return "NCS 2000 Light Gray"

    # NCS Hue
    if h_deg < 30 or h_deg >= 330:
        hue_desc = "Red"
    elif h_deg < 60:
        hue_desc = "Yellow-Red"
    elif h_deg < 90:
        hue_desc = "Yellow"
    elif h_deg < 120:
        hue_desc = "Green-Yellow"
    elif h_deg < 150:
        hue_desc = "Green"
    elif h_deg < 180:
        hue_desc = "Cyan-Green"
    elif h_deg < 210:
        hue_desc = "Cyan"
    elif h_deg < 240:
        hue_desc = "Blue-Cyan"
    elif h_deg < 270:
        hue_desc = "Blue"
    elif h_deg < 300:
        hue_desc = "Red-Blue"
    else:
        hue_desc = "Red"

    whiteness = int((1 - max_val) * 100)
    blackness = int((1 - l) * 100)
    return f"NCS S {whiteness:02d}{blackness:02d} {hue_desc}"


def get_pantone_name(rgb):
    """Pantone mapping removed — return placeholder."""
    return "Pantone N/A"


def get_color_name(rgb):
    """Find the nearest HTML/CSS color name using CIEDE2000 distance."""
    target_lab = rgb_to_lab(rgb)
    
    min_distance = float('inf')
    nearest_name = None
    
    for name, html_rgb in HTML_COLORS.items():
        html_rgb_array = np.array(html_rgb)
        html_lab = rgb_to_lab(html_rgb_array)
        distance = delta_e_cie2000(target_lab, html_lab)
        
        if distance < min_distance:
            min_distance = distance
            nearest_name = name
    
    return nearest_name


def calculate_min_pairwise_distance(colors):
    """Calculate minimum pairwise CIEDE2000 distance."""
    if len(colors) < 2:
        return float('inf')
    
    lab_colors = [rgb_to_lab(color) for color in colors]
    min_dist = float('inf')
    
    for i, j in itertools.combinations(range(len(colors)), 2):
        dist = delta_e_cie2000(lab_colors[i], lab_colors[j])
        min_dist = min(min_dist, dist)
    
    return min_dist


def objective_function(flat_colors, n_colors):
    """Objective to maximize: minimum pairwise distance (negated for minimization)."""
    colors = flat_colors.reshape(n_colors, 3)
    
    # Check WCAG compliance and apply a smooth penalty instead of hard cutoff.
    white_bg = np.array([255, 255, 255])
    deficit_sum = 0.0
    for color in colors:
        ratio = contrast_ratio(color, white_bg)
        if ratio < 7.0:
            deficit_sum += (7.0 - ratio)
    if deficit_sum > 0:
        # Large but finite penalty that scales with total contrast deficit
        # (keeps optimizer exploring instead of immediate rejection).
        return 1e5 + 1e4 * deficit_sum
    
    # Calculate minimum pairwise distance between colors
    min_pairwise_dist = calculate_min_pairwise_distance(colors)
    
    # Calculate minimum distance from black
    black = np.array([0, 0, 0])
    black_lab = rgb_to_lab(black)
    min_black_dist = float('inf')
    for color in colors:
        color_lab = rgb_to_lab(color)
        dist = delta_e_cie2000(color_lab, black_lab)
        min_black_dist = min(min_black_dist, dist)
    
    # Combine objectives: prioritize pairwise distance, but also consider black distance
    # Weight pairwise distance more heavily
    combined_score = 0.8 * min_pairwise_dist + 0.2 * min_black_dist
    
    return -combined_score  # Negative because we minimize


def generate_optimal_palette(n_colors=6, n_iterations=100):
    """Generate color palette with maximal CIEDE2000 distance."""
    print(f"Generating {n_colors} WCAG-compliant colors with maximal CIEDE2000 distance...")
    print("This may take a minute...\n")
    
    # Initial guess: sample candidate colors
    candidates = generate_candidate_colors(n_samples=5000)
    
    if len(candidates) < n_colors:
        raise ValueError("Not enough WCAG-compliant colors found!")
    
    # Use diverse starting points
    indices = np.linspace(0, len(candidates) - 1, n_colors, dtype=int)
    initial_colors = candidates[indices]
    
    # Define bounds for RGB values
    bounds = [(0, 255)] * (n_colors * 3)
    
    # Per-generation callback to log progress
    iteration_count = [0]
    def callback(xk, convergence):
        iteration_count[0] += 1
        obj_value = objective_function(xk, n_colors)
        min_dist = -obj_value if obj_value < 1e5 else 0.0
        print(f"Iteration {iteration_count[0]:3d}: Min Distance = {min_dist:6.2f}, Convergence = {convergence:.6f}")
        return False

    # Seed the initial population with WCAG-compliant candidates to avoid
    # large early penalties. Build x0 of shape (npop, ndim).
    ndim = n_colors * 3
    popsize = 7
    npop = popsize * ndim

    # Ensure there are enough candidate samples
    candidates = generate_candidate_colors(n_samples=max(5000, npop * 2))
    if len(candidates) < n_colors:
        raise ValueError("Not enough WCAG-compliant colors found to seed population!")

    x0 = []
    for i in range(npop):
        idx = np.random.choice(len(candidates), size=n_colors, replace=True)
        individual = candidates[idx].flatten()
        jitter = np.random.normal(scale=1.0, size=individual.shape)
        x0.append(np.clip(individual + jitter, 0, 255))
    x0 = np.array(x0)

    # Run optimization
    result = differential_evolution(
        lambda x: objective_function(x, n_colors),
        bounds,
        maxiter=n_iterations,
        popsize=popsize,
        seed=42,
        workers=1,
        init=x0,
        atol=0.01,
        tol=0.01,
        callback=callback,
        updating='immediate'
    )
    
    # Extract optimized colors
    colors = result.x.reshape(n_colors, 3).astype(int)
    colors = np.clip(colors, 0, 255)
    
    return colors


def visualize_palette(colors):
    """Visualize the color palette with information."""
    n_colors = len(colors)
    white_bg = np.array([255, 255, 255])
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), 
                            gridspec_kw={'height_ratios': [2, 1]})
    
    # Top plot: Color swatches
    ax1 = axes[0]
    ax1.set_xlim(0, n_colors)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    
    for i, color in enumerate(colors):
        rect = Rectangle((i, 0), 1, 1, facecolor=color/255.0, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        
        # Add text information
        rgb_text = f"RGB({color[0]}, {color[1]}, {color[2]})"
        hex_text = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
        contrast = contrast_ratio(color, white_bg)
        color_name = get_color_name(color)
        
        # Use white or black text depending on color darkness
        text_color = 'white' if calculate_luminance(color) < 0.5 else 'black'
        
        ax1.text(i + 0.5, 0.85, color_name, ha='center', va='center', 
                fontsize=8, fontweight='bold', color=text_color, style='italic')
        ax1.text(i + 0.5, 0.65, hex_text, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=text_color, family='monospace')
        ax1.text(i + 0.5, 0.45, rgb_text, ha='center', va='center', 
                fontsize=8, color=text_color, family='monospace')
        ax1.text(i + 0.5, 0.25, f"CR: {contrast:.2f}", ha='center', va='center', 
                fontsize=9, color=text_color)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("WCAG AAA Compliant Color Palette (7:1 on White)", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Bottom plot: Distance matrix including black and white
    ax2 = axes[1]
    
    # Add black and white to the color list
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    extended_colors = np.vstack([colors, [black], [white]])
    n_extended = len(extended_colors)
    
    lab_colors_extended = [rgb_to_lab(color) for color in extended_colors]
    
    distance_matrix = np.zeros((n_extended, n_extended))
    for i in range(n_extended):
        for j in range(n_extended):
            if i != j:
                distance_matrix[i, j] = delta_e_cie2000(lab_colors_extended[i], lab_colors_extended[j])
    
    im = ax2.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')
    
    # Create color patches for axis labels instead of text
    ax2.set_xticks(range(n_extended))
    ax2.set_yticks(range(n_extended))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.tick_params(length=0)
    
    # Draw color patches on the axes
    patch_size = 0.4
    for i in range(n_extended):
        # X-axis patches (top)
        color_rgb = extended_colors[i] / 255.0
        x_rect = Rectangle((i - patch_size/2, n_extended - 0.5 + 0.1), 
                           patch_size, 0.3, 
                           facecolor=color_rgb, edgecolor='black', linewidth=1,
                           clip_on=False, transform=ax2.transData)
        ax2.add_patch(x_rect)
        
        # Y-axis patches (left)
        y_rect = Rectangle((-0.5 - 0.4, i - patch_size/2), 
                           0.3, patch_size,
                           facecolor=color_rgb, edgecolor='black', linewidth=1,
                           clip_on=False, transform=ax2.transData)
        ax2.add_patch(y_rect)
    
    ax2.set_title("CIEDE2000 Distance Matrix (including Black & White)", 
                 fontsize=12, fontweight='bold', pad=25)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('ΔE 2000', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(n_extended):
        for j in range(n_extended):
            if i != j:
                text = ax2.text(j, i, f'{distance_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    # Print statistics
    print("\n" + "="*70)
    print("COLOR PALETTE STATISTICS")
    print("="*70)
    
    black = np.array([0, 0, 0])
    black_lab = rgb_to_lab(black)
    
    for i, color in enumerate(colors, 1):
        hex_code = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
        rgb = f"RGB({color[0]:3d}, {color[1]:3d}, {color[2]:3d})"
        contrast = contrast_ratio(color, white_bg)
        color_name = get_color_name(color)
        
        # Calculate distance from black
        color_lab = rgb_to_lab(color)
        black_dist = delta_e_cie2000(color_lab, black_lab)
        
        print(f"Color {i}: {color_name}")
        print(f"  {hex_code} | {rgb}")
        print(f"  Contrast (white): {contrast:.2f}:1 | ΔE from black: {black_dist:.2f}")
    
    print("\n" + "-"*60)
    print("PAIRWISE CIEDE2000 DISTANCES:")
    print("-"*60)
    
    distances = []
    for i in range(n_colors):
        for j in range(i+1, n_colors):
            dist = distance_matrix[i, j]
            distances.append(dist)
            print(f"  C{i+1} ↔ C{j+1}: {dist:.2f}")
    
    # Calculate distances from black
    black_distances = []
    for i, color in enumerate(colors, 1):
        color_lab = rgb_to_lab(color)
        black_dist = delta_e_cie2000(color_lab, black_lab)
        black_distances.append(black_dist)
    
    print("\n" + "-"*70)
    print("SUMMARY STATISTICS:")
    print("-"*70)
    print("Pairwise distances:")
    print(f"  Minimum: {min(distances):.2f}")
    print(f"  Maximum: {max(distances):.2f}")
    print(f"  Average: {np.mean(distances):.2f}")
    print("\nDistances from black (#000000):")
    print(f"  Minimum: {min(black_distances):.2f}")
    print(f"  Maximum: {max(black_distances):.2f}")
    print(f"  Average: {np.mean(black_distances):.2f}")
    print("="*70)
    
    plt.savefig('/tmp/color_palette.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /tmp/color_palette.png")
    plt.show()


def main():
    """Main function."""
    N_COLORS = 6
    
    # Generate optimal palette
    colors = generate_optimal_palette(n_colors=N_COLORS, n_iterations=100)
    
    # Visualize
    visualize_palette(colors)


if __name__ == "__main__":
    main()
# Test comment
