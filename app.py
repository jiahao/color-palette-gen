#!/usr/bin/env python3
"""
Streamlit web app for interactive WCAG-compliant color palette generation.
"""

import streamlit as st
import numpy as np
from color_palette_generator import (
    rgb_to_lab, delta_e_cie2000, calculate_luminance, contrast_ratio,
    meets_wcag_aa_large, get_color_name, get_ncs_name, get_pantone_name,
    get_all_names,
    generate_optimal_palette, HTML_COLORS
)

st.set_page_config(page_title="Color Palette Generator", layout="wide")

st.title("🎨 WCAG AAA Color Palette Generator")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Configuration")

    color_col1, color_col2 = st.columns(2)
    with color_col1:
        background_color_hex = st.color_picker(
            "Background Color:",
            value="#FFFFFF",
            help="Default background color for contrast calculations"
        )
    with color_col2:
        foreground_color_hex = st.color_picker(
            "Foreground Color:",
            value="#000000",
            help="Default foreground color (e.g., for text)"
        )

    existing_colors_input = st.text_area(
        "Additional colors (hex codes, comma-separated):",
        value="",
        height=100,
        help="Include additional existing colors"
    )

    n_colors = st.slider(
        "Number of colors to generate:",
        min_value=2,
        max_value=32,
        value=6,
        help="Total includes existing, foreground, background, and generated colors"
    )

with col2:
    st.subheader("Info")
    st.info("""
    This tool generates color palettes that:
    - Meet WCAG AAA (7:1 contrast) on specified background
    - Maximize perceptual distance (CIEDE2000)
    - Distance from foreground and background colors
    - Include ISCC-NBS, NCS, and Pantone color names
    """)

# Parse background and foreground colors
background_color = np.array([
    int(background_color_hex[1:3], 16),
    int(background_color_hex[3:5], 16),
    int(background_color_hex[5:7], 16)
])

foreground_color = np.array([
    int(foreground_color_hex[1:3], 16),
    int(foreground_color_hex[3:5], 16),
    int(foreground_color_hex[5:7], 16)
])

# Parse additional existing colors (not including bg/fg as they're just for reference)
additional_colors = []
if existing_colors_input.strip():
    try:
        hex_codes = [h.strip() for h in existing_colors_input.split(",") if h.strip()]
        for hex_code in hex_codes:
            hex_code = hex_code.lstrip("#")
            r, g, b = int(hex_code[0:2], 16), int(hex_code[2:4], 16), int(hex_code[4:6], 16)
            new_color = np.array([r, g, b])
            # Avoid duplicates
            if not any(np.array_equal(new_color, existing) for existing in additional_colors):
                additional_colors.append(new_color)
    except ValueError:
        st.error("Invalid hex code format. Use #RRGGBB or RRGGBB")

# Generate palette automatically
if n_colors > 0:
    with st.spinner("Generating color palette..."):
        n_to_generate = max(0, n_colors - len(additional_colors))

        if n_to_generate > 0:
            generated_colors = generate_optimal_palette(n_colors=n_to_generate, n_iterations=100)
        else:
            generated_colors = np.array([])

        if len(generated_colors) > 0:
            palette_colors = np.vstack([additional_colors, generated_colors]) if len(additional_colors) > 0 else generated_colors
        else:
            palette_colors = np.array(additional_colors) if len(additional_colors) > 0 else np.array([])

        if len(palette_colors) == 0:
            st.error("No colors to display")
        else:

            # Display color swatches as native HTML
            st.subheader("Generated Color Palette")

            # Create color swatches using Streamlit columns
            cols = st.columns(len(palette_colors))

            for col_idx, (col, color) in enumerate(zip(cols, palette_colors)):
                with col:
                    hex_code = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
                    contrast = contrast_ratio(color, background_color)
                    names = get_all_names(color)
                    iscc_name = names.get('ISCC-NBS')
                    ncs_name = names.get('NCS')
                    pantone_name = names.get('Pantone')
                    ral_name = names.get('RAL Classic')

                    text_color = 'white' if calculate_luminance(color) < 0.5 else '#333333'

                    # Create color swatch HTML
                    rgb_hex = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
                    swatch_html = f"""
                    <div style="
                        background-color: {rgb_hex};
                        border: 2px solid #333;
                        border-radius: 8px;
                        padding: 12px;
                        text-align: center;
                        color: {text_color};
                        font-family: monospace;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                        min-height: 280px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    ">
                        <div>
                            <div style="font-size: 11px; font-weight: bold; margin-bottom: 4px; font-style: italic; font-family: sans-serif;">{iscc_name}</div>
                            <div style="font-size: 9px; margin-bottom: 2px; font-style: italic; font-family: sans-serif;">{ncs_name}</div>
                            <div style="font-size: 9px; margin-bottom: 8px; font-style: italic; font-family: sans-serif;">{pantone_name.replace('Pantone ', '')}</div>
                            <div style="font-size: 9px; margin-bottom: 8px; font-style: italic; font-family: sans-serif;">{ral_name}</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; font-weight: bold; margin-bottom: 4px;">{hex_code}</div>
                            <div style="font-size: 9px; margin-bottom: 4px;">RGB({color[0]}, {color[1]}, {color[2]})</div>
                            <div style="font-size: 10px; font-weight: bold;">CR: {contrast:.2f}:1</div>
                        </div>
                    </div>
                    """
                    st.markdown(swatch_html, unsafe_allow_html=True)

            # Color statistics table
            st.subheader("Color Details")

            color_data = []
            for i, color in enumerate(palette_colors, 1):
                hex_code = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
                contrast = contrast_ratio(color, background_color)
                color_lab = rgb_to_lab(color)
                fg_lab = rgb_to_lab(foreground_color)
                fg_dist = delta_e_cie2000(color_lab, fg_lab)
                names = get_all_names(color)
                iscc_name = names.get('ISCC-NBS')
                ncs_name = names.get('NCS')
                pantone_name = names.get('Pantone')
                ral_name = names.get('RAL Classic')

                color_data.append({
                    "Hex": hex_code,
                    "ISCC-NBS": iscc_name,
                    "NCS": ncs_name,
                    "Pantone": pantone_name,
                    "RAL": ral_name,
                    "Contrast (vs BG)": f"{contrast:.2f}:1",
                    "ΔE from FG": f"{fg_dist:.2f}"
                })

            st.dataframe(color_data, width="stretch")

            # Distance matrix
            st.subheader("CIEDE2000 Distance Matrix")

            # Include background and foreground colors in distance matrix
            all_colors_with_refs = [background_color, foreground_color]
            for color in palette_colors:
                # Avoid duplicates
                if not any(np.array_equal(color, ref) for ref in all_colors_with_refs):
                    all_colors_with_refs.append(color)
            
            extended_colors = np.array(all_colors_with_refs)
            n_extended = len(extended_colors)

            lab_colors_extended = [rgb_to_lab(color) for color in extended_colors]

            distance_matrix = np.zeros((n_extended, n_extended))
            for i in range(n_extended):
                for j in range(n_extended):
                    if i != j:
                        distance_matrix[i, j] = delta_e_cie2000(lab_colors_extended[i], lab_colors_extended[j])

            # Render distance matrix as native HTML table
            max_distance = np.max(distance_matrix)
            
            # Build HTML table rows as a list
            html_parts = []
            html_parts.append('''
<style>
.distance-matrix {
    border-collapse: collapse;
    margin: 20px auto;
    font-family: monospace;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.distance-matrix td, .distance-matrix th {
    border: 1px solid #ddd;
    text-align: center;
    min-width: 60px;
    height: 60px;
    padding: 5px;
}
.distance-matrix .color-cell {
    width: 50px;
    height: 50px;
    border: 2px solid #333;
    margin: 0 auto;
}
.distance-matrix .value-cell {
    font-size: 11px;
    font-weight: 600;
}
.distance-matrix .diagonal {
    background-color: #f0f0f0;
    color: #999;
}
</style>
<table class="distance-matrix">
<thead>
<tr>
<th></th>
''')
            
            # Header row with color patches
            for i in range(n_extended):
                color = extended_colors[i]
                hex_code = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
                html_parts.append(f'<th><div class="color-cell" style="background-color: {hex_code};"></div></th>')
            
            html_parts.append('</tr>\n</thead>\n<tbody>\n')
            
            # Data rows
            for i in range(n_extended):
                color_i = extended_colors[i]
                hex_i = f"#{color_i[0]:02X}{color_i[1]:02X}{color_i[2]:02X}"
                html_parts.append(f'<tr><th><div class="color-cell" style="background-color: {hex_i};"></div></th>')
                
                for j in range(n_extended):
                    distance = distance_matrix[i, j]
                    
                    if i == j:
                        html_parts.append('<td class="value-cell diagonal">&mdash;</td>')
                    else:
                        # Color intensity based on distance (yellow to red gradient)
                        intensity = distance / max_distance if max_distance > 0 else 0
                        # YlOrRd colormap approximation
                        r = int(255)
                        g = int(255 * (1 - intensity * 0.7))
                        b = int(100 * (1 - intensity))
                        bg_color = f"rgb({r}, {g}, {b})"
                        
                        html_parts.append(f'<td class="value-cell" style="background-color: {bg_color};">{distance:.1f}</td>')
                
                html_parts.append('</tr>\n')
            
            html_parts.append('</tbody>\n</table>')
            
            st.markdown(''.join(html_parts), unsafe_allow_html=True)

            # Distance statistics
            st.subheader("Distance Statistics")

            col1, col2, col3 = st.columns(3)

            distances = []
            # Calculate distances only among palette colors (skip bg/fg which are first 2)
            palette_start = 2  # Skip background and foreground
            for i in range(palette_start, n_extended):
                for j in range(i+1, n_extended):
                    distances.append(distance_matrix[i, j])

            if distances:
                with col1:
                    st.metric("Min Pairwise Distance", f"{min(distances):.2f}")
                with col2:
                    st.metric("Max Pairwise Distance", f"{max(distances):.2f}")
                with col3:
                    st.metric("Avg Pairwise Distance", f"{np.mean(distances):.2f}")

if __name__ == "__main__":
    pass
