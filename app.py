#!/usr/bin/env python3
"""
Streamlit web app for interactive WCAG-compliant color palette generation.
"""

import streamlit as st
import numpy as np
import threading, os, time
from color_palette_generator import (
    rgb_to_lab, delta_e_cie2000, calculate_luminance, contrast_ratio,
    meets_wcag_aa_large, get_color_name, get_ncs_name, get_pantone_name,
    get_all_names,
    generate_optimal_palette, HTML_COLORS, visualize_palette
)

# Detect availability of streamlit-autorefresh at import time to avoid
# showing a misleading UI warning if the package is available to the
# running interpreter but an import inside the UI rerun fails for other
# reasons. We set a simple flag and reuse the imported function if present.
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    _st_autorefresh = None
    AUTOREFRESH_AVAILABLE = False

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
        "Number of colors to generate (generated colors only):",
        min_value=2,
        max_value=32,
        value=6,
        help="Number of colors the optimizer should generate (does not include any existing colors you list)"
    )

    # Right column intentionally left blank (no Info panel)

    # Interactive controls moved to left-aligned columns for better useable space
    left_col, right_col = st.columns([3, 1])
    with left_col:
        # setup session state for background optimization
        if 'opt_thread' not in st.session_state:
            st.session_state['opt_thread'] = None
        if 'opt_stop_event' not in st.session_state:
            st.session_state['opt_stop_event'] = None

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

        # `n_colors` is the number of colors to generate (ignore additional existing colors)
        n_to_generate = int(n_colors)

        # Optionally include existing colors in the native display
        include_additional_in_native = st.checkbox('Include existing colors in native view', value=False,
                              help='If checked, existing colors you listed will be prepended to the native palette display')

        # Optimizer selection and refine option
        optimizer = st.selectbox(
            'Optimizer',
            options=['de', 'cma', 'basinhopping', 'greedy', 'local'],
            index=0,
            help='Select the optimizer to use for palette generation'
        )

        refine = st.checkbox('Refine with CMA-ES (when applicable)', value=False,
                             help='Run a CMA-ES refinement pass after the main optimizer')

        # Place optimization iterations and interactive controls stacked vertically
        iters = st.number_input('Optimization iterations', min_value=1, max_value=2000, value=100)

        # Buttons on a single row for compactness
        btn_cols = st.columns([1, 1, 1])
        with btn_cols[0]:
            start = st.button('Start optimization')
        with btn_cols[1]:
            refresh = st.button('Refresh view')
        with btn_cols[2]:
            stop_btn = st.button('Stop optimization')

        st.write('To see interim palettes, click Refresh or Stop (updates view).')
        status = st.empty()
        img_holder = st.empty()

        def _ui_callback(palette, meta=None):
            # Write interim palette to disk so the main Streamlit thread can show it
            try:
                np.save('/tmp/current_palette.npy', np.array(palette).astype(int))
            except Exception:
                try:
                    with open('/tmp/current_palette_error.txt', 'w') as f:
                        f.write('Failed to save interim palette')
                except Exception:
                    pass
            # Save visualization image (best-effort)
            try:
                visualize_palette(np.array(palette).astype(int))
            except Exception:
                pass

        def _run_background():
            # Background runner for optimization
            try:
                stop_evt = st.session_state.get('opt_stop_event')
                res = generate_optimal_palette(n_colors=n_to_generate, n_iterations=iters, optimizer=optimizer, refine=refine, callback=_ui_callback, stop_event=stop_evt)
                np.save('/tmp/current_palette_final.npy', np.array(res).astype(int))
                try:
                    visualize_palette(np.array(res).astype(int))
                except Exception:
                    pass
            except Exception as e:
                try:
                    with open('/tmp/current_palette_error.txt', 'w') as f:
                        f.write(str(e))
                except Exception:
                    pass

        if start:
            # create/clear stop event
            st.session_state['opt_stop_event'] = threading.Event()
            # start background thread
            t = threading.Thread(target=_run_background, daemon=True)
            st.session_state['opt_thread'] = t
            t.start()
            status.info('Optimization started in background')

        if stop_btn:
            if st.session_state.get('opt_stop_event') is not None:
                st.session_state['opt_stop_event'].set()
                status.warning('Stop requested — optimization will stop at next checkpoint')
            else:
                status.info('No optimization thread running')
    # keep right_col available for layout continuity
    with right_col:
        pass

    # Remove static visualization image from interactive optimization view.
    # The UI uses live interim palettes from `/tmp/current_palette.npy` instead.
    img_holder.text('Use Refresh or Auto-refresh to view interim palettes (no static image shown).')

    # Always auto-refresh interim palette view every 1 second (1000 ms)
    if AUTOREFRESH_AVAILABLE and _st_autorefresh is not None:
        try:
            _st_autorefresh(1000, key="palette_autorefresh")
            if os.path.exists('/tmp/current_palette.npy'):
                try:
                    palette_colors = np.load('/tmp/current_palette.npy')
                except Exception:
                    st.error('Failed to load interim palette during auto-refresh')
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                with open('/tmp/autorefresh_error.txt', 'w') as f:
                    f.write(tb)
            except Exception:
                pass
            st.error('Auto-refresh failed to initialize during this session. See /tmp/autorefresh_error.txt for details.')
    else:
        st.info('Auto-refresh helper not available in the app environment. Install with `pip install streamlit-autorefresh` and restart the app to enable it.')
    # If the user clicked Refresh (or an interim exists), load the interim palette for native rendering
    if refresh or os.path.exists('/tmp/current_palette.npy'):
        try:
            palette_colors = np.load('/tmp/current_palette.npy')
        except Exception as e:
            st.error('Failed to load interim palette: ' + str(e))

    # If a palette was loaded (interim or final), render the native HTML palette and distance matrix
    if 'palette_colors' not in locals() and os.path.exists('/tmp/current_palette_final.npy'):
        try:
            palette_colors = np.load('/tmp/current_palette_final.npy')
        except Exception:
            palette_colors = None

    if 'palette_colors' in locals() and palette_colors is not None and palette_colors.size > 0:
        # combine additional colors (if any) with generated palette
        try:
            gen_palette = np.array(palette_colors)
            if gen_palette.ndim == 1:
                gen_palette = gen_palette.reshape(-1, 3)
            # Ensure generated palette has exactly the requested number of generated colors
            try:
                expected = int(n_to_generate)
            except Exception:
                expected = gen_palette.shape[0]
            # Always trim or pad to the requested size so UI matches the slider exactly
            if gen_palette.shape[0] >= expected:
                gen_palette = gen_palette[:expected]
            else:
                # pad with copies of the last color if for some reason fewer were found
                if gen_palette.shape[0] > 0:
                    pad = np.repeat(gen_palette[-1].reshape(1,3), expected - gen_palette.shape[0], axis=0)
                    gen_palette = np.vstack([gen_palette, pad])
                else:
                    # fallback: zeros
                    gen_palette = np.zeros((expected,3), dtype=int)
            # Prepend existing colors only when explicitly requested
            if include_additional_in_native and len(additional_colors) > 0:
                palette_colors = np.vstack([np.array(additional_colors), gen_palette])
            else:
                palette_colors = gen_palette
        except Exception:
            palette_colors = np.array(palette_colors)

        # Display color swatches as native HTML (full detail)
        st.subheader("Generated Color Palette (Native View)")
        cols = st.columns(len(palette_colors))
        for col_idx, (col, color) in enumerate(zip(cols, palette_colors)):
            with col:
                hex_code = f"#{int(color[0]):02X}{int(color[1]):02X}{int(color[2]):02X}"
                contrast = contrast_ratio(color, background_color)
                names = get_all_names(color)
                iscc_name = names.get('ISCC-NBS')
                ncs_name = names.get('NCS')
                pantone_name = names.get('Pantone')
                ral_name = names.get('RAL Classic')
                text_color = 'white' if calculate_luminance(color) < 0.5 else '#333333'
                swatch_html = f"""
                <div style="
                    background-color: {hex_code};
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
                        <div style="font-size: 9px; margin-bottom: 4px;">RGB({int(color[0])}, {int(color[1])}, {int(color[2])})</div>
                        <div style="font-size: 10px; font-weight: bold;">CR: {contrast:.2f}:1</div>
                    </div>
                </div>
                """
                st.markdown(swatch_html, unsafe_allow_html=True)

        # Color statistics table
        st.subheader("Color Details")
        color_data = []
        for i, color in enumerate(palette_colors, 1):
            hex_code = f"#{int(color[0]):02X}{int(color[1]):02X}{int(color[2]):02X}"
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

        max_distance = np.max(distance_matrix)
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
        for i in range(n_extended):
            color = extended_colors[i]
            hex_code = f"#{int(color[0]):02X}{int(color[1]):02X}{int(color[2]):02X}"
            html_parts.append(f'<th><div class="color-cell" style="background-color: {hex_code};"></div></th>')
        html_parts.append('</tr>\n</thead>\n<tbody>\n')
        for i in range(n_extended):
            color_i = extended_colors[i]
            hex_i = f"#{int(color_i[0]):02X}{int(color_i[1]):02X}{int(color_i[2]):02X}"
            html_parts.append(f'<tr><th><div class="color-cell" style="background-color: {hex_i};"></div></th>')
            for j in range(n_extended):
                distance = distance_matrix[i, j]
                if i == j:
                    html_parts.append('<td class="value-cell diagonal">&mdash;</td>')
                else:
                    intensity = distance / max_distance if max_distance > 0 else 0
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
        palette_start = 2
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
