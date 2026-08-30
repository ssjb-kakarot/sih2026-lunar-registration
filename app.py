import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison

from engine_pipeline import SparkEngine
from utils_visual import generate_3d_lunar_surface
from report_pdf import generate_mission_report

st.set_page_config(
    page_title="ISRO Lunar Image Registration Tool", 
    page_icon="🌌", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    .stApp { 
        background: radial-gradient(circle at top left, #0b1120 0%, #020617 100%);
        color: #e2e8f0; 
        font-family: 'Inter', sans-serif; 
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    h1, h2, h3, h4, h5, h6 { 
        font-family: 'Space Grotesk', sans-serif !important; 
        color: #f8fafc;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .hero-container {
        padding: 3rem 2rem;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.8) 0%, rgba(2, 6, 23, 0) 100%);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 2rem;
        margin-top: -4rem;
        border-radius: 0 0 16px 16px;
    }
    .hero-title {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.98) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px dashed rgba(148, 163, 184, 0.4) !important;
        border-radius: 8px;
    }

    .stButton>button { 
        width: 100%;
        background: linear-gradient(135deg, #1e40af 0%, #0369a1 100%);
        color: #ffffff; 
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600; 
        border: 1px solid rgba(255,255,255,0.1); 
        border-radius: 8px; 
        padding: 0.6rem 0; 
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: translateY(-1px); 
        border-color: rgba(255,255,255,0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #94a3b8;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        padding: 8px 16px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(51, 65, 85, 0.9) !important;
        color: #ffffff !important;
    }

    .hud-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .hud-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.05);
        border-top: 2px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 8px;
    }
    .hud-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .hud-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        color: #f8fafc;
        font-weight: 600;
    }
    
    iframe[title*="image_comparison"] {
        width: 100% !important;
        height: 75vh !important;
        min-height: 700px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        background-color: #000;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_engine(): 
    return SparkEngine()

def save_upload(file) -> str:
    if file is None: return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(file.getvalue())
        return tmp.name

def main():
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Lunar Image Registration Interface</div>
            <div class="hero-subtitle">Multi-Modal Spatial & Photometric Alignment Tool</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div style="text-align: center; padding-bottom: 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg" width="100" style="opacity: 0.9;">
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### Image Data Input")
    src_file = st.sidebar.file_uploader("Source Image Payload", type=['png', 'jpg', 'tif', 'tiff'])
    ref_file = st.sidebar.file_uploader("Reference Base Map", type=['png', 'jpg', 'tif', 'tiff'])
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("### Process Parameters")
    
    use_tps = st.sidebar.toggle("Thin Plate Splines Transformation", value=True)
    use_extreme_zoom = st.sidebar.toggle("Dynamic Scale Pyramid", value=True)
    use_asift = st.sidebar.toggle("Affine-SIFT Simulation", value=True)
    enable_tints = st.sidebar.toggle("Visibility Tints Overlay", value=True)
    
    st.sidebar.markdown("""
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem; color: #64748b; text-align: center;">
            ISRO Hackathon 2026 | Problem Statement 26166<br>Registration Framework
        </div>
    """, unsafe_allow_html=True)

    if src_file and ref_file:
        st.markdown("### Input Images")
        col1, col2 = st.columns(2)
        col1.image(src_file, caption="Source Input", use_container_width=True)
        col2.image(ref_file, caption="Reference Matrix", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Execute Registration Pipeline", use_container_width=True):
            engine = load_engine()
            src_path, ref_path = save_upload(src_file), save_upload(ref_file)

            progress_bar = st.progress(0, text="Initializing operational resources...")

            def progress_callback(percent, message):
                progress_bar.progress(percent, text=message)

            try:
                result = engine.execute_pipeline(
                    src_path, ref_path, 
                    enable_asift=use_asift, 
                    enable_tps=use_tps,
                    enable_extreme_zoom=use_extreme_zoom,
                    progress_callback=progress_callback
                )
            except Exception as e:
                progress_bar.empty()
                st.error(f"Fatal Engine Execution Error: {str(e)}")
                return

            progress_bar.empty()

            if not result.get('success', False):
                st.error(f"Process Error: {result.get('message', 'Unknown analytical error')}")
                return

            st.success("Alignment Process Complete.")
            
            # Display fallback diagnostics safely without modifying structure
            diagnostics = result.get('diagnostics', {})
            if diagnostics.get('tps_fallback'):
                st.warning(f"Engine Fallback Triggered: {diagnostics.get('tps_fallback_msg')}", icon="⚠️")
            
            metrics = result['metrics']
            warped_img = result['warped_img']
            ref_raw = result['ref_raw']
            match_vis = result['match_vis']

            st.markdown("<br>", unsafe_allow_html=True)

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Visual Alignment", 
                "3D Topological Map", 
                "Evaluation Metrics", 
                "Detected Matches",
                "Export Records"
            ])

            with tab1:
                st.markdown("<h3 style='margin-bottom:0;'>Alignment Overlay Assesment</h3>", unsafe_allow_html=True)
                st.caption("Interactive axis allows evaluation of topographical accuracy.")
                
                target_width = 1600
                scale_factor = target_width / float(ref_raw.shape[1])
                target_height = int(ref_raw.shape[0] * scale_factor)
                
                ref_large = cv2.resize(ref_raw, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                warped_large = cv2.resize(warped_img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                
                ref_rgb = cv2.cvtColor(ref_large, cv2.COLOR_GRAY2RGB)
                warped_rgb = cv2.cvtColor(warped_large, cv2.COLOR_GRAY2RGB)
                
                if enable_tints:
                    cyan_tint = np.zeros_like(warped_rgb)
                    cyan_tint[:, :, 0] = 50 
                    cyan_tint[:, :, 1] = 50 
                    mask = (warped_large > 0)[:, :, None] 
                    warped_rgb = np.where(mask, cv2.add(warped_rgb, cyan_tint), warped_rgb)

                try:
                    image_comparison(
                        img1=ref_rgb, 
                        img2=warped_rgb, 
                        label1="Reference Image", 
                        label2="Registered Source" + (" (Enhanced Display)" if enable_tints else ""),
                        width=1600,
                        make_responsive=True
                    )
                except Exception:
                    st.image(cv2.addWeighted(ref_rgb, 0.6, warped_rgb, 0.4, 0), use_container_width=True)

            with tab2:
                st.markdown("<h3 style='margin-bottom:0;'>3D Intensity Topography</h3>", unsafe_allow_html=True)
                st.caption("Representation mapped upon pseudo-elevation generated from image intensity (Not a true DEM).")
                with st.spinner("Processing rendering arrays..."):
                    fig = generate_3d_lunar_surface(warped_img, ref_raw, apply_tint=enable_tints)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1'))
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.markdown("<h3 style='margin-bottom:0;'>Quality Parameters</h3>", unsafe_allow_html=True)
                st.caption("Quantitative data defining the precision of alignment.")
                
                status_color = "#10b981" if metrics.get('quality_status') in ["Excellent", "Good"] else "#f59e0b" if metrics.get('quality_status') == "Marginal" else "#ef4444"
                
                hud_html = f"""
                <div class="hud-grid">
                    <div class="hud-card">
                        <div class="hud-label">RMS Error</div>
                        <div class="hud-value">{metrics.get('RMSE', 0.0):.4f} <span style="font-size:0.5em;color:#94a3b8;">px</span></div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-label">Verified Inliers</div>
                        <div class="hud-value">{metrics.get('Inliers', 0)}</div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-label">Inlier Ratio</div>
                        <div class="hud-value">{metrics.get('Inlier_Ratio', 0.0)*100:.1f}<span style="font-size:0.5em;color:#94a3b8;">%</span></div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-label">Mutual Information</div>
                        <div class="hud-value">{metrics.get('Mutual_Info', 0.0):.3f}</div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-label">SSIM Measurement</div>
                        <div class="hud-value">{metrics.get('SSIM', 0.0):.2f}<span style="font-size:0.5em;color:#94a3b8;">%</span></div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-label">Median Error</div>
                        <div class="hud-value">{metrics.get('validation_median_error', 0.0):.2f}<span style="font-size:0.5em;color:#94a3b8;">px</span></div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-label">Alignment Status</div>
                        <div class="hud-value" style="color: {status_color}; font-size:1.4rem;">{metrics.get('quality_status', 'Unknown')}</div>
                    </div>
                </div>
                """
                st.markdown(hud_html, unsafe_allow_html=True)

            with tab4:
                st.markdown("<h3 style='margin-bottom:0;'>Correlation Visualisation</h3>", unsafe_allow_html=True)
                st.caption(f"Identifying points structure evaluated at dynamic scale factor: {result.get('best_scale_ratio', 'N/A')}")
                if match_vis is not None:
                    st.image(match_vis, use_container_width=True)

            with tab5:
                st.markdown("<h3 style='margin-bottom:0;'>Record Export</h3>", unsafe_allow_html=True)
                st.caption("Download reports and matrix products for secondary usage.")
                
                try:
                    pdf_bytes = generate_mission_report(
                        metrics=metrics, 
                        src_shape=result.get('src_shape', (0,0)), 
                        ref_shape=result.get('ref_shape', (0,0)), 
                        tps_requested=use_tps, 
                        asift_enabled=use_asift, 
                        scale_ratio=str(result.get('best_scale_ratio', 'N/A')),
                        diagnostics=diagnostics,
                        actual_tps_used=result.get('actual_tps_used', False)
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        st.download_button(
                            label="Export Evaluation Report (.PDF)", 
                            data=pdf_bytes, 
                            file_name="ISRO_Registration_Report.pdf", 
                            mime="application/pdf", 
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error during report compilation: {e}")
                
                if warped_img is not None:
                    is_success, buffer = cv2.imencode(".tif", warped_img)
                    if is_success:
                        with col_btn2:
                            st.download_button(
                                label="Export Result Image (.TIF)", 
                                data=buffer.tobytes(), 
                                file_name="Aligned_Output_Matrix.tif", 
                                mime="image/tiff",
                                use_container_width=True
                            )
    else:
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 30vh; opacity: 0.6;">
                <p style="font-size: 1.1rem; color: #94a3b8; font-family: 'Inter', sans-serif;">Select and input datasets from the panel to initiate processing.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()