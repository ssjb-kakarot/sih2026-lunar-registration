import numpy as np
import cv2
import plotly.graph_objects as go

def validate_visual_inputs(warped_img: np.ndarray, ref_img: np.ndarray):
    if warped_img is None or ref_img is None:
        raise ValueError("Image matrices cannot be None for visualization.")
    if warped_img.size == 0 or ref_img.size == 0:
        raise ValueError("Image matrices cannot be empty for visualization.")

def generate_3d_lunar_surface(warped_img: np.ndarray, ref_img: np.ndarray, apply_tint: bool = True):
    validate_visual_inputs(warped_img, ref_img)
    
    max_safe_dim = 800.0
    scale = max_safe_dim / float(max(max(ref_img.shape), max(warped_img.shape)))
    
    if scale < 1.0:
        r_small = cv2.resize(ref_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        w_small = cv2.resize(warped_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        r_small = ref_img.copy()
        w_small = warped_img.copy()
        
    if len(r_small.shape) == 3:
        r_small_gray = cv2.cvtColor(r_small, cv2.COLOR_BGR2GRAY)
    else:
        r_small_gray = r_small

    if len(w_small.shape) == 3:
        w_small_gray = cv2.cvtColor(w_small, cv2.COLOR_BGR2GRAY)
    else:
        w_small_gray = w_small

    dem = cv2.bilateralFilter(r_small_gray, d=15, sigmaColor=75, sigmaSpace=75)
    dem = cv2.GaussianBlur(dem, (11, 11), 0).astype(np.float32)
    
    dem_range = np.max(dem) - np.min(dem)
    if dem_range < 1e-6:
        dem_range = 1e-6
        
    dem = (dem - np.min(dem)) / dem_range
    z_elevation = dem * 2.5 

    surf_base = go.Surface(
        z=z_elevation,
        surfacecolor=r_small_gray,
        colorscale='gray',     
        showscale=False,
        name='Reference Base Matrix',
        opacity=0.9
    )

    mask = w_small_gray > 0
    z_warped = z_elevation.copy()
    
    if np.any(mask):
        z_warped[mask] += 0.2 
    z_warped[~mask] = np.nan 

    surf_warped = go.Surface(
        z=z_warped,
        surfacecolor=w_small_gray,
        colorscale='haline' if apply_tint else 'gray', 
        showscale=False,
        name='Aligned Payload'
    )

    try:
        fig = go.Figure(data=[surf_base, surf_warped])
        h, w = r_small_gray.shape
        fig.update_layout(
            autosize=True,
            height=900, 
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False, range=[-1, 5]),
                aspectratio=dict(x=1, y=h/w, z=0.1),
                camera=dict(eye=dict(x=0.0, y=-1.2, z=0.8)) 
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    except Exception as e:
        # Fallback to an empty figure safely if plotly strict sizing fails
        fig = go.Figure()
        fig.add_annotation(text="Unable to render 3D Topology for this payload.", showarrow=False, font=dict(color="white"))
        fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')
        return fig

def create_alpha_blend(img1, img2, alpha=0.5):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)