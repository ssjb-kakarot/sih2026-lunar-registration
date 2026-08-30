================================================================================
SMART INDIA HACKATHON 2026 
Problem Statement ID: 26166
Theme: Space Technology | Organization: ISRO / Department of Space
Title: Multi-modal, Sun angle and scale invariant image correspondence using 
       Chandrayaan-2 optical images (OHRC, TMC and IIRS)
================================================================================

An advanced, generic software solution designed to seamlessly register Lunar Source 
Payload Imagery against Reference Base Maps. This application utilizes a robust 
computer vision pipeline to handle extreme topological, illumination, and scale 
variances, delivering sub-pixel accuracy and comprehensive evaluation metrics.

--------------------------------------------------------------------------------
1. DATASET SOURCES (FOR TESTING)
--------------------------------------------------------------------------------
To test this software, utilize the official datasets provided by ISRO and NASA:

Source Payload Images (Chandrayaan-2 OHRC, TMC-2, IIRS):
* ISSDC Portal: https://chmapbrowse.issdc.gov.in/

Reference Images (LRO NAC, SELENE):
* LROC QuickMap (Interactive): https://quickmap.lroc.im-ldi.com
* LROC Image Downloads: https://lroc.im-ldi.com/images/downloads/

--------------------------------------------------------------------------------
2. HOW THIS SOLUTION ADDRESSES THE PROBLEM STATEMENT
--------------------------------------------------------------------------------
* Illumination Variation (Sun azimuth/elevation): 
  Solved using a Multi-Scale Retinex algorithm combined with CLAHE. This normalizes 
  harsh lunar lighting and shadow differences, extracting pure topological structures 
  invariant to the sun angle.

* Viewpoint Variation (Geometric/Perspective distortions): 
  Solved using a custom ASIFT (Affine-SIFT) implementation. Multi-threaded workers 
  simulate tilt and rotation parameters to detect true viewpoint-invariant features.

* Scale Variation (Different altitudes/resolutions): 
  Solved using a Dynamic Scale Pyramid. The engine automatically rescales the payload 
  images across extreme zoom levels to establish correlation with global reference maps.

* Sub-pixel Accuracy & Uniform Distribution: 
  Matches are sorted into a spatial grid (Bucketing) to ensure they span the entire 
  image equally rather than clustering. These matches are then refined using fast 
  corner sub-pixel math, pushing spatial accuracy well below a single pixel limit.

--------------------------------------------------------------------------------
3. SETUP & INSTALLATION GUIDE
--------------------------------------------------------------------------------
Requirements:
* Python 3.10 or higher is REQUIRED. 
* Windows PowerShell (for the exact commands below).

Step 1: Open your terminal/PowerShell and navigate to the project folder.

Step 2: Create a virtual environment specifically using Python 3.10:
  > py -3.10 -m venv venv  

Step 3: Activate the virtual environment:
  > .\venv\Scripts\Activate.ps1  

Step 4: Install the required dependencies:
  > pip install opencv-python numpy streamlit streamlit-image-comparison fpdf2 scikit-image plotly
  (Alternatively, you can run: pip install -r requirements.txt)

--------------------------------------------------------------------------------
4. RUNNING THE APPLICATION
--------------------------------------------------------------------------------
Launch the application locally by running:
  > streamlit run app.py

The terminal will output a Local URL (usually http://localhost:8501). 
Open this link in your web browser to access the tool.

--------------------------------------------------------------------------------
5. USAGE GUIDE
--------------------------------------------------------------------------------
1. Input Datasets: 
   Upload your Source Image (e.g., Chandrayaan-2) and Reference Base Map (e.g., LRO NAC). 
   Supports .png, .jpg, .tif, .tiff.
   
2. Configure Parameters (Left Sidebar):
   - Thin Plate Splines: Toggle for non-rigid mapping (adapts specifically to craters).
   - Dynamic Scale Pyramid: Leave ON to handle altitude/resolution gaps.
   - Affine-SIFT Simulation: Leave ON for angled payload images.
   - Visibility Tints: Adds a cyan overlay for visual contrast on the reference map.
   
3. Execute: 
   Click "Execute Registration Pipeline" and wait for the multi-threaded processing.
   
4. Analyze Output Tabs:
   - Visual Alignment: Interactive slider to visually assess spatial overlay.
   - 3D Topological Map: Plotly-rendered 3D pseudo-elevation mapped surface.
   - Evaluation Metrics: The required SIH metrics (RMSE, Inliers, SSIM, etc.).
   - Detected Matches: Review the generated scale-invariant match points.
   - Export Records: Download the Aligned .TIF Product and the PDF Evaluation Report.

================================================================================
Developed for SMART INDIA HACKATHON 2026 | Indian Space Research Organisation
================================================================================
