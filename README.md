<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg" alt="ISRO Logo" width="120" />

  <h1>🌌 Lunar Image Registration & Alignment Engine</h1>
  <p><strong>Multi-modal, Sun angle, and Scale-invariant Image Correspondence</strong></p>

  <p>
    <img src="https://img.shields.io/badge/SIH-2026-orange?style=for-the-badge&logo=hackerearth" alt="SIH 2026" />
    <img src="https://img.shields.io/badge/Problem-26166-blue?style=for-the-badge" alt="PS 26166" />
    <img src="https://img.shields.io/badge/Python-3.10%2B-green?style=for-the-badge&logo=python" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" alt="License" />
  </p>
  
  <i>Developed for the Smart India Hackathon 2026 • Department of Space / ISRO</i>
</div>

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Problem Statement Breakdown](#-problem-statement-breakdown)
3. [The Core Challenges & Our Solutions](#-the-core-challenges--our-solutions)
4. [Algorithmic Architecture (Deep Dive)](#-algorithmic-architecture-deep-dive)
5. [Evaluation Metrics & Verification](#-evaluation-metrics--verification)
6. [Datasets & Sources](#-datasets--sources)
7. [System Architecture](#-system-architecture)
8. [Installation & Setup (Windows/PowerShell)](#-installation--setup-windowspowershell)
9. [User Interface & Usage Guide](#-user-interface--usage-guide)
10. [Future Scope](#-future-scope)

---

## 🚀 Executive Summary

This repository contains a highly optimized, generic software solution for aligning Lunar Source Payload Imagery (Chandrayaan-2 OHRC, TMC-2, IIRS) against Global Reference Base Maps (LRO NAC, SELENE). 

Built using a robust computer vision pipeline, this tool seamlessly resolves extreme variations in **illumination, viewpoint, and scale** by implementing mathematical models such as **Affine-SIFT (ASIFT), Multi-Scale Retinex, Thin Plate Splines (TPS)**, and **Sub-pixel Corner Refinement**. The solution outputs registered datasets with sub-pixel accuracy, wrapped in an interactive UI that auto-generates comprehensive mission reports.

---

## 🎯 Problem Statement Breakdown

* **Organization:** Indian Space Research Organisation (ISRO)
* **Theme:** Space Technology
* **Category:** Software
* **Objective:** Develop a generic software solution for finding correspondence between Chandrayaan-2 acquired optical images and Lunar reference images with **sub-pixel accuracy**, maintaining a **uniform distribution** of match points across the images.

---

## 🧩 The Core Challenges & Our Solutions

### 1. Illumination Variation ☀️
* **The Problem:** Changes in sun azimuth and elevation drastically alter the appearance of lunar craters and topological features, making direct pixel-matching impossible.
* **Our Solution:** We pre-process both image matrices through a **Multi-Scale Retinex (MSR)** algorithm paired with **CLAHE**. This strips away the harsh lighting and shadow gradients, leaving only the pure, illumination-invariant structural topography of the lunar surface.

### 2. Viewpoint Variation 🛰️
* **The Problem:** Geometric distortions caused by different orbital trajectories (camera positions/orientations). Objects appear shifted or perspective-distorted.
* **Our Solution:** A custom multi-threaded **Affine-SIFT (ASIFT)** implementation. The engine mathematically simulates varying longitudinal tilts and latitudinal rotations (phi angles) on the source image, extracting features that are fundamentally viewpoint-invariant.

### 3. Scale Variation 🔍
* **The Problem:** Chandrayaan-2 operates at different altitudes and resolutions compared to the LRO Reference maps.
* **Our Solution:** A **Dynamic Scale Pyramid**. The system iteratively tests the payload image against the reference map at extreme zoom levels (from 4.0x down to 0.125x) to automatically lock onto the correct spatial frequency.

---

## 🔬 Algorithmic Architecture (Deep Dive)

To achieve the sub-pixel accuracy and uniform distribution demanded by ISRO, our pipeline executes the following sequential algorithms:

### A. Spatial Quadtree Bucketing (Uniform Distribution)
Highly textured areas (like complex craters) naturally attract thousands of feature matches, while flat lunar mare are ignored. If all points cluster in one crater, the global alignment fails. 
* **Mechanism:** We overlay a mathematical grid (e.g., 10x10) across the image. The algorithm sorts match points into these buckets and caps the maximum points per bucket (e.g., max 50). This forces the homography matrix to account for the *entire* spatial extent of the payload image.

### B. Sub-Pixel Corner Refinement
Standard feature matching snaps to the nearest integer pixel. For high-resolution lunar mapping, this introduces unacceptable geometric error.
* **Mechanism:** We utilize `cv2.cornerSubPix`. By analyzing the intensity gradients in the local neighborhood (5x5 window) of an initial match, the algorithm calculates the exact sub-pixel coordinate of the topological corner using iterative mathematical optimization.

### C. Thin Plate Splines (TPS) vs. Robust Homography
* **Primary Transformation:** A global `USAC_MAGSAC` (RANSAC-based) Homography matrix is calculated to handle planar projection.
* **Non-Rigid Transformation (TPS):** The lunar surface is not a flat plane; it has elevation (craters, mountains). When enabled, the engine maps the sub-pixel points using a **Thin Plate Spline shape transformer**. This allows the image to warp non-rigidly, perfectly wrapping the source payload over the reference topography. 

---

## 📊 Evaluation Metrics & Verification

The engine provides real-time quantitative feedback to verify alignment quality, compliant with ISRO's evaluation requirements:

* **RMSE (Root Mean Square Error):** Measures the geometric distance between the projected source points and the reference points. Lower is better (Sub-pixel RMSE achieved).
* **Inlier Match Count:** The total number of verified, geometrically consistent correspondences.
* **Inlier Ratio:** The percentage of matches that survived spatial filtering and RANSAC verification.
* **Mutual Information:** Measures the statistical dependence and entropy overlap between the two images post-registration.
* **SSIM (Structural Similarity Index):** Computes the perceived structural quality difference after alignment.

---

## 📂 Datasets & Sources

This software is designed to consume optical datasets from the following official repositories:

| Dataset Type | Satellite / Sensor | Access Portal |
| :--- | :--- | :--- |
| **Source Payload** | Chandrayaan-2 (OHRC, TMC-2, IIRS) | [ISSDC Map Browse](https://chmapbrowse.issdc.gov.in/) |
| **Reference Maps** | LRO NAC (Lunar Reconnaissance Orbiter) | [LROC QuickMap](https://quickmap.lroc.im-ldi.com) |
| **Reference Maps** | SELENE (Kaguya) | [LROC Data Archive](https://lroc.im-ldi.com/images/downloads/) |

---

## 🏗️ System Architecture

```text
📦 sih2026-lunar-registration
 ┣ 📜 app.py               # Streamlit GUI & Visual Render Engine
 ┣ 📜 engine_pipeline.py   # Core registration execution logic
 ┣ 📜 algo_asift.py        # Multi-threaded Affine-SIFT logic
 ┣ 📜 utils_processing.py  # Retinex, TPS, Metrics, & Sub-pixel math
 ┣ 📜 utils_visual.py      # Plotly 3D Surface Topography generation
 ┣ 📜 report_pdf.py        # Dynamic PDF Mission Report compiler
 ┗ 📜 requirements.txt     # Dependency matrix
```

---

## 💻 Installation & Setup (Windows/PowerShell)

**Prerequisite:** Ensure you have **Python 3.10+** installed on your system.

**1. Clone the repository and navigate into it:**
```powershell
git clone https://github.com/ssjb-kakarot/sih2026-lunar-registration
cd sih2026-lunar-registration
```

**2. Create a Python 3.10 Virtual Environment:**
```powershell
py -3.10 -m venv venv
```

**3. Activate the Virtual Environment:**
```powershell
.\venv\Scripts\Activate.ps1
```

**4. Install Dependencies:**
```powershell
pip install opencv-python numpy streamlit streamlit-image-comparison fpdf2 scikit-image plotly
```

**5. Launch the Application Engine:**
```powershell
streamlit run app.py
```
*The application will boot up automatically in your default web browser at `http://localhost:8501`.*

---

## 🎮 User Interface & Usage Guide

Our solution utilizes a streamlined, dark-mode GUI designed for space-mission command interfaces.

### Step 1: Input Data
Upload your Source Image (Chandrayaan-2) and your Reference Base Map (LRO NAC) via the left sidebar.

### Step 2: Toggle Execution Parameters
* **Thin Plate Splines (TPS):** Adapts specifically to extreme elevation changes (craters).
* **Dynamic Scale Pyramid:** Essential if your images have massive spatial resolution gaps.
* **Affine-SIFT Simulation:** Essential for heavily angled payload shots.
* **Visibility Tints:** Overlays a cyan tint on the output matrix for easier visual differentiation.

### Step 3: Analytics & Exports
Once the engine completes the multi-threaded processing, navigate the tabs to view:
1. **Visual Alignment:** Interactive slider overlaying the registered output onto the reference.
2. **3D Topological Map:** An interactive, manipulatable 3D surface generated by interpreting image intensities as pseudo-elevation data.
3. **Evaluation Metrics:** A detailed Heads-Up Display (HUD) showing RMSE, Inliers, Coverage, and SSIM.
4. **Export Records:** Download the geometrically aligned `.TIF` image and the automated **ISRO Mission Report (`.PDF`)**.

---

## 🔮 Future Scope

While this solution solves Problem Statement 26166, the architecture is designed to scale:
* **GPU Acceleration:** Future integration with `CuPy` and OpenCV-CUDA for near-instantaneous ASIFT processing.
* **Multi-Spectral Support:** Adapting the mutual information metric to perfectly align thermal/hyperspectral imagery with optical base maps.
* **Automated Batch Processing:** Allowing CLI-based folder-to-folder bulk registration of orbital image dumps.

---
<div align="center">
  <b>Built with logic, precision, and passion for the Indian Space Research Organisation.</b><br>
  <i>Smart India Hackathon 2026</i>
</div>
