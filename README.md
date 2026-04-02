<div align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">

  <h1>🌌 Vector Calculus: Sobel Edge Detection </h1>
  
  <p><b>A sleek, minimal web application to explore the mathematical beauty of edge detection.</b></p>
</div>

<br/>

## 🎯 Overview

This project provides an interactive **Streamlit Web Application** designed to visualize the **Sobel Edge Detection** algorithm. 
By applying principles of vector calculus directly to pixel intensity fields, the app computes both gradient components ($G_x$ and $G_y$) to extract hard edges dynamically.

It's completely built from scratch using raw matrix operations with **NumPy**, devoid of high-level OpenCV abstractions. It highlights:
- `x-gradient` calculation using convolution.
- `y-gradient` calculation using convolution.
- Image gradient **Magnitude** visualization.
- Thresholding for binary edge representation.

---

## ✨ Features

- 🖼️ **Image Processing From Scratch**: Implemented native convolution and edge-finding logic exclusively using Python and NumPy.
- 🎛️ **Live Parameter Adjustments**: Interactive slider for tuning the Edge Threshold to see how gradient magnitudes translate directly to edge maps.
- 📊 **Vector Calculus Visualization**: Breaks down the edge detection into its partial derivative matrices ($G_x$ and $G_y$).
- ⚡ **Streamlit Backend**: A smooth, reactive, and professional dark-mode-ready UI.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed.

```bash
# Clone the repository
git clone https://github.com/your-username/edge-detector.git
cd edge-detector

# Install dependencies (Streamlit, Pillow, NumPy)
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit web app by running:

```bash
streamlit run app.py
```

The application will launch on your local network (usually `localhost:8501`).

---

## 🧠 How it Works

The app implements the **Sobel Operator**, a discrete differentiation operator used to compute an approximation of the gradient of the image intensity function.

1. **Greyscaling:** The image is translated into a 2D matrix of intensities.
2. **Convolution kernel application:** Matrices $G_x$ and $G_y$ (Sobel kernels) are convolved with the original image to calculate approximations of the derivatives - one for horizontal changes, and one for vertical.
3. **Magnitude resolution:** The gradient magnitude is computed as $\sqrt{G_x^2 + G_y^2}$.
4. **Thresholding:** Pixels with a magnitude higher than the user-input limit are classified as edges and rendered.

---

## 📂 Project Structure

```text
├── app.py                            # Streamlit entry point & visualizer
├── src/
│   └── edge_detection_algorithms/
│       ├── sobel_edge_detector.py    # Core Sobel convolution logic
│       ├── helper_greyscale.py       # Intensity mapping
│       └── helper_blur.py            # Pre-processing noise reduction
├── requirements.txt                  # App dependencies
└── README.md                         # You are here!
```

---

<div align="center">
  <i>Developed with ❤️ for mathematical computer vision.</i>
</div>
