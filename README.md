# 3D Feature Extractor

A desktop application for loading 3D models, extracting features using machine learning, and visualizing these features.

## Features

- Load and visualize 3D models in STL or OBJ format
- Extract features from 3D models using various methods:
  - Cluster by Position: Group points using KMeans clustering based on position
  - Cluster by Curvature: Group points using KMeans clustering based on position and curvature
  - Find Edges: Identify edges in the model based on curvature
- Visualize extracted features using color-coding

## Requirements

- Python 3.7+
- VTK 9.2+
- scikit-learn 1.2+
- NumPy 1.24+
- PyQt5 5.15+
- Matplotlib 3.7+

## Installation

1. Clone this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Run the application:

```
python main.py
```

2. Click "Load 3D Model" and select an STL or OBJ file
3. Select a feature extraction method from the dropdown menu
4. Adjust parameters as needed:
   - For clustering methods: Choose the number of clusters (2-10)
   - For edge detection: Adjust the threshold using the slider
5. Click "Extract Features" to process the model and visualize the results

## Controls

- Rotate: Left mouse button
- Pan: Middle mouse button
- Zoom: Right mouse button or scroll wheel

## Project Structure

- `main.py`: Main application entry point
- `app/gui.py`: User interface implementation
- `app/model_loader.py`: 3D model loading and processing
- `app/feature_extractor.py`: ML-based feature extraction
- `app/visualizer.py`: 3D visualization