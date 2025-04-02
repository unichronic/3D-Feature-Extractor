# 3D Feature Extractor

A desktop application for loading 3D models, extracting features using machine learning, and visualizing these features. The application provides advanced 3D model visualization with interactive gesture control, feature extraction, connectivity analysis, clipping tools, and customizable lighting.

## Features

- **3D Model Visualization**: Load and visualize 3D models in STL or OBJ format
- **Feature Extraction**: Apply machine learning techniques to extract and visualize model features:
  - **Cluster by Position**: Group points using KMeans clustering based on spatial position
  - **Cluster by Curvature**: Group points using KMeans clustering based on position and curvature
  - **Find Edges**: Identify edges in the model based on curvature
- **Model Analysis Tools**:
  - **Connectivity Segmentation**: Analyze and visualize connected components with distinct colors
  - **Clipping Planes**: Apply clipping planes to see inside models
- **Gesture Control**: Control the camera view using hand gestures via webcam
- **Customizable Lighting**: Adjust ambient and diffuse lighting to improve visualization

## Installation

1. Clone this repository: 
```
https://github.com/unichronic/Feature-Extractor.git
```
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage Guide

### Basic Usage

1. Run the application:

```
python main.py
```

2. Click "Load 3D Model" and select an STL or OBJ file
3. Use mouse controls to navigate the 3D view:
   - Rotate: Left mouse button drag
   - Pan: Middle mouse button drag
   - Zoom: Right mouse button drag or scroll wheel

### Feature Extraction

1. Select a feature extraction method from the dropdown menu:
   - **Cluster by Position**: Groups points based on their spatial coordinates
   - **Cluster by Curvature**: Groups points based on both position and surface curvature
   - **Find Edges**: Highlights edges based on surface curvature

2. Adjust parameters as needed:
   - For clustering methods: Choose the number of clusters (2-10)
   - For edge detection: Adjust the threshold using the slider (0-100%)

3. Click "Extract Features" to process the model
   - Results are visualized with color-coding directly on the model

### Model Analysis

#### Connectivity Segmentation

1. Click "Segment by Connectivity" to analyze and visualize separate connected components
   - Each disconnected component will be displayed in a distinct color
   - Useful for identifying separate parts in assemblies or checking model integrity

#### Clipping Plane

1. Click "Apply Clip Plane" to create a cutting plane through the model
   - The default plane is YZ at the center of the model
2. Click "Toggle Clip Side" to switch between viewing different sides of the clip
3. Click "Remove Clip" to restore the full model view

### Gesture Control

1. Click "Start Gesture Control" to enable camera-based hand gesture control
   - Requires a webcam and sufficient lighting
   - A camera preview window will open showing hand tracking

2. Use the following gestures:
   - **Pinching**: Pinch your thumb and index finger together to zoom
     - Moving fingers closer/further apart controls zoom level
   - **Swiping**: Move your open hand to rotate the model
     - Horizontal movement rotates around Y-axis
     - Vertical movement rotates around X-axis

3. Adjust gesture sensitivity using the sliders:
   - **Zoom Sensitivity**: Controls pinch gesture responsiveness
   - **Rotation Sensitivity**: Controls swipe gesture responsiveness

4. Click "Stop Gesture Control" to disable camera tracking

### Lighting Control

1. Adjust the light settings using the sliders in the Lighting Controls section:
   - **Ambient**: Controls the overall base light level (0-100%)
   - **Diffuse**: Controls directional light intensity (0-100%)

2. Changes to lighting are applied immediately to improve model visualization

