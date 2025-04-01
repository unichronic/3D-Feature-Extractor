import os
import sys
import vtk
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                            QPushButton, QFileDialog, QComboBox, QLabel, 
                            QSlider, QSpinBox, QFrame)
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from app.model_loader import ModelLoader
from app.feature_extractor import FeatureExtractor
from app.visualizer import Visualizer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Feature Extractor")
        self.resize(1200, 800)
        
        # Create model loader, feature extractor, and visualizer
        self.model_loader = ModelLoader()
        self.feature_extractor = FeatureExtractor()
        self.visualizer = Visualizer()
        
        # Create the UI
        self.setup_ui()
        
        # Initialize state
        self.model_loaded = False
        
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create VTK widget for rendering
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        
        # Get the render window provided by the QVTK widget
        render_window = self.vtk_widget.GetRenderWindow()
        # Add the renderer managed by our Visualizer class
        render_window.AddRenderer(self.visualizer.get_renderer())
        # Get the interactor provided by the QVTK widget
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        # Set the interactor style
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        # Initialize the interactor (do NOT call Start())
        interactor.Initialize()
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        # Load model section
        load_section = QWidget()
        load_layout = QVBoxLayout()
        load_section.setLayout(load_layout)
        
        load_label = QLabel("Model Loading")
        load_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.load_button = QPushButton("Load 3D Model")
        self.load_button.clicked.connect(self.load_model)
        
        load_layout.addWidget(load_label)
        load_layout.addWidget(self.load_button)
        
        # Feature extraction section
        feature_section = QWidget()
        feature_layout = QVBoxLayout()
        feature_section.setLayout(feature_layout)
        
        feature_label = QLabel("Feature Extraction")
        feature_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Cluster by Position", "Cluster by Curvature", "Find Edges"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        
        # Parameters section
        params_layout = QHBoxLayout()
        
        # Cluster count for clustering methods
        self.cluster_label = QLabel("Clusters:")
        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 10)
        self.cluster_spin.setValue(5)
        
        # Threshold for edge detection
        self.thresh_label = QLabel("Threshold:")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(70)  # 0.7 by default
        
        params_layout.addWidget(self.cluster_label)
        params_layout.addWidget(self.cluster_spin)
        params_layout.addWidget(self.thresh_label)
        params_layout.addWidget(self.thresh_slider)
        
        # Initially hide the threshold controls
        self.thresh_label.setVisible(False)
        self.thresh_slider.setVisible(False)
        
        # Connect the method combo box to update the visible parameters
        self.method_combo.currentIndexChanged.connect(self.update_parameter_visibility)
        
        self.extract_button = QPushButton("Extract Features")
        self.extract_button.clicked.connect(self.extract_features)
        self.extract_button.setEnabled(False)
        
        feature_layout.addWidget(feature_label)
        feature_layout.addLayout(method_layout)
        feature_layout.addLayout(params_layout)
        feature_layout.addWidget(self.extract_button)
        
        # Add sections to control panel
        control_layout.addWidget(load_section)
        control_layout.addWidget(QHLine())
        control_layout.addWidget(feature_section)
        control_layout.addStretch()
        
        # Add widgets to main layout and set central widget
        main_layout.addWidget(self.vtk_widget, 1)
        main_layout.addWidget(control_panel)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def update_parameter_visibility(self):
        method = self.method_combo.currentText()
        
        if method == "Find Edges":
            self.cluster_label.setVisible(False)
            self.cluster_spin.setVisible(False)
            self.thresh_label.setVisible(True)
            self.thresh_slider.setVisible(True)
        else:
            self.cluster_label.setVisible(True)
            self.cluster_spin.setVisible(True)
            self.thresh_label.setVisible(False)
            self.thresh_slider.setVisible(False)
    
    def load_model(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open 3D Model", "", "3D Models (*.stl *.obj)"
        )
        
        if file_path:
            try:
                # Clear the renderer via the visualizer
                self.visualizer.remove_all_actors()
                
                # Load the model (returns the actor)
                actor = self.model_loader.load_model(file_path)
                
                # Add actor to visualizer's renderer
                self.visualizer.add_actor(actor)
                # Reset the camera via the visualizer's renderer
                self.visualizer.reset_camera()
                # Render the VTK widget explicitly
                self.vtk_widget.GetRenderWindow().Render()
                
                # Enable extract button
                self.extract_button.setEnabled(True)
                self.model_loaded = True
                
            except Exception as e:
                print(f"Error loading model: {e}")
                # Optionally show an error message box to the user
                self.extract_button.setEnabled(False)
                self.model_loaded = False
    
    def extract_features(self):
        if not self.model_loaded:
            print("No model loaded to extract features from.")
            return
        
        try:
            # Get the method and parameters
            method = self.method_combo.currentText()
            points = self.model_loader.get_points()
            polydata = self.model_loader.polydata
            
            if points is None or polydata is None:
                print("Error: Model data is not available.")
                return
            
            labels = None
            
            # Extract features based on selected method
            if method == "Cluster by Position":
                n_clusters = self.cluster_spin.value()
                labels = self.feature_extractor.cluster_by_position(points, n_clusters)
            elif method == "Cluster by Curvature":
                n_clusters = self.cluster_spin.value()
                # Ensure polydata is passed correctly
                labels = self.feature_extractor.cluster_by_curvature(points, polydata, n_clusters)
            elif method == "Find Edges":
                threshold = self.thresh_slider.value() / 100.0
                # Ensure polydata is passed correctly
                labels = self.feature_extractor.find_edges(points, polydata, threshold)
            
            if labels is None:
                print("Feature extraction did not produce labels.")
                return
            
            # Update model colors based on labels (ModelLoader updates the actor)
            self.model_loader.update_colors(labels)
            
            # Render the VTK widget explicitly to show changes
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Optionally show an error message box to the user
    
    def closeEvent(self, event):
        # No need to finalize the vtk_widget explicitly here,
        # Qt handles widget cleanup.
        # If there were other resources to clean, do it here.
        super().closeEvent(event) # Ensure default behavior happens

# Horizontal line widget for UI separation
class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken) 