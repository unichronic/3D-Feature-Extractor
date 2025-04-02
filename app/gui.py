import os
import sys
import vtk
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                            QPushButton, QFileDialog, QComboBox, QLabel, 
                            QSlider, QSpinBox, QFrame, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.cm as cm
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap

from app.model_loader import ModelLoader
from app.feature_extractor import FeatureExtractor
from app.visualizer import Visualizer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Feature Extractor")
        self.resize(1200, 800)
        
        self.model_loader = ModelLoader()
        self.feature_extractor = FeatureExtractor()
        self.visualizer = Visualizer()
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Warning: Could not open webcam.")
            self.camera_available = False
        else:
            self.camera_available = True

        self.gesture_timer = QTimer(self)
        self.gesture_timer.timeout.connect(self.process_gesture_frame)

        self.gesture_state = None
        self.previous_pinch_distance = None
        self.previous_wrist_position = None
        self.zoom_sensitivity = 0.9
        self.rotation_sensitivity = 0.3
        self.swipe_velocity_threshold = 4
        
        self.setup_ui()
        
        self.model_loaded = False
        
        self.clipper = None
        self.clip_plane = None
        
    def setup_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self.visualizer.get_renderer())
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.Initialize()
        
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        load_section = QWidget()
        load_layout = QVBoxLayout()
        load_section.setLayout(load_layout)
        
        load_label = QLabel("Model Loading")
        load_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.load_button = QPushButton("Load 3D Model")
        self.load_button.clicked.connect(self.load_model)
        
        load_layout.addWidget(load_label)
        load_layout.addWidget(self.load_button)
        
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
        
        params_layout = QHBoxLayout()
        
        self.cluster_label = QLabel("Clusters:")
        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 10)
        self.cluster_spin.setValue(5)
        
        self.thresh_label = QLabel("Threshold:")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(70)
        
        params_layout.addWidget(self.cluster_label)
        params_layout.addWidget(self.cluster_spin)
        params_layout.addWidget(self.thresh_label)
        params_layout.addWidget(self.thresh_slider)
        
        self.thresh_label.setVisible(False)
        self.thresh_slider.setVisible(False)
        
        self.method_combo.currentIndexChanged.connect(self.update_parameter_visibility)
        
        self.extract_button = QPushButton("Extract Features")
        self.extract_button.clicked.connect(self.extract_features)
        self.extract_button.setEnabled(False)
        
        feature_layout.addWidget(feature_label)
        feature_layout.addLayout(method_layout)
        feature_layout.addLayout(params_layout)
        feature_layout.addWidget(self.extract_button)
        
        gesture_section = QWidget()
        gesture_layout = QVBoxLayout()
        gesture_section.setLayout(gesture_layout)

        gesture_label = QLabel("Gesture Control")
        gesture_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        sensitivity_layout = QHBoxLayout()
        zoom_label = QLabel("Zoom Sensitivity:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 100)
        self.zoom_slider.setValue(int(self.zoom_sensitivity * 100))
        self.zoom_slider.valueChanged.connect(self.update_zoom_sensitivity)
        
        sensitivity_layout.addWidget(zoom_label)
        sensitivity_layout.addWidget(self.zoom_slider)
        
        rotation_layout = QHBoxLayout()
        rotation_label = QLabel("Rotation Sensitivity:")
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(5, 100)
        self.rotation_slider.setValue(int(self.rotation_sensitivity * 100))
        self.rotation_slider.valueChanged.connect(self.update_rotation_sensitivity)
        
        rotation_layout.addWidget(rotation_label)
        rotation_layout.addWidget(self.rotation_slider)
        
        self.start_gesture_button = QPushButton("Start Gesture Control")
        self.start_gesture_button.clicked.connect(self.start_gesture_control)
        self.start_gesture_button.setEnabled(self.camera_available)

        self.stop_gesture_button = QPushButton("Stop Gesture Control")
        self.stop_gesture_button.clicked.connect(self.stop_gesture_control)
        self.stop_gesture_button.setEnabled(False)

        gesture_layout.addWidget(gesture_label)
        gesture_layout.addLayout(sensitivity_layout)
        gesture_layout.addLayout(rotation_layout)
        gesture_layout.addWidget(self.start_gesture_button)
        gesture_layout.addWidget(self.stop_gesture_button)
        
        analysis_section = QWidget()
        analysis_layout = QVBoxLayout()
        analysis_section.setLayout(analysis_layout)

        analysis_label = QLabel("Model Analysis")
        analysis_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.segment_button = QPushButton("Segment by Connectivity")
        self.segment_button.clicked.connect(self.segment_by_connectivity)
        self.segment_button.setEnabled(False)

        self.clip_plane_button = QPushButton("Apply Clip Plane (YZ at Center)")
        self.clip_plane_button.clicked.connect(self.apply_clip_plane)
        self.clip_plane_button.setEnabled(False)

        self.toggle_clip_button = QPushButton("Toggle Clip Side")
        self.toggle_clip_button.clicked.connect(self.toggle_clip_side)
        self.toggle_clip_button.setEnabled(False)

        self.remove_clip_button = QPushButton("Remove Clip")
        self.remove_clip_button.clicked.connect(self.remove_clip)
        self.remove_clip_button.setEnabled(False)

        analysis_layout.addWidget(analysis_label)
        analysis_layout.addWidget(self.segment_button)
        analysis_layout.addWidget(self.clip_plane_button)
        analysis_layout.addWidget(self.toggle_clip_button)
        analysis_layout.addWidget(self.remove_clip_button)
        
        lighting_group = self.setup_lighting_controls()
        
        control_layout.addWidget(load_section)
        control_layout.addWidget(QHLine())
        control_layout.addWidget(feature_section)
        control_layout.addWidget(QHLine())
        control_layout.addWidget(analysis_section)
        control_layout.addWidget(QHLine())
        control_layout.addWidget(gesture_section)
        control_layout.addWidget(QHLine())
        control_layout.addWidget(lighting_group)
        control_layout.addStretch()

        main_layout.addWidget(self.vtk_widget, 1)
        main_layout.addWidget(control_panel)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def setup_lighting_controls(self):
        lighting_group = QGroupBox("Lighting Controls")
        lighting_layout = QVBoxLayout()
        
        ambient_layout = QHBoxLayout()
        ambient_label = QLabel("Ambient:")
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setMinimum(0)
        self.ambient_slider.setMaximum(100)
        self.ambient_slider.setValue(30)
        self.ambient_slider.setTickPosition(QSlider.TicksBelow)
        self.ambient_slider.setTickInterval(10)
        self.ambient_slider.valueChanged.connect(self.update_ambient_light)
        ambient_layout.addWidget(ambient_label)
        ambient_layout.addWidget(self.ambient_slider)
        
        diffuse_layout = QHBoxLayout()
        diffuse_label = QLabel("Diffuse:")
        self.diffuse_slider = QSlider(Qt.Horizontal)
        self.diffuse_slider.setMinimum(0)
        self.diffuse_slider.setMaximum(100)
        self.diffuse_slider.setValue(100)
        self.diffuse_slider.setTickPosition(QSlider.TicksBelow)
        self.diffuse_slider.setTickInterval(10)
        self.diffuse_slider.valueChanged.connect(self.update_diffuse_light)
        diffuse_layout.addWidget(diffuse_label)
        diffuse_layout.addWidget(self.diffuse_slider)
        
        lighting_layout.addLayout(ambient_layout)
        lighting_layout.addLayout(diffuse_layout)
        lighting_group.setLayout(lighting_layout)
        
        return lighting_group
    
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
    
    def load_model(self, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Open 3D Model", "", "3D Models (*.stl *.obj)")
            if not filename:
                return False

        try:
            self.statusBar().showMessage(f"Loading model from {filename}...")
            
            if self.visualizer and hasattr(self.visualizer, 'renderer'):
                current_actors = self.visualizer.renderer.GetActors()
                current_actors.InitTraversal()
                actor = current_actors.GetNextActor()
                while actor:
                    self.visualizer.renderer.RemoveActor(actor)
                    actor = current_actors.GetNextActor()
            
            self.model_loader = ModelLoader()
            actor = self.model_loader.load_model(filename)
            
            if actor:
                self.visualizer.add_actor(actor)
                
                self.model_loaded = True
                
                self.extract_button.setEnabled(True)
                self.segment_button.setEnabled(True)
                self.clip_plane_button.setEnabled(True)
                
                self.ambient_slider.setEnabled(True)
                self.diffuse_slider.setEnabled(True)
                
                if self.visualizer:
                    ambient_value = self.visualizer.get_light_intensity('ambient')
                    diffuse_value = self.visualizer.get_light_intensity('diffuse')
                    
                    self.ambient_slider.setValue(int(ambient_value * 100))
                    self.diffuse_slider.setValue(int(diffuse_value * 100))
                
                self.vtk_widget.GetRenderWindow().Render()
                self.statusBar().showMessage(f"Model loaded successfully: {os.path.basename(filename)}")
                
                return True
            else:
                self.model_loaded = False
                
                self.extract_button.setEnabled(False)
                self.segment_button.setEnabled(False)
                self.clip_plane_button.setEnabled(False)
                
                self.ambient_slider.setEnabled(False)
                self.diffuse_slider.setEnabled(False)
                
                self.statusBar().showMessage("Failed to load model")
                
                return False
                
        except Exception as e:
            self.statusBar().showMessage(f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")
            return False
    
    def extract_features(self):
        if not self.model_loaded:
            print("No model loaded to extract features from.")
            return
        if self.model_loader.actor is None or self.model_loader.polydata is None:
             print("Model actor or polydata is missing.")
             return

        try:
            method = self.method_combo.currentText()
            points = self.model_loader.get_points()
            polydata = self.model_loader.polydata

            if points is None:
                 print("Error: Model points data is not available.")
                 return

            labels = None
            print(f"Extracting features using: {method}")

            if method == "Cluster by Position":
                n_clusters = self.cluster_spin.value()
                labels = self.feature_extractor.cluster_by_position(points, n_clusters)
            elif method == "Cluster by Curvature":
                n_clusters = self.cluster_spin.value()
                labels = self.feature_extractor.cluster_by_curvature(points, polydata, n_clusters)
            elif method == "Find Edges":
                threshold = self.thresh_slider.value() / 100.0
                labels = self.feature_extractor.find_edges(points, polydata, threshold)

            if labels is None:
                print("Feature extraction did not produce labels.")
                return

            print(f"Generated {len(np.unique(labels))} unique labels.")

            updated_actor = self.model_loader.update_colors(labels)
            if updated_actor is None:
                 print("Failed to update model colors.")
                 return

            mapper = updated_actor.GetMapper()
            if mapper:
                 print("Configuring mapper for feature colors...")
                 
                 mapper.SetInputData(self.model_loader.polydata)
                 
                 mapper.SetScalarModeToUsePointData()
                 
                 mapper.SelectColorArray("Colors")
                 
                 mapper.SetColorModeToDirectScalars()
                 
                 mapper.UseLookupTableScalarRangeOff()
                 
                 mapper.SetScalarVisibility(True)
                 
                 if self.clipper:
                      self.remove_clip()
            else:
                 print("Warning: Could not get mapper to apply feature colors.")

            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            print(f"Error extracting features: {e}")

    def start_gesture_control(self):
        if self.camera_available and not self.gesture_timer.isActive():
            print("Starting gesture control...")
            self.gesture_timer.start(30)
            self.start_gesture_button.setEnabled(False)
            self.stop_gesture_button.setEnabled(True)

    def stop_gesture_control(self):
        if self.gesture_timer.isActive():
            print("Stopping gesture control.")
            self.gesture_timer.stop()
            if self.camera_available:
                self.start_gesture_button.setEnabled(True)
                self.stop_gesture_button.setEnabled(False)
            cv2.destroyWindow('MediaPipe Hands')
            self.gesture_state = None
            self.previous_pinch_distance = None
            self.previous_wrist_position = None

    def process_gesture_frame(self):
        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        debug_frame = frame.copy()
        current_wrist_position = None
        current_pinch_distance = None
        is_pinch_possible = False
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            image_height, image_width, _ = frame.shape

            wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
            thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

            current_wrist_position = (wrist.x * image_width, wrist.y * image_height)

            dx = thumb_tip.x - index_tip.x
            dy = thumb_tip.y - index_tip.y
            current_pinch_distance = np.sqrt(dx*dx + dy*dy)
            pinch_threshold = 0.1
            is_pinching_now = current_pinch_distance < pinch_threshold

            new_state = None
            if is_pinching_now:
                new_state = 'pinching'
            elif hand_detected:
                new_state = 'swiping'

            if new_state != 'pinching':
                self.previous_pinch_distance = None

            self.gesture_state = new_state

            camera = self.visualizer.get_renderer().GetActiveCamera()
            needs_render = False

            if self.gesture_state == 'pinching' and current_pinch_distance is not None:
                stable_pinch_dist = max(current_pinch_distance, 0.01)

                if self.previous_pinch_distance is not None:
                    delta_distance = stable_pinch_dist - self.previous_pinch_distance
                    zoom_factor = 1.0 - delta_distance / self.zoom_sensitivity
                    
                    zoom_factor = max(0.8, min(1.2, zoom_factor))
                    camera.Dolly(zoom_factor)
                    
                    self.visualizer.fix_clipping_range()
                    
                    needs_render = True
                self.previous_pinch_distance = stable_pinch_dist

            if self.gesture_state == 'swiping' and current_wrist_position is not None:
                if self.previous_wrist_position is not None:
                    delta_x = current_wrist_position[0] - self.previous_wrist_position[0]
                    delta_y = current_wrist_position[1] - self.previous_wrist_position[1]
                    velocity = np.sqrt(delta_x*delta_x + delta_y*delta_y)

                    if velocity > self.swipe_velocity_threshold:
                        camera = self.visualizer.get_renderer().GetActiveCamera()
                        camera.Azimuth(delta_x * self.rotation_sensitivity)
                        
                        delta_elevation = delta_y * self.rotation_sensitivity
                        self.visualizer.safe_elevation(delta_elevation)
                        
                        needs_render = True

            if hand_detected:
                 self.previous_wrist_position = current_wrist_position
            else:
                 self.gesture_state = None
                 self.previous_pinch_distance = None
                 self.previous_wrist_position = None

            if needs_render and self.vtk_widget:
                self.vtk_widget.GetRenderWindow().Render()

            self.mp_drawing.draw_landmarks(
                debug_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS)
                
            camera_elevation = round(self.visualizer.get_camera_elevation(), 1)
            cv2.putText(debug_frame, f"State: {self.gesture_state}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(debug_frame, f"Elevation: {camera_elevation}°", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            if self.gesture_state == 'pinching' and current_pinch_distance is not None:
                cv2.putText(debug_frame, f"Pinch dist: {current_pinch_distance:.3f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            if self.gesture_state is not None:
                self.gesture_state = None
                self.previous_pinch_distance = None
                self.previous_wrist_position = None

        cv2.imshow('MediaPipe Hands', debug_frame)
        if cv2.waitKey(5) & 0xFF == 27:
             self.stop_gesture_control()

    def update_zoom_sensitivity(self):
        self.zoom_sensitivity = self.zoom_slider.value() / 100.0

    def update_rotation_sensitivity(self):
        self.rotation_sensitivity = self.rotation_slider.value() / 100.0

    def segment_by_connectivity(self):
        if not self.model_loaded or self.model_loader.polydata is None:
            print("No model loaded or polydata missing.")
            return

        print("Segmenting model by connectivity...")
        try:
            polydata_to_segment = self.model_loader.polydata

            connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
            connectivity_filter.SetInputData(polydata_to_segment)
            connectivity_filter.SetExtractionModeToAllRegions()
            connectivity_filter.ColorRegionsOn()
            connectivity_filter.Update()

            segmented_polydata = connectivity_filter.GetOutput()
            number_of_regions = connectivity_filter.GetNumberOfExtractedRegions()
            print(f"Found {number_of_regions} connected regions.")

            if number_of_regions <= 1:
                print("Model is a single connected component. No segmentation applied.")
                self.reset_model_appearance()
                return

            mapper = self.model_loader.actor.GetMapper()
            if not mapper:
                print("Error: Could not get mapper from actor.")
                return

            mapper.SetInputData(segmented_polydata)

            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(number_of_regions)
            lut.SetTableRange(0, number_of_regions - 1)
            lut.SetHueRange(0.0, 1.0)
            lut.SetSaturationRange(0.8, 1.0)
            lut.SetValueRange(0.8, 1.0)
            lut.Build()

            region_id_location = ""
            if segmented_polydata.GetPointData().HasArray("RegionId"):
                region_id_location = "point"
                mapper.SetScalarModeToUsePointData()
            elif segmented_polydata.GetCellData().HasArray("RegionId"):
                region_id_location = "cell"
                mapper.SetScalarModeToUseCellData()
            else:
                print("Error: RegionId array not found in data")
                return

            print(f"Using RegionId from {region_id_location} data")

            mapper.SetLookupTable(lut)
            mapper.SelectColorArray("RegionId")
            mapper.SetScalarRange(0, number_of_regions - 1)
            mapper.SetScalarVisibility(True)

            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
            
    def reset_model_appearance(self):
        if not self.model_loaded or self.model_loader.actor is None:
             return

        mapper = self.model_loader.actor.GetMapper()
        if mapper:
             mapper.SetScalarVisibility(False)
             if hasattr(self.model_loader, 'normals_output') and self.model_loader.normals_output:
                 mapper.SetInputData(self.model_loader.normals_output)
             elif self.model_loader.polydata:
                 mapper.SetInputData(self.model_loader.polydata)

        self.vtk_widget.GetRenderWindow().Render()

    def apply_clip_plane(self):
        if not self.model_loaded or self.model_loader.actor is None:
            print("No model loaded.")
            return

        source_data = self.model_loader.normals_output
        if source_data is None:
             print("Source data for clipping not available.")
             return

        print("Applying clip plane...")
        try:
            center = source_data.GetCenter()
            self.clip_plane = vtk.vtkPlane()
            self.clip_plane.SetOrigin(center)
            self.clip_plane.SetNormal(1, 0, 0)

            self.clipper = vtk.vtkClipPolyData()
            self.clipper.SetInputData(source_data)
            self.clipper.SetClipFunction(self.clip_plane)
            self.clipper.SetValue(0)
            self.clipper.GenerateClippedOutputOff()
            self.clipper.SetInsideOut(False)
            self.clipper.Update()

            mapper = self.model_loader.actor.GetMapper()
            if mapper:
                mapper.SetInputConnection(self.clipper.GetOutputPort())
                mapper.SetScalarVisibility(False)
                self.toggle_clip_button.setEnabled(True)
                self.remove_clip_button.setEnabled(True)
                self.clip_plane_button.setEnabled(False)
                self.vtk_widget.GetRenderWindow().Render()
            else:
                print("Error: Could not get mapper.")

        except Exception as e:
            print(f"Error applying clip plane: {e}")
            self.remove_clip()

    def toggle_clip_side(self):
        if self.clipper:
            current_inside_out = self.clipper.GetInsideOut()
            self.clipper.SetInsideOut(not current_inside_out)
            self.clipper.Update()
            self.vtk_widget.GetRenderWindow().Render()
            print(f"Toggled clip side. InsideOut: {not current_inside_out}")
        else:
            print("No active clipper to toggle.")

    def remove_clip(self):
        if not self.model_loaded or self.model_loader.actor is None:
            return

        print("Removing clip.")
        mapper = self.model_loader.actor.GetMapper()
        if mapper:
            if self.model_loader.normals_output:
                mapper.SetInputData(self.model_loader.normals_output)
            elif self.model_loader.polydata:
                 mapper.SetInputData(self.model_loader.polydata)
            else:
                 mapper.SetInputData(None)

            self.toggle_clip_button.setEnabled(False)
            self.remove_clip_button.setEnabled(False)
            self.clip_plane_button.setEnabled(True)
            self.clipper = None
            self.clip_plane = None

            self.vtk_widget.GetRenderWindow().Render()

    def update_ambient_light(self):
        if self.visualizer:
            value = self.ambient_slider.value() / 100.0
            self.visualizer.set_light_intensity('ambient', value)
            self.vtk_widget.GetRenderWindow().Render()

    def update_diffuse_light(self):
        if self.visualizer:
            value = self.diffuse_slider.value() / 100.0
            self.visualizer.set_light_intensity('diffuse', value)
            self.vtk_widget.GetRenderWindow().Render()

    def update_specular_light(self):
        if self.model_loaded:
            value = self.specular_slider.value() / 100.0
            self.model_loader.set_actor_specular_intensity(value)
            self.vtk_widget.GetRenderWindow().Render()

    def update_lighting_sliders(self):
        if self.model_loaded:
            self.ambient_slider.setValue(int(self.visualizer.get_light_intensity('ambient') * 100))
            self.diffuse_slider.setValue(int(self.visualizer.get_light_intensity('diffuse') * 100))
            self.specular_slider.setValue(int(self.model_loader.get_actor_specular_intensity() * 100))

    def closeEvent(self, event):
        print("Closing application...")
        self.gesture_timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().closeEvent(event)

class QHLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setLineWidth(1)
        self.setMidLineWidth(0)
        self.setContentsMargins(0, 0, 0, 0)
        
        palette = self.palette()
        palette.setColor(palette.WindowText, Qt.gray)
        self.setPalette(palette) 