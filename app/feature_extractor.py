import numpy as np
import vtk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    def __init__(self):
        self.points = None
        self.features = None
        self.labels = None

    def extract_curvature(self, polydata):
        """
        Extract curvature features from the model
        """
        curvature = vtk.vtkCurvatures()
        curvature.SetInputData(polydata)
        curvature.SetCurvatureTypeToMean()
        curvature.Update()
        
        curv_data = curvature.GetOutput().GetPointData().GetScalars()
        
        curv_array = np.zeros(curv_data.GetNumberOfTuples())
        for i in range(curv_data.GetNumberOfTuples()):
            curv_array[i] = curv_data.GetTuple1(i)
            
        return curv_array
    
    def cluster_by_position(self, points, n_clusters=5):
        """
        Cluster points by their position using KMeans
        """
        scaler = StandardScaler()
        points_std = scaler.fit_transform(points)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(points_std)
        
        return self.labels

    def cluster_by_curvature(self, points, polydata, n_clusters=5):
        """
        Cluster points by their curvature using KMeans
        """
        curvature = self.extract_curvature(polydata)
        
        features = np.column_stack((points, curvature.reshape(-1, 1)))
        
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(features_std)
        
        return self.labels

    def find_edges(self, points, polydata, edge_threshold=0.7):
        """
        Find edges in the model based on curvature
        """
        curvature = self.extract_curvature(polydata)

        min_curv = np.min(curvature)
        max_curv = np.max(curvature)
        delta_curv = max_curv - min_curv

        if delta_curv < 1e-6:
            print("Warning: Curvature is effectively constant across the model. Cannot perform edge detection.")
            self.labels = np.zeros(points.shape[0], dtype=int)
        else:
            curvature_norm = (curvature - min_curv) / delta_curv
            self.labels = np.where(curvature_norm > edge_threshold, 1, 0)
            print(f"Edge detection complete. Found {np.sum(self.labels)} edge points.")

        return self.labels 

    def simulate_deformation(self, original_points, time_step):
        """
        Applies a simple, predefined deformation based on a time step.
        This is a placeholder for a real ML prediction for MVP purposes.
        Example: Simple scaling/pulsing along the Z-axis.
        """
        if original_points is None:
            return None

        deformed_points = original_points.copy()
        scale_factor = 1.0 + 0.1 * np.sin(time_step * 0.1)
        
        deformed_points[:, 2] = original_points[:, 2] * scale_factor 
        
        return deformed_points 