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
        # Create the curvature filter
        curvature = vtk.vtkCurvatures()
        curvature.SetInputData(polydata)
        curvature.SetCurvatureTypeToMean()
        curvature.Update()
        
        # Get the curvature data
        curv_data = curvature.GetOutput().GetPointData().GetScalars()
        
        # Convert to numpy array
        curv_array = np.zeros(curv_data.GetNumberOfTuples())
        for i in range(curv_data.GetNumberOfTuples()):
            curv_array[i] = curv_data.GetTuple1(i)
            
        return curv_array
    
    def cluster_by_position(self, points, n_clusters=5):
        """
        Cluster points by their position using KMeans
        """
        # Standardize the data
        scaler = StandardScaler()
        points_std = scaler.fit_transform(points)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(points_std)
        
        return self.labels

    def cluster_by_curvature(self, points, polydata, n_clusters=5):
        """
        Cluster points by their curvature using KMeans
        """
        # Extract curvature
        curvature = self.extract_curvature(polydata)
        
        # Combine position and curvature
        features = np.column_stack((points, curvature.reshape(-1, 1)))
        
        # Standardize the data
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(features_std)
        
        return self.labels

    def find_edges(self, points, polydata, edge_threshold=0.7):
        """
        Find edges in the model based on curvature
        """
        # Extract curvature
        curvature = self.extract_curvature(polydata)
        
        # Normalize curvature
        curvature_norm = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))
        
        # Classify points as edge or non-edge
        self.labels = np.where(curvature_norm > edge_threshold, 1, 0)
        
        return self.labels 