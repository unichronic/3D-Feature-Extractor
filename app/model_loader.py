import vtk
import numpy as np
import os

class ModelLoader:
    def __init__(self):
        self.points = None
        self.cells = None
        self.polydata = None
        self.actor = None
        self.file_path = None
    
    def load_model(self, file_path):
        """
        Load a 3D model from the given file path.
        Supports STL and OBJ formats.
        """
        self.file_path = file_path
        
        # Get the file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Create appropriate reader based on file extension
        if ext == '.stl':
            reader = vtk.vtkSTLReader()
        elif ext == '.obj':
            reader = vtk.vtkOBJReader()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        reader.SetFileName(file_path)
        reader.Update()
        
        self.polydata = reader.GetOutput()
        
        # Extract point data
        points = self.polydata.GetPoints()
        self.points = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.polydata)
        
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        
        return self.actor
    
    def get_points(self):
        """Return the points as a numpy array"""
        return self.points
    
    def update_colors(self, labels):
        """
        Update the colors of the model based on the feature labels
        """
        if self.polydata is None:
            return
        
        # Create color map
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # Set colors based on labels
        unique_labels = np.unique(labels)
        color_map = {}
        
        # Generate distinct colors for each label
        import matplotlib.cm as cm
        colormap = cm.get_cmap('tab10')
        
        for i, label in enumerate(unique_labels):
            color = colormap(i % 10)[:3]  # Get RGB from colormap
            color_map[label] = [int(255*c) for c in color]
        
        # Apply colors to points
        for i, label in enumerate(labels):
            colors.InsertNextTuple3(*color_map[label])
        
        self.polydata.GetPointData().SetScalars(colors)
        return self.actor 