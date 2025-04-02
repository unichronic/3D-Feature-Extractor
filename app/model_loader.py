import vtk
import numpy as np

class ModelLoader:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.actor = None
        self.points = None
        self.normals = None
        self.point_count = 0
        self.feature_labels = None
        self.file_path = None
        
        self.reset_actor_properties()
        
    def reset_actor_properties(self):
        self.actor_ambient = 0.3
        self.actor_diffuse = 0.7
        self.actor_specular = 0.2
        self.actor_specular_power = 10
        
    def load_model(self, file_path):
        self.file_path = file_path
        
        lower_file_path = file_path.lower()
        reader = None
        
        if lower_file_path.endswith('.stl'):
            reader = vtk.vtkSTLReader()
        elif lower_file_path.endswith('.ply'):
            reader = vtk.vtkPLYReader()
        elif lower_file_path.endswith('.obj'):
            reader = vtk.vtkOBJReader()
        else:
            return False
        
        reader.SetFileName(file_path)
        reader.Update()
        
        if not reader.GetOutput().GetNumberOfPoints():
            return False
        
        if self.actor:
            self.visualizer.get_renderer().RemoveActor(self.actor)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        
        self.actor.GetProperty().SetAmbient(self.actor_ambient)
        self.actor.GetProperty().SetDiffuse(self.actor_diffuse)
        self.actor.GetProperty().SetSpecular(self.actor_specular)
        self.actor.GetProperty().SetSpecularPower(self.actor_specular_power)
        
        self.points = self.get_points()
        self.normals = self.get_normals()
        self.point_count = len(self.points)
        self.feature_labels = None
        
        self.visualizer.add_actor(self.actor)
        
        return True
    
    def get_points(self):
        if not self.actor:
            return None
        
        poly_data = self.actor.GetMapper().GetInput()
        points_vtk = poly_data.GetPoints()
        
        point_count = points_vtk.GetNumberOfPoints()
        points = np.zeros((point_count, 3))
        
        for i in range(point_count):
            points[i] = points_vtk.GetPoint(i)
        
        return points
    
    def get_normals(self):
        if not self.actor:
            return None
        
        poly_data = self.actor.GetMapper().GetInput()
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(poly_data)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.Update()
        
        poly_data_with_normals = normals_filter.GetOutput()
        normals_vtk = poly_data_with_normals.GetPointData().GetNormals()
        
        normal_count = normals_vtk.GetNumberOfTuples()
        normals = np.zeros((normal_count, 3))
        
        for i in range(normal_count):
            normals[i] = normals_vtk.GetTuple(i)
        
        return normals
    
    def update_colors(self, feature_labels, color_map=None, smooth_transition=False):
        """Updates the colors of the model based on feature labels."""
        if not self.actor or feature_labels is None:
            return False
            
        self.feature_labels = feature_labels
            
        if color_map is None:
            color_map = self.get_default_color_map()
            
        poly_data = self.actor.GetMapper().GetInput()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        for i in range(len(feature_labels)):
            label = feature_labels[i]
            if label >= 0 and label < len(color_map):
                r, g, b = color_map[label]
                r_int, g_int, b_int = int(r*255), int(g*255), int(b*255)
                colors.InsertNextTuple3(r_int, g_int, b_int)
            else:
                colors.InsertNextTuple3(100, 100, 100) 
                
        poly_data.GetPointData().SetScalars(colors)
        self.actor.GetProperty().SetInterpolationToGouraud()
        
        return True
        
    def get_default_color_map(self):
        return [
            (0.9, 0.1, 0.1),  
            (0.1, 0.9, 0.1),  
            (0.1, 0.1, 0.9),  
            (0.9, 0.9, 0.1),  
            (0.9, 0.1, 0.9),  
            (0.1, 0.9, 0.9),  
            (0.7, 0.3, 0.3),  
            (0.3, 0.7, 0.3),  
            (0.3, 0.3, 0.7),  
            (0.7, 0.7, 0.3),  
            (0.7, 0.3, 0.7),  
            (0.3, 0.7, 0.7),  
            (0.5, 0.25, 0.25), 
            (0.25, 0.5, 0.25), 
            (0.25, 0.25, 0.5)  
        ]
        
    def reset_colors(self):
        """Resets the model to its original colors."""
        if not self.actor:
            return
            
        self.feature_labels = None
        
        poly_data = self.actor.GetMapper().GetInput()
        if poly_data.GetPointData().GetScalars():
            poly_data.GetPointData().SetScalars(None)
            
        self.actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        
    def set_actor_property(self, property_name, value):
        """Sets a material property on the actor."""
        if not self.actor:
            return False
            
        if property_name == "ambient":
            self.actor_ambient = value
            self.actor.GetProperty().SetAmbient(value)
        elif property_name == "diffuse":
            self.actor_diffuse = value
            self.actor.GetProperty().SetDiffuse(value)
        elif property_name == "specular":
            self.actor_specular = value
            self.actor.GetProperty().SetSpecular(value)
        elif property_name == "specular_power":
            self.actor_specular_power = value
            self.actor.GetProperty().SetSpecularPower(value)
        else:
            return False
            
        return True
        
    def get_actor_property(self, property_name):
        """Gets a material property from the actor."""
        if not self.actor:
            return 0.0
            
        if property_name == "ambient":
            return self.actor_ambient
        elif property_name == "diffuse":
            return self.actor_diffuse
        elif property_name == "specular":
            return self.actor_specular
        elif property_name == "specular_power":
            return self.actor_specular_power
        else:
            return 0.0