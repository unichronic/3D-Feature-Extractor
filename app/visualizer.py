import vtk
import math

class Visualizer:
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        self.model_loaded = False
        self.prev_elevation = 0.0

        self.renderer.SetUseFXAA(True) 
        self.light = vtk.vtkLight()
        self.light.SetLightTypeToHeadlight() 
        self.light.SetIntensity(1.0) 
        
        self.renderer.SetAmbient(0.3, 0.3, 0.3) 
        self.light.SetDiffuseColor(1.0, 1.0, 1.0) 
        self.light.SetSpecularColor(1.0, 1.0, 1.0) 
        self.renderer.AddLight(self.light)

    def get_renderer(self):
        return self.renderer

    def add_actor(self, actor):
        self.renderer.AddActor(actor)
        self.model_loaded = True
        self.configure_camera_for_model()

    def remove_all_actors(self):
        self.renderer.RemoveAllViewProps() 
        self.model_loaded = False

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.fix_clipping_range()
        self.prev_elevation = 0.0

    def get_camera_elevation(self):
        camera = self.renderer.GetActiveCamera()
        direction = camera.GetDirectionOfProjection()
        view_up = camera.GetViewUp()
        elevation = math.degrees(math.asin(direction[1]))
        return elevation
        
    def safe_elevation(self, delta):
        """Apply elevation changes while preventing flipping issues"""
        if not self.renderer:
            return
        
        camera = self.renderer.GetActiveCamera()
        if not camera:
            return
            
        current_elevation = self.get_camera_elevation()
        
        danger_zone_start = 5.0  
        danger_zone_end = 9.0    
        
        new_elevation = current_elevation + delta
        
        reduction_factor = 0.25  
        if (current_elevation < danger_zone_start and new_elevation >= danger_zone_start) or \
           (current_elevation >= danger_zone_start and current_elevation <= danger_zone_end):
            delta *= reduction_factor
            
        elif (current_elevation > -danger_zone_start and new_elevation <= -danger_zone_start) or \
             (current_elevation <= -danger_zone_start and current_elevation >= -danger_zone_end):
            delta *= reduction_factor
            
        if ((current_elevation < danger_zone_start and new_elevation > danger_zone_end) or
            (current_elevation > -danger_zone_start and new_elevation < -danger_zone_end)):
            if delta > 0:
                camera.Elevation(danger_zone_end - current_elevation + 1)
            else:
                camera.Elevation(-danger_zone_end - current_elevation - 1)
        else:
            camera.Elevation(delta)
            
        if abs(current_elevation) < 1.0:  
            camera.SetViewUp(0, 1, 0)  
            
        self.fix_clipping_range()
        
        self.prev_elevation = current_elevation
    
    def fix_clipping_range(self):
        """Adjust near and far clipping planes to prevent cutoff issues"""
        if not self.model_loaded:
            return
            
        camera = self.renderer.GetActiveCamera()
        
        bounds = [0, 0, 0, 0, 0, 0]
        self.renderer.ComputeVisiblePropBounds(bounds)
        
        diagonal = math.sqrt((bounds[1]-bounds[0])**2 + 
                             (bounds[3]-bounds[2])**2 + 
                             (bounds[5]-bounds[4])**2)
        
        camera_pos = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        
        distance = math.sqrt(sum((camera_pos[i] - focal_point[i])**2 for i in range(3)))
        
        near_clip = distance - diagonal * 0.75
        far_clip = distance + diagonal * 2.0
        
        near_clip = max(0.01, near_clip)
        
        camera.SetClippingRange(near_clip, far_clip)
    
    def configure_camera_for_model(self):
        """Configure camera settings optimally for the loaded model"""
        if not self.model_loaded:
            return
            
        self.renderer.ResetCamera()
        
        camera = self.renderer.GetActiveCamera()
        
        camera.SetViewUp(0, 1, 0)  
        
        camera.SetViewAngle(30.0)
        
        self.fix_clipping_range()
        
        self.prev_elevation = 0.0

    def set_light_intensity(self, light_type, intensity):
        """Sets the intensity for ambient or diffuse light."""
        intensity = max(0.0, min(1.0, intensity))
        
        if light_type == 'ambient':
            self.renderer.SetAmbient(intensity, intensity, intensity)
            actors = self.renderer.GetActors()
            actors.InitTraversal()
            actor = actors.GetNextActor()
            while actor:
                actor.GetProperty().SetAmbient(intensity)
                actor = actors.GetNextActor()
        elif light_type == 'diffuse':
            self.light.SetIntensity(intensity)
        else:
            print(f"Unknown light type in Visualizer: {light_type}")

    def get_light_intensity(self, light_type):
        """Gets the current intensity for ambient or diffuse light."""
        if light_type == 'ambient':
            return self.renderer.GetAmbient()[0]
        elif light_type == 'diffuse':
            return self.light.GetIntensity()
        else:
            print(f"Unknown light type in Visualizer: {light_type}")
            return 0.5 
