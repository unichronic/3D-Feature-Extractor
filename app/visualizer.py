import vtk

class Visualizer:
    def __init__(self):
        # Create renderer ONLY
        self.renderer = vtk.vtkRenderer()

        # Removed render_window and interactor creation
        # Removed interactor style setting

        # Set background color
        self.renderer.SetBackground(0.2, 0.2, 0.2)

        # Initialize flags
        self.model_loaded = False

    def get_renderer(self):
        """Return the renderer instance"""
        return self.renderer

    def add_actor(self, actor):
        """Add an actor to the renderer"""
        self.renderer.AddActor(actor)
        self.model_loaded = True

    def remove_all_actors(self):
        """Remove all actors from the renderer"""
        self.renderer.RemoveAllViewProps() # Use RemoveAllViewProps for renderers
        self.model_loaded = False

    def reset_camera(self):
        """Reset the camera of the renderer"""
        self.renderer.ResetCamera()

    # Removed start() method - Qt event loop manages interaction
    # Removed update() method - GUI will trigger render on the widget
    # Removed get_render_window() and get_interactor() methods 