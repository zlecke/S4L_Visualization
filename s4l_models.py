import numpy as np
import vtk

from mayavi.filters.cut_plane import CutPlane
from mayavi.filters.data_set_clipper import DataSetClipper
from mayavi.modules.surface import Surface
from mayavi.sources.api import VTKFileReader, ArraySource
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

from tvtk.pyface.scene_editor import SceneEditor

from traits.api import (
    Any,
    Instance,
    HasTraits,
    Array,
    observe,
    File,
    Bool
)
from traitsui.api import View, Item, Group

vtk.vtkObject.GlobalWarningDisplayOff()

scale_factor = 0.001
swap_xy = True


def arg_find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


class MayaviPlot(HasTraits):
    scene = Instance(MlabSceneModel, ())

    csf_model = File()
    csf_model_reader = Instance(VTKFileReader)
    csf_surface = Instance(Surface)

    rotation_filter = Any()

    origin = Array(value=np.array([0, 0, 0]))
    normal = Array(value=np.array([0, 0, 1]))

    data_set_clipper = Instance(DataSetClipper)

    show_full_model = Bool(False)
    full_csf_surface = Instance(Surface)

    grid_x = Array(value=np.array([0]))
    grid_y = Array(value=np.array([0]))
    grid_z = Array(value=np.array([0]))

    grid_data = Array(value=np.array([[[0]]]))

    src = Instance(ArraySource)
    cut = Instance(CutPlane)
    surf = Instance(Surface)

    line_low_point = Array(value=np.array([0, 0, 0]))
    line_high_point = Array(value=np.array([0, 0, 1]))

    line = Instance(Surface)

    log_scale = Bool(True)

    @observe('line_low_point', post_init=True)
    @observe('line_high_point', post_init=True)
    def draw_line(self, event):
        x = [self.line_low_point[0], self.line_high_point[0]]
        y = [self.line_low_point[1], self.line_high_point[1]]
        z = [self.line_low_point[2], self.line_high_point[2]]

        if not hasattr(self.line, 'mlab_source'):
            self.line = self.scene.mlab.plot3d(x, y, z, tube_radius=0.2, color=(1, 0, 0),
                                               figure=self.scene.mayavi_scene)
        else:
            self.line.mlab_source.x = x
            self.line.mlab_source.y = y
            self.line.mlab_source.z = z

        self.scene.mlab.draw()

    @observe('csf_model', post_init=True)
    def _model_updated(self, event):
        self.csf_model_reader.file_path = event.new
        self.scene.mlab.draw()

    @observe('show_full_model', post_init=True)
    def toggle_full_model(self, event):
        self.full_csf_surface.visible = event.new
        self.csf_surface.visible = event.old
        self.scene.mlab.draw()

    @observe('log_scale', post_init=True)
    def _update_log_scale(self, event):
        if event.new:
            self.surf.parent.scalar_lut_manager.lut.scale = 'log10'
        else:
            self.surf.parent.scalar_lut_manager.lut.scale = 'linear'

        self.scene.mlab.draw()

    @observe('origin', post_init=True)
    def _update_origin(self, event):
        if hasattr(self.data_set_clipper, 'widget'):
            if self.data_set_clipper.widget.widget.enabled:
                self.cut.filters[0].widget.enabled = False
                self.data_set_clipper.widget.widget.enabled = False

                if self.log_scale:
                    self.surf.parent.scalar_lut_manager.lut.scale = 'log10'
                else:
                    self.surf.parent.scalar_lut_manager.lut.scale = 'linear'

                self.surf.parent.scalar_lut_manager.show_legend = True
                self.surf.parent.scalar_lut_manager.use_default_name = False
                self.surf.parent.scalar_lut_manager.data_name = 'J (A/m^2)'
                self.surf.parent.scalar_lut_manager.shadow = True
                self.surf.parent.scalar_lut_manager.use_default_range = False
                self.surf.parent.scalar_lut_manager.data_range = np.array(
                        [np.nanmin(self.grid_data), np.nanmax(self.grid_data)])

            self.data_set_clipper.widget.widget.origin = self.origin
            self.cut.filters[0].widget.origin = self.origin

            self.scene.mlab.draw()

    @observe('normal', post_init=True)
    def _update_normal(self, event):
        if hasattr(self.data_set_clipper, 'widget'):
            self.data_set_clipper.widget.widget.normal = self.normal
            self.cut.filters[0].widget.normal = self.normal

            self.scene.mlab.draw()

    def _create_surfaces(self):
        self.csf_model_reader = self.scene.engine.open(self.csf_model)

        self.data_set_clipper = DataSetClipper()
        self.data_set_clipper.widget.widget_mode = 'ImplicitPlane'
        self.data_set_clipper.widget.widget.normal = self.normal
        self.data_set_clipper.widget.widget.origin = self.origin
        self.data_set_clipper.widget.widget.enabled = False
        self.data_set_clipper.widget.widget.key_press_activation = False
        self.data_set_clipper.filter.inside_out = True

        self.scene.engine.add_filter(self.data_set_clipper, self.csf_model_reader)

        self.csf_surface = Surface()
        self.csf_surface.actor.property.opacity = 0.3
        self.csf_surface.actor.property.specular_color = (0.0, 0.0, 1.0)
        self.csf_surface.actor.property.specular = 1.0
        self.csf_surface.actor.actor.use_bounds = False

        self.scene.engine.add_filter(self.data_set_clipper, self.csf_surface)

        self.full_csf_surface = Surface()
        self.full_csf_surface.actor.property.opacity = 0.3
        self.full_csf_surface.actor.property.specular_color = (0.0, 0.0, 1.0)
        self.full_csf_surface.actor.property.specular = 1.0
        self.full_csf_surface.actor.actor.use_bounds = False
        self.full_csf_surface.visible = False

        self.scene.engine.add_filter(self.full_csf_surface, self.csf_model_reader)

        self.src = self.scene.mlab.pipeline.scalar_field(self.grid_x, self.grid_y, self.grid_z, self.grid_data)
        self.cut = self.scene.mlab.pipeline.cut_plane(self.src)
        self.cut.filters[0].widget.normal = self.normal
        self.cut.filters[0].widget.origin = self.origin
        self.cut.filters[0].widget.enabled = False
        self.surf = self.scene.mlab.pipeline.surface(self.cut, colormap='jet')
        self.surf.actor.actor.use_bounds = False
        self.surf.parent.scalar_lut_manager.lut.nan_color = np.array([0, 0, 0, 0])

        self.scene.mlab.draw()

        self.line_low_point = np.array([0, 0, self.grid_z[0, 0, :].min()])
        self.line_high_point = np.array([0, 0, self.grid_z[0, 0, :].max()])

    @observe('scene.activated')
    def _initialize_camera(self, info):
        self.scene.scene.camera.position = [195, 200, 200]
        self.scene.scene.camera.focal_point = [-5.0, -0.5, -0.3]
        self.scene.scene.camera.view_angle = 30.0
        self.scene.scene.camera.view_up = [0.0, 0.0, 1.0]
        self.scene.scene.camera.clipping_range = [215, 500]
        self.scene.scene.camera.compute_view_plane_normal()

    def default_traits_view(self):
        return View(
                Group(
                        Item('scene',
                             show_label=False,
                             editor=SceneEditor(scene_class=MayaviScene))
                )
        )
