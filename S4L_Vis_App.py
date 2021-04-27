import matplotlib

# We want matplotlib to use a QT backend
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure

from traits.api import Any, Instance, Int, HasTraits, Enum, Str, observe, Button, File, Bool, ListStr
from traits.trait_numeric import Array

from traitsui.api import View, Item, Group, HSplit, Action, Menu, MenuBar, EnumEditor, Handler, ArrayEditor
from traitsui.qt4.extra.qt_view import QtView
import traitsui.menu as menu
from traitsui.undo import UndoItem

from tvtk.pyface.scene_editor import SceneEditor

from pyface.api import FileDialog, OK, GUI

from PySide2.QtWidgets import QFileDialog

from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.api import PipelineBase
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.sources.array_source import ArraySource
from mayavi.filters.cut_plane import CutPlane
from mayavi.modules.surface import Surface
from mayavi.filters.data_set_clipper import DataSetClipper
from mayavi.filters.transform_data import TransformData

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
import copy
import os

from mpl_figure_editor import MPLFigureEditor
from q_range_editor import QRangeEditor

from preferences import default_csf_model

import vtk
import pickle

vtk.vtkObject.GlobalWarningDisplayOff()

scale_factor = 0.001
swap_xy = True


class SVHandler(Handler):

    def close(self, info, is_ok):
        try:
            out_data = {
                'plane_type': info.object.plotter.plane_type,
                'low_point': info.object.plotter.line_low_point,
                'high_point': info.object.plotter.line_high_point,
                'origin': info.object.plotter.origin,
                'normal': info.object.plotter.normal,
                'csf_model': info.object.csf_model,
                'show_model': info.object.show_full_model,
                'log_scale': info.object.log_scale,
                'data_dir': info.object.data_dir,
                'coord': info.object.plotter.coord,
                # 'scene'     :info.object.scene,
                'masked_gr_x': info.object.masked_gr_x,
                'masked_gr_y': info.object.masked_gr_y,
                'masked_gr_z': info.object.masked_gr_z,
                'masked_grid_data': info.object.masked_grid_data,
                'coord_map': info.object.coord_map,
                'low': info.object.low,
                'high': info.object.high,
                'x_vals': info.object.x_vals,
                'y_vals': info.object.y_vals,
                'z_vals': info.object.z_vals,
                'dataArr': info.object.dataArr,
                'low_label': info.object.low_label,
                'high_label': info.object.high_label,

            }
        except AttributeError:
            pass
        else:
            with open("last_state.pickle", 'wb') as f:
                pickle.dump(out_data, f)
        finally:
            return super(SVHandler, self).close(info, is_ok)

    def _on_undo(self, info):
        super(SVHandler, self)._on_undo(info)

    def object_line_panel_low_point_changed(self, info):
        if info.initialized:
            info.ui.do_undoable(info.object.line_panel.update_line_figure, info)

    def object_plotter_high_point_changed(self, info):
        if info.initialized:
            info.ui.do_undoable(info.object.plotter.draw_line, info)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def arg_find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


class Plotter(HasTraits):
    figure = Instance(Figure, ())

    scene = Instance(MlabSceneModel)
    mplot = Instance(PipelineBase)

    plane_type = Enum('Normal to Z',
                      'Normal to X',
                      'Normal to Y',
                      'Arbitrary Plane')

    line_low_point = Array(value=np.array([0, 0, 0]))
    line_high_point = Array(value=np.array([0, 0, 1]))

    line = Instance(Surface)
    line_cross = Any()

    origin = Array(value=np.array([0, 0, 0]))
    normal = Array(value=np.array([0, 0, 1]))

    data_set_clipper = Instance(DataSetClipper)

    src = Instance(ArraySource)
    cut = Instance(CutPlane)
    surf = Instance(Surface)

    csf_model = File(value=default_csf_model)
    csf_model_reader = Any()
    csf_surface = Instance(Surface)

    rotation_filter = Any()

    show_full_model = Bool(False)
    full_csf_surface = Instance(Surface)

    grid_x = Array(value=np.array([0]))
    grid_y = Array(value=np.array([0]))
    grid_z = Array(value=np.array([0]))

    grid_data = Array(value=np.array([[[0]]]))

    mycmap = Any()
    norm = Any()
    pcm = Any()
    clb = Any()

    coord_map = Array(value=np.array([0]))

    coord = Int(0)
    low = Int(0)
    high = Int(0)

    log_scale = Bool(True)

    @observe('line_low_point', post_init=True)
    @observe('line_high_point', post_init=True)
    def draw_line(self, event):
        if self.line_cross is not None:
            ind = arg_find_nearest(np.linspace(self.line_low_point[2], self.line_high_point[2], 1000), self.coord)
            x = np.linspace(self.line_low_point[0], self.line_high_point[0], 1000)[ind]
            y = np.linspace(self.line_low_point[1], self.line_high_point[1], 1000)[ind]

            self.line_cross[0].set_data([x], [y])

            canvas = self.figure.canvas
            if canvas is not None:
                canvas.draw()
            # self.figure.canvas.draw()

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

    @observe('plane_type', post_init=True)
    def change_plane_type(self, event):
        if self.plane_type == 'Normal to X':
            self.normal = np.array([1, 0, 0])
        elif self.plane_type == 'Normal to Y':
            self.normal = np.array([0, 1, 0])
        elif self.plane_type == 'Normal to Z':
            self.normal = np.array([0, 0, 1])

    @observe('show_full_model', post_init=True)
    def toggle_full_model(self, event):
        self.csf_surface.visible = not self.show_full_model
        self.full_csf_surface.visible = self.show_full_model

        self.scene.mlab.draw()

    @observe('csf_model', post_init=True)
    def change_cord_model(self, event):
        self.csf_model_reader.initialize(self.csf_model)

    def create_plot(self):
        normal = self.normal

        self.scene.mlab.clf()

        self.csf_model_reader = self.scene.engine.open(self.csf_model)
        self.csf_surface = Surface()

        self.data_set_clipper = DataSetClipper()

        if swap_xy:
            rot_mat = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            self.rotation_filter = TransformData()
            self.scene.engine.add_filter(self.rotation_filter, self.csf_model_reader)
            self.rotation_filter.transform.matrix.__setstate__({'elements': list(rot_mat.flatten())})
            self.rotation_filter.widget.set_transform(self.rotation_filter.transform)
            self.rotation_filter.filter.update()
            self.rotation_filter.widget.enabled = False
            self.scene.engine.add_filter(self.data_set_clipper, self.rotation_filter)
        else:
            self.scene.engine.add_filter(self.data_set_clipper, self.csf_model_reader)

        self.data_set_clipper.widget.widget_mode = 'ImplicitPlane'
        self.data_set_clipper.widget.widget.normal = normal
        self.data_set_clipper.widget.widget.origin = self.origin
        self.data_set_clipper.widget.widget.enabled = False
        self.data_set_clipper.widget.widget.key_press_activation = False
        self.data_set_clipper.filter.inside_out = True
        self.csf_surface.actor.property.opacity = 0.3
        self.csf_surface.actor.property.specular_color = (0.0, 0.0, 1.0)
        self.csf_surface.actor.property.specular = 1.0
        self.csf_surface.actor.actor.use_bounds = False

        self.scene.engine.add_filter(self.csf_surface, self.data_set_clipper)

        self.full_csf_surface = Surface()
        self.full_csf_surface.actor.property.opacity = 0.3
        self.full_csf_surface.actor.property.specular_color = (0.0, 0.0, 1.0)
        self.full_csf_surface.actor.property.specular = 1.0
        self.full_csf_surface.actor.actor.use_bounds = False
        self.full_csf_surface.visible = False

        self.scene.engine.add_filter(self.full_csf_surface, self.csf_model_reader)

        self.src = self.scene.mlab.pipeline.scalar_field(self.grid_x, self.grid_y, self.grid_z, self.grid_data)
        self.cut = self.scene.mlab.pipeline.cut_plane(self.src)
        self.cut.filters[0].widget.normal = normal
        self.cut.filters[0].widget.origin = self.origin
        self.cut.filters[0].widget.enabled = False
        self.surf = self.scene.mlab.pipeline.surface(self.cut, colormap='jet')
        self.surf.actor.actor.use_bounds = False
        self.surf.parent.scalar_lut_manager.lut.nan_color = np.array([0, 0, 0, 0])

        self.scene.mlab.draw()

        self.line_low_point = np.array([0, 0, self.grid_z[0, 0, :].min()])
        self.line_high_point = np.array([0, 0, self.grid_z[0, 0, :].max()])

        dataset = self.cut.outputs[0].output
        datax, datay, dataz = dataset.points.to_array().T
        scalar_data = dataset.point_data.scalars.to_array()

        true_x = np.unique(np.floor(datax / scale_factor).astype(int)) * scale_factor
        true_y = np.unique(np.floor(datay / scale_factor).astype(int)) * scale_factor

        true_data = np.empty((true_y.size, true_x.size))
        true_data[:] = np.nan

        for i in range(scalar_data.size):
            true_data[arg_find_nearest(true_y, datay[i]), arg_find_nearest(true_x, datax[i])] = scalar_data[i]

        self.mycmap = copy.copy(cm.get_cmap('jet'))

        self.figure.clear()

        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(right=0.75, bottom=0.15)

        if self.log_scale:
            self.norm = LogNorm(vmin=np.nanmin(self.grid_data), vmax=np.nanmax(self.grid_data))
        else:
            self.norm = Normalize(vmin=np.nanmin(self.grid_data), vmax=np.nanmax(self.grid_data))

        self.pcm = ax.pcolormesh(true_x, true_y, true_data, shading='nearest', cmap=self.mycmap, norm=self.norm)

        self.line_cross = ax.plot([0], [0], 'rx')

        ax.set_ylim(bottom=self.grid_y[0, :, 0].min() - 2, top=self.grid_y[0, :, 0].max() + 2)
        ax.set_xlim(left=self.grid_x[:, 0, 0].min() - 2, right=self.grid_x[:, 0, 0].max() + 2)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

        ax.set_title('Current Density Magnitude')

        self.clb = self.figure.colorbar(cm.ScalarMappable(norm=self.norm, cmap=self.mycmap), ax=ax)
        self.clb.set_label('Current Density ($A/m^2$)', rotation=270, labelpad=15)

        self.figure.canvas.draw()

    @observe('log_scale', post_init=True)
    def change_scale(self, event):
        if self.log_scale:
            self.norm = LogNorm(vmin=np.nanmin(self.grid_data), vmax=np.nanmax(self.grid_data))
            self.surf.parent.scalar_lut_manager.lut.scale = 'log10'
        else:
            self.norm = Normalize(vmin=np.nanmin(self.grid_data), vmax=np.nanmax(self.grid_data))
            self.surf.parent.scalar_lut_manager.lut.scale = 'linear'

        self.clb.update_normal(cm.ScalarMappable(norm=self.norm, cmap=self.mycmap))

        self.scene.mlab.draw()
        self.update_plot(event)

    @observe('coord', post_init=True)
    def update_coord(self, event):
        if hasattr(self.data_set_clipper, 'widget'):
            if self.data_set_clipper.widget.widget.enabled:
                self.cut.filters[0].widget.enabled = False
                self.data_set_clipper.widget.widget.enabled = False
                self.rotation_filter.widget.enabled = False
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

            coord = self.coord_map[self.coord]

            if (self.plane_type == 'Normal to Z' or self.plane_type == 'Arbitrary Plane') and self.origin[2] != coord:
                self.origin = np.array([self.origin[0], self.origin[1], coord])
            elif self.plane_type == 'Normal to Y' and self.origin[1] != coord:
                self.origin = np.array([self.origin[0], coord, self.origin[2]])
            elif self.plane_type == 'Normal to X' and self.origin[0] != coord:
                self.origin = np.array([coord, self.origin[1], self.origin[2]])

            self.data_set_clipper.widget.widget.origin = self.origin

            self.cut.filters[0].widget.origin = self.origin

            ind = arg_find_nearest(np.linspace(self.line_low_point[2], self.line_high_point[2], 1000), coord)
            x = np.linspace(self.line_low_point[0], self.line_high_point[0], 1000)[ind]
            y = np.linspace(self.line_low_point[1], self.line_high_point[1], 1000)[ind]

            self.line_cross[0].set_data([x], [y])

        self.scene.mlab.draw()
        self.update_plot(event)

    def update_plot(self, event):
        if hasattr(self, 'pcm'):
            dataset = self.cut.outputs[0].output
            datax, datay, dataz = dataset.points.to_array().T
            scalar_data = dataset.point_data.scalars.to_array()

            true_x = np.unique(np.floor(datax / scale_factor).astype(int)) * scale_factor
            true_y = np.unique(np.floor(datay / scale_factor).astype(int)) * scale_factor

            true_data = np.empty((true_y.size, true_x.size))
            true_data[:] = np.nan

            for i in range(scalar_data.size):
                true_data[arg_find_nearest(true_y, datay[i]), arg_find_nearest(true_x, datax[i])] = scalar_data[i]

            axes = self.figure.axes[0]
            self.pcm.remove()
            self.pcm = axes.pcolormesh(true_x, true_y, true_data, shading='nearest', cmap=self.mycmap, norm=self.norm)
            canvas = self.figure.canvas
            if canvas is not None:
                canvas.draw()

    @observe('normal', post_init=True)
    def update_plane(self, event):
        self.data_set_clipper.widget.widget.normal = self.normal
        self.cut.filters[0].widget.normal = self.normal
        self.scene.mlab.draw()
        self.update_coord(event)

    def clippers_have_plane(self):
        return hasattr(self.data_set_clipper, 'widget')


class LinePanel(HasTraits):
    line_figure = Instance(Figure, ())

    low_point = Array(value=np.array([0, 0, -1]))
    high_point = Array(value=np.array([0, 0, 1]))

    interp_func = Instance(RegularGridInterpolator)

    grid_x = Array(value=np.array([[[0]]]))
    grid_y = Array(value=np.array([[[0]]]))
    grid_z = Array(value=np.array([[[0]]]))

    grid_data = Array(value=np.array([[[0]]]))

    @observe('low_point', post_init=True)
    @observe('high_point', post_init=True)
    def update_line_figure(self, event):
        x = np.linspace(self.low_point[0], self.high_point[0], 1000)
        y = np.linspace(self.low_point[1], self.high_point[1], 1000)
        z = np.linspace(self.low_point[2], self.high_point[2], 1000)

        line_pos = np.linspace(0, np.linalg.norm(self.high_point - self.low_point), 1000)
        line_data = self.interp_func((x, y, z))

        axes = self.line_figure.axes[0]
        if not np.isnan(line_data).all():
            axes.set_ylim(0, (1 + 0.1) * np.nanmax(line_data))
            axes.set_xlim(0, line_pos[-1])
        self.plot[0].set_data(line_pos, line_data)
        canvas = self.line_figure.canvas
        if canvas is not None:
            canvas.draw()

    def create_line_figure(self):
        self.interp_func = RegularGridInterpolator((self.grid_x, self.grid_y, self.grid_z), self.grid_data,
                                                   bounds_error=False)

        x = np.linspace(self.low_point[0], self.high_point[0], 1000)
        y = np.linspace(self.low_point[1], self.high_point[1], 1000)
        z = np.linspace(self.low_point[2], self.high_point[2], 1000)

        line_pos = np.linspace(0, np.linalg.norm(self.high_point - self.low_point), 1000)

        line_data = self.interp_func((x, y, z))

        self.line_figure.clear()
        ax = self.line_figure.add_subplot(111)
        self.plot = ax.plot(line_pos, line_data)
        ax.set_ylabel('Current Density ($A/m^2$)')
        ax.set_xlabel('Distance from Point 1 (mm)')

        self.line_figure.suptitle('Current Density Along Line')

    def default_traits_view(self):
        return View(Group(
            Item('line_figure', editor=MPLFigureEditor(), show_label=False),
            Group(
                Item('low_point', label='Point 1', editor=ArrayEditor(width=-60)),
                Item('high_point', label='Point 2', editor=ArrayEditor(width=-60)),
            ),
        ),
            handler=LeftHandler()
        )


class SlicePanel(HasTraits):
    figure = Instance(Figure)

    coord_index = Int(0)
    index_low = Int(0)
    index_high = Int(0)

    coord_map = Array(value=np.array([0]))
    coord_label = Str('Z')

    low_label = Str()
    high_label = Str()

    plane_type = Enum('Normal to Z',
                      'Normal to X',
                      'Normal to Y',
                      'Arbitrary Plane')

    normal = Array(value=np.array([0, 0, 1]), dtype=np.float)
    origin = Array(value=np.array([0, 0, 0]), dtype=np.float)

    @observe('plane_type')
    def change_plane_type(self, event):
        if self.plane_type == 'Normal to Z' or self.plane_type == 'Arbitrary Plane':
            self.coord_label = 'Z'
        elif self.plane_type == 'Normal to X':
            self.coord_label = 'X'
        elif self.plane_type == 'Normal to Y':
            self.coord_label = 'Y'

    def default_traits_view(self):
        return View(Group(
            Item('figure', editor=MPLFigureEditor(), show_label=False),
            Group(
                Item(
                    'coord_index',
                    label=self.coord_label,
                    editor=QRangeEditor(
                        low_name='index_low',
                        high_name='index_high',
                        low_label_name='low_label',
                        high_label_name='high_label',
                        map_to_values_name='coord_map',
                        mode='slider',
                        is_float=False, ),
                    padding=15,
                ),
            ),
            Item('plane_type',
                 editor=EnumEditor(
                     values={
                         'Normal to X':'1:Normal to X',
                         # 'Normal to Y':'2:Normal to Y',
                         'Normal to Z': '3:Normal to Z',
                         'Arbitrary Plane': '4:Arbitrary Plane',
                     },
                     format_func=str,
                     cols=4
                 ),
                 style='custom',
                 show_label=False),
            Group(
                Item('normal', editor=ArrayEditor(width=-60)),
                Item('origin', editor=ArrayEditor(width=-60)),
                visible_when='plane_type == "Arbitrary Plane"')))


class LeftPanel(HasTraits):
    line_panel = Instance(LinePanel)
    slice_panel = Instance(SlicePanel)

    def default_traits_view(self):
        return View(Group(
            Item('slice_panel', dock='tab', style='custom', show_label=False),  # ,width=1000,height=800,),
            Item('line_panel', dock='tab', style='custom', show_label=False),
            layout='tabbed',
        ),
            handler=LeftHandler()
        )


class LeftHandler(Handler):

    def setattr(self, info, object, name, value):
        if info.ui.history is None:
            if info.ui.parent.history is None:
                info.ui.parent.history = info.ui.parent.parent.history
            info.ui.history = info.ui.parent.history
        info.ui.history.add(
            UndoItem(object=object, name=name, old_value=object.trait_get([name])[name], new_value=value))
        info.ui.do_undoable(Handler.setattr, self, info, object, name, value)


class RightPanel(HasTraits):
    plot_3d = Instance(MlabSceneModel)

    view = View(Group(
        Item('plot_3d',
             show_label=False,
             editor=SceneEditor(scene_class=MayaviScene)),
    ))


class MainWindow(HasTraits):
    prog_title = Str("Sim4Life Field Data Viewer")
    scene = Instance(MlabSceneModel, ())
    plotter = Instance(Plotter)

    slice_panel = Instance(SlicePanel)
    line_panel = Instance(LinePanel)

    left_panel = Instance(LeftPanel)
    right_panel = Instance(RightPanel)

    data_dir = Str()
    csf_model = File(value=default_csf_model)

    show_full_model = Bool(False)

    tmp_action = Action(name="Action_Name")

    open_file_action = Action(name="Open File", action='open_new_file')
    open_file_button = Button(label='Open data file', style='button')

    open_recent_button = Button(label="Restore previous session", style='button', visible_when='has_state')

    save_all_action = Action(name='Save All', action='save_all', enabled_when='data_dir != ""')
    save_slice_action = Action(name='Slice Figure', aciton='save_slice', enabled_when='data_dir != ""')

    new_model_action = Action(name='New cord model', action='change_cord_model', enabled_when='data_dir != ""')

    full_model_action = menu.Action(name='Show full model', action='toggle_full_model', checked_when='show_full_model',
                                    style='toggle', enabled_when='data_dir != ""')

    masked_gr_x = Array()
    masked_gr_y = Array()
    masked_gr_z = Array()

    coord = Int(0)
    low = Int(0)
    high = Int(0)

    low_label = Str()
    high_label = Str()

    coord_map = Array(value=np.array([0]))

    low_point = Array(value=np.array([0, 0, 0]))
    high_point = Array(value=np.array([0, 0, 1]))

    log_scale = Bool(True)

    has_state = Bool(False)

    field_initialized = Bool(False)

    available_fields = ListStr()
    current_field = Str()

    def toggle_full_model(self, info):
        info.ui.do_undoable(self._toggle_full_model, info)

    def _toggle_full_model(self, info):
        info.ui.history.add(UndoItem(object=self, name='show_full_model', old_value=self.show_full_model,
                                     new_value=not self.show_full_model))
        self.show_full_model = not self.show_full_model

    @observe('scene.activated')
    def _initialize_camera(self, info):
        # self.scene.mlab.options.offscreen = True
        self.scene.scene.camera.position = [195, 200, 200]
        self.scene.scene.camera.focal_point = [-5.0, -0.5, -0.3]
        self.scene.scene.camera.view_angle = 30.0
        self.scene.scene.camera.view_up = [0.0, 0.0, 1.0]
        self.scene.scene.camera.clipping_range = [215, 500]
        self.scene.scene.camera.compute_view_plane_normal()

    def initialize_field(self, dataPath):
        if dataPath.endswith('.txt'):
            jFieldDF = pd.read_table(dataPath, names=['x', 'y', 'z', 'ReJx', 'ImJx', 'ReJy', 'ImJy', 'ReJz', 'ImJz'],
                                     dtype=np.float64, comment='#', index_col=False)

            jFieldDF.loc[:, 'x':'z'] = jFieldDF.loc[:, 'x':'z'] * 1000

            jFieldDF = jFieldDF.assign(Jx=lambda x: x.ReJx + complex(0, 1) * x.ImJx)
            jFieldDF = jFieldDF.assign(Jy=lambda x: x.ReJy + complex(0, 1) * x.ImJy)
            jFieldDF = jFieldDF.assign(Jz=lambda x: x.ReJz + complex(0, 1) * x.ImJz)

            jFieldDF = jFieldDF.drop(columns=['ReJx', 'ImJx', 'ReJy', 'ImJy', 'ReJz', 'ImJz'])

            jFieldDF = jFieldDF.assign(J_mag=np.sqrt(
                np.square(np.abs(jFieldDF.loc[:, 'Jx'])) + np.square(np.abs(jFieldDF.loc[:, 'Jy'])) + np.square(
                    np.abs(jFieldDF.loc[:, 'Jz']))))

            grouped = jFieldDF.groupby(['x', 'y', 'z'])['J_mag'].mean()

            shape = tuple(map(len, grouped.index.levels))
            self.dataArr = np.full(shape, np.nan)

            self.dataArr[tuple(grouped.index.codes)] = np.where(grouped.values.flat > 0, grouped.values.flat, np.nan)

            self.x_vals = jFieldDF.x.drop_duplicates().sort_values().values
            self.y_vals = jFieldDF.y.drop_duplicates().sort_values().values
            self.z_vals = jFieldDF.z.drop_duplicates().sort_values().values
        elif dataPath.endswith('.mat'):
            mat_data = loadmat(dataPath)

            g_z = mat_data['Axis0'][0] * 1000
            g_y = mat_data['Axis1'][0] * 1000
            g_x = mat_data['Axis2'][0] * 1000

            self.x_vals = np.array([(g_x[i] + g_x[i + 1]) / 2 for i in range(g_x.size - 1)])
            self.y_vals = np.array([(g_y[i] + g_y[i + 1]) / 2 for i in range(g_y.size - 1)])
            self.z_vals = np.array([(g_z[i] + g_z[i + 1]) / 2 for i in range(g_z.size - 1)])

            self.available_fields = [x for x in mat_data.keys() if 'Snapshot' in x]
            if self.current_field is None or self.current_field == '':
                self.current_field = [x for x in self.available_fields if 'J' in x][0]

            jx, jy, jz = abs(mat_data[self.current_field]).T

            self.dataArr = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2).reshape(self.x_vals.size, self.y_vals.size,
                                                                        self.z_vals.size)
            self.dataArr[self.dataArr == 0] = np.nan

            tmp = self.x_vals
            self.x_vals = self.z_vals
            self.z_vals = tmp

            self.dataArr = np.swapaxes(self.dataArr, 0, 2)

        self.x_min = int(np.ceil(self.x_vals.min()))
        self.x_max = int(np.floor(self.x_vals.max()))
        self.y_min = int(np.ceil(self.y_vals.min()))
        self.y_max = int(np.floor(self.y_vals.max()))
        self.z_min = int(np.ceil(self.z_vals.min()))
        self.z_max = int(np.floor(self.z_vals.max()))

        self.gr_x, self.gr_y, self.gr_z = np.mgrid[self.x_min:self.x_max:len(self.x_vals) * 1j,
                                          self.y_min:self.y_max:len(self.y_vals) * 1j,
                                          self.z_min:self.z_max:len(self.z_vals) * 1j]

        points = np.array(
            [[self.gr_x[i, j, k], self.gr_y[i, j, k], self.gr_z[i, j, k]] for i in range(self.gr_x.shape[0]) for j
             in range(self.gr_x.shape[1]) for k in range(self.gr_x.shape[2])])

        interp_func = RegularGridInterpolator((self.x_vals, self.y_vals, self.z_vals), self.dataArr)
        self.grid_data = interp_func(points).reshape(self.dataArr.shape)

        mask = np.all(np.isnan(self.grid_data), axis=(0, 1))
        self.masked_grid_data = self.grid_data[:, :, ~mask]
        self.masked_gr_x = self.gr_x[:, :, ~mask]
        self.masked_gr_y = self.gr_y[:, :, ~mask]
        self.masked_gr_z = self.gr_z[:, :, ~mask]

        maskx = np.all(np.isnan(self.masked_grid_data), axis=(1, 2))
        self.masked_grid_data = self.masked_grid_data[~maskx, :, :]
        self.masked_gr_x = self.masked_gr_x[~maskx, :, :]
        self.masked_gr_y = self.masked_gr_y[~maskx, :, :]
        self.masked_gr_z = self.masked_gr_z[~maskx, :, :]

        masky = np.all(np.isnan(self.masked_grid_data), axis=(0, 2))
        self.masked_grid_data = self.masked_grid_data[:, ~masky, :]
        self.masked_gr_x = self.masked_gr_x[:, ~masky, :]
        self.masked_gr_y = self.masked_gr_y[:, ~masky, :]
        self.masked_gr_z = self.masked_gr_z[:, ~masky, :]

        self.max_ind = np.unravel_index(np.nanargmax(self.masked_grid_data), self.masked_grid_data.shape)
        jMax = self.masked_grid_data[self.max_ind]

        self.masked_z_vals = self.masked_gr_z[0, 0, :]
        self.high = len(self.masked_grid_data[0, 0, :]) - 1
        self.coord_map = self.masked_z_vals

        self.low_label = '{:.2f} mm'.format(self.masked_z_vals[0])
        self.high_label = '{:.2f} mm'.format(self.masked_z_vals[self.high])

        self.coord = self.max_ind[-1]

        self.field_initialized = True

    def evaluate_slider_values(self, value):
        return find_nearest(self.z_vals, value)

    def default_traits_view(self):
        return QtView(Group(
            Group(
                Item(
                    'prog_title',
                    show_label=False,
                    style='readonly',
                    style_sheet='*{qproperty-alignment:AlignHCenter; font-size: 20px; font-weight: bold}',
                    width=0.9
                ),
                Group(
                    Item(
                        'open_file_button',
                        show_label=False,
                        padding=15
                    ),
                    Item(
                        'open_recent_button',
                        show_label=False,
                        padding=15,
                        visible_when='has_state',
                    ),
                    orientation='horizontal',
                    show_labels=False
                ),
                visible_when='data_dir == ""'),
            HSplit(
                Item('left_panel', dock='vertical', style='custom', width=1000, height=800),
                Item('right_panel', style='custom', width=500, height=800),
                show_labels=False,
                visible_when='data_dir != ""',
            ),
        ),
            resizable=True,
            title=self.prog_title,
            buttons=menu.NoButtons,
            handler=SVHandler(),
            x=0.1,
            y=0.1,
            # statusbar = ['data_dir','csf_model'],
            menubar=MenuBar(
                Menu(
                    self.open_file_action,
                    Menu(
                        Action(name="Export Slice", action="export_slice",
                               enabled_when='data_dir != ""'),
                        Action(name="Export Line", action="export_line", enabled_when='data_dir != ""'),
                        name='&Export'
                    ),
                    name='&File'
                ),
                Menu(
                    menu.UndoAction,
                    menu.RedoAction,
                    self.new_model_action,
                    self.full_model_action,
                    Action(name='Log scale', action='change_scale', enabled_when='data_dir != ""',
                           checked_when='log_scale', style='toggle'),
                    # ActionGroup((Action(name=x.split('Snapshot')[0].replace('_', ' '))),
                    #             name='Select Field',
                    #             enabled_when='len(available_fields) > 0'
                    #             ),

                    name='&Edit'
                )
            )
        )

    def change_scale(self, info):
        self.log_scale = not self.log_scale

    def export_slice(self, info):
        dataset = info.object.plotter.cut.outputs[0].output
        datax, datay, dataz = dataset.points.to_array().T
        scalar_data = dataset.point_data.scalars.to_array()

        true_x = np.unique(np.floor(datax / scale_factor).astype(int)) * scale_factor
        true_y = np.unique(np.floor(datay / scale_factor).astype(int)) * scale_factor

        true_data = np.empty((true_y.size, true_x.size))
        true_data[:] = np.nan

        for i in range(scalar_data.size):
            true_data[arg_find_nearest(true_y, datay[i]), arg_find_nearest(true_x, datax[i])] = scalar_data[i]

        outDF = pd.DataFrame(data=true_data, index=true_y, columns=true_x)

        out_file, file_type = QFileDialog.getSaveFileName(None, 'Export Data', os.path.join(os.getcwd(), '_slice'),
                                                          "Excel files (*.xlsx);;Comma separated files (*.csv *.txt)",
                                                          "Excel files (*.xlsx)")

        if '.xlsx' in file_type:
            if out_file.split('.')[-1] != 'xlsx':
                out_file = '.'.join(out_file.split('.')[:-1].append('xlsx'))
            outDF.to_excel(out_file)
        elif '.csv' in file_type:
            if out_file.split('.')[-1] != 'csv' and out_file.split('.')[-1] != 'txt':
                out_file = '.'.join(out_file.split('.')[:-1].append('csv'))
            outDF.to_csv(out_file)

    def export_line(self, info):
        l = info.object.plotter.line_low_point
        h = info.object.plotter.line_high_point

        x = np.linspace(l[0], h[0], 1000)
        y = np.linspace(l[1], h[1], 1000)
        z = np.linspace(l[2], h[2], 1000)

        line_pos = np.linspace(0, np.linalg.norm(h - l), 1000)
        line_data = info.object.line_panel.interp_func((x, y, z))

        outDF = pd.DataFrame(data=line_data, index=line_pos, columns=['y'])

        out_file, file_type = QFileDialog.getSaveFileName(None, 'Export Data', os.path.join(os.getcwd(), '_slice'),
                                                          "Excel files (*.xlsx);;Comma separated files (*.csv *.txt)",
                                                          "Excel files (*.xlsx)")

        if '.xlsx' in file_type:
            if out_file.split('.')[-1] != 'xlsx':
                out_file = '.'.join(out_file.split('.')[:-1].append('xlsx'))
            outDF.to_excel(out_file)
        elif '.csv' in file_type:
            if out_file.split('.')[-1] != 'csv' and out_file.split('.')[-1] != 'txt':
                out_file = '.'.join(out_file.split('.')[:-1].append('csv'))
            outDF.to_csv(out_file)

    @observe('open_recent_button', post_init=True)
    def restore_session(self, info):
        with open('last_state.pickle', 'rb') as f:
            l_state = pickle.load(f)
        # from pprint import pprint
        # pprint(l_state)

        self.data_dir = l_state['data_dir']
        # self.initialize_field(self.data_dir)

        self.csf_model = l_state['csf_model']
        self.coord = l_state['coord']
        # self.scene = l_state['scene']
        self.masked_gr_x = l_state['masked_gr_x']
        self.masked_gr_y = l_state['masked_gr_y']
        self.masked_gr_z = l_state['masked_gr_z']
        self.masked_grid_data = l_state['masked_grid_data']
        self.coord_map = l_state['coord_map']
        self.low = l_state['low']
        self.high = l_state['high']
        self.x_vals = l_state['x_vals']
        self.y_vals = l_state['y_vals']
        self.z_vals = l_state['z_vals']
        self.dataArr = l_state['dataArr']
        self.low_label = l_state['low_label']
        self.high_label = l_state['high_label']

        self.field_initialized = True

        self.open_data_file(info)
        self.plotter.line_low_point = l_state['low_point']
        self.plotter.line_high_point = l_state['high_point']
        self.plotter.normal = l_state['normal']
        self.plotter.origin = l_state['origin']
        self.plotter.plane_type = l_state['plane_type']

        self.show_model = l_state['show_model']
        self.log_scale = l_state['log_scale']

    def change_cord_model(self, info):
        self.plotter.csf_model, _ = QFileDialog.getOpenFileName(None, 'Open Spinal Cord Model File',
                                                                os.path.basename(self.plotter.csf_model),
                                                                'VTK Files (*.vtk)')
        self._initialize_camera(None)

    def save_all(self, info):
        out_prefix, _ = QFileDialog.getSaveFileName(None, 'Save Figures As', os.getcwd())

        self.slice_panel.figure.savefig(''.join([out_prefix, '_slice.png']))
        self.line_panel.line_figure.savefig(''.join([out_prefix, '_line.png']))

    def save_slice(self, info):
        print('saving slice')
        out_name, _ = QFileDialog.getSaveFileName(None, 'Save F:xile', os.path.join(os.getcwd(), 'slice.png'),
                                                  'Images (*.png *.jpg)')

    def open_new_file(self):
        dialog = FileDialog(action='open', wildcard=FileDialog.create_wildcard('Data Files', ['*.mat']))
        result = dialog.open()

        if result == OK:
            self.data_dir = dialog.path
        else:
            return None

        GUI.set_busy(True)

        self.initialize_field(self.data_dir)

        self.plotter.grid_x = self.masked_gr_x
        self.plotter.grid_y = self.masked_gr_y
        self.plotter.grid_z = self.masked_gr_z
        self.plotter.grid_data = self.masked_grid_data
        self.plotter.coord_map = self.coord_map
        self.plotter.coord = self.coord
        self.plotter.low = self.low
        self.plotter.high = self.high

        self.line_panel.grid_x = self.x_vals
        self.line_panel.grid_y = self.y_vals
        self.line_panel.grid_z = self.z_vals
        self.line_panel.grid_data = self.dataArr

        self.slice_panel.coord_index = self.coord
        self.slice_panel.index_low = self.low
        self.slice_panel.index_high = self.high
        self.slice_panel.coord_map = self.coord_map
        self.slice_panel.low_label = self.low_label
        self.slice_panel.high_label = self.high_label

        GUI.set_busy(False)

    @observe('open_file_button', post_init=True)
    def open_data_file(self, info):
        GUI.set_busy(True)
        if not self.field_initialized:
            self.data_dir, _ = QFileDialog.getOpenFileName(None, "Open Data File", os.getcwd(),
                                                           'Data Files (*.mat)')
            self.initialize_field(self.data_dir)

        self.plotter = Plotter(
            scene=self.scene,
            csf_model=self.csf_model,
            grid_x=self.masked_gr_x,
            grid_y=self.masked_gr_y,
            grid_z=self.masked_gr_z,
            grid_data=self.masked_grid_data,
            coord_map=self.coord_map,
            coord=self.coord,
            low=self.low,
            high=self.high,
            log_scale=self.log_scale,
        )

        self.plotter.create_plot()

        self.line_panel = LinePanel(
            low_point=self.plotter.line_low_point,
            high_point=self.plotter.line_high_point,
            grid_x=self.x_vals,
            grid_y=self.y_vals,
            grid_z=self.z_vals,
            grid_data=self.dataArr
        )

        self.line_panel.create_line_figure()

        self.slice_panel = SlicePanel(
            figure=self.plotter.figure,
            coord_index=self.coord,
            index_low=self.low,
            index_high=self.high,
            coord_map=self.coord_map,
            low_label=self.low_label,
            high_label=self.high_label
        )

        self.left_panel = LeftPanel(
            line_panel=self.line_panel,
            slice_panel=self.slice_panel
        )

        self.right_panel = RightPanel(
            plot_3d=self.scene,
            normal=self.plotter.normal,
            origin=self.plotter.origin
        )

        self.plotter.sync_trait('figure', self.slice_panel)
        self.plotter.sync_trait('scene', self.right_panel, alias='plot_3d')
        self.plotter.sync_trait('coord', self.slice_panel, alias='coord_index')
        self.plotter.sync_trait('high', self.slice_panel, alias='high_index')
        self.plotter.sync_trait('low', self.slice_panel, alias='low_index')
        self.plotter.sync_trait('origin', self.slice_panel)
        self.plotter.sync_trait('normal', self.slice_panel)
        self.plotter.sync_trait('plane_type', self.slice_panel)
        self.plotter.sync_trait('line_low_point', self.line_panel, alias='low_point')
        self.plotter.sync_trait('line_high_point', self.line_panel, alias='high_point')
        self.plotter.sync_trait('show_full_model', self)
        self.plotter.sync_trait('log_scale', self)
        self.plotter.sync_trait('csf_model', self)

        GUI.set_busy(False)


if __name__ == '__main__':
    from traits.api import push_exception_handler

    push_exception_handler(reraise_exceptions=True)
    has_state = os.path.exists('last_state.pickle')
    app = MainWindow(has_state=has_state)
    app.configure_traits()
