import copy

import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

# We want matplotlib to use a QT backend
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize

from .mpl_figure_editor import MPLFigureEditor
from .util import ArrayClass, scale_factor, scalar_fields, arg_find_nearest

from matplotlib.figure import Figure

import numpy as np
from pyface.tasks.api import Editor
from scipy.interpolate import RegularGridInterpolator

from traits.api import (HasTraits, File, Dict, Str, Bool, List, Any, Instance, observe, Array, ListStr, Button,
                        DelegatesTo)
import traits.observation.api as ob

from traitsui.api import View, Item, Group, Spring

from tvtk.pyface.scene_editor import SceneEditor

from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.sources.array_source import ArraySource
from mayavi.filters.cut_plane import CutPlane
from mayavi.modules.surface import Surface
from mayavi.filters.data_set_clipper import DataSetClipper

from scipy.io import loadmat

from .preferences import default_csf_model


class EMFields(HasTraits):
    """ A collection of EM fields output from Sim4Life EM simulations
    """
    # Path to field data file
    data_path = File()

    data_dict = Dict()
    field_keys = ListStr()
    selected_field_key = Str

    x_vals = Array()
    y_vals = Array()
    z_vals = Array()
    
    data_arr = Array()

    masked_gr_x = Array()
    masked_gr_y = Array()
    masked_gr_z = Array()

    masked_grid_data = Array()

    @observe('data_path')
    def _update_data_path(self, event):
        self.data_dict = loadmat(event.new)

        g_z = self.data_dict['Axis0'][0] * 1000
        g_y = self.data_dict['Axis1'][0] * 1000
        g_x = self.data_dict['Axis2'][0] * 1000

        self.x_vals = np.array([(g_x[i] + g_x[i + 1]) / 2 for i in range(g_x.size - 1)])
        self.y_vals = np.array([(g_y[i] + g_y[i + 1]) / 2 for i in range(g_y.size - 1)])
        self.z_vals = np.array([(g_z[i] + g_z[i + 1]) / 2 for i in range(g_z.size - 1)])

        tmp = self.x_vals
        self.x_vals = self.z_vals
        self.z_vals = tmp

        self.field_keys = [key for key in self.data_dict.keys() if 'Snapshot' in key and
                           not any([field in key for field in scalar_fields])]

        if self.selected_field_key not in self.field_keys:
            self.selected_field_key = self.field_keys[0]
            for key in self.field_keys:
                if key.startswith("J"):
                    self.selected_field_key = key
                    break

        self.calculate_field()

    @observe('selected_field_key', post_init=True)
    def calculate_field(self, event=None):
        data_x, data_y, data_z = abs(self.data_dict[self.selected_field_key]).T

        self.data_arr = np.sqrt(data_x ** 2 + data_y ** 2 + data_z ** 2).reshape(self.z_vals.size,
                                                                                 self.y_vals.size,
                                                                                 self.x_vals.size)
        self.data_arr[self.data_arr == 0] = np.nan

        self.data_arr = np.swapaxes(self.data_arr, 0, 2)

        x_min = int(np.ceil(self.x_vals.min()))
        x_max = int(np.floor(self.x_vals.max()))
        y_min = int(np.ceil(self.y_vals.min()))
        y_max = int(np.floor(self.y_vals.max()))
        z_min = int(np.ceil(self.z_vals.min()))
        z_max = int(np.floor(self.z_vals.max()))

        gr_x, gr_y, gr_z = np.mgrid[x_min:x_max:len(self.x_vals) * 1j,
                                    y_min:y_max:len(self.y_vals) * 1j,
                                    z_min:z_max:len(self.z_vals) * 1j]

        points = np.array(
            [[gr_x[i, j, k], gr_y[i, j, k], gr_z[i, j, k]] for i in range(gr_x.shape[0]) for j
             in range(gr_x.shape[1]) for k in range(gr_x.shape[2])])

        interp_func = RegularGridInterpolator((self.x_vals, self.y_vals, self.z_vals), self.data_arr)
        grid_data = interp_func(points).reshape(self.data_arr.shape)

        mask = np.all(np.isnan(grid_data), axis=(0, 1))
        masked_grid_data = grid_data[:, :, ~mask]
        masked_gr_x = gr_x[:, :, ~mask]
        masked_gr_y = gr_y[:, :, ~mask]
        masked_gr_z = gr_z[:, :, ~mask]

        maskx = np.all(np.isnan(masked_grid_data), axis=(1, 2))
        masked_grid_data = masked_grid_data[~maskx, :, :]
        masked_gr_x = masked_gr_x[~maskx, :, :]
        masked_gr_y = masked_gr_y[~maskx, :, :]
        masked_gr_z = masked_gr_z[~maskx, :, :]

        masky = np.all(np.isnan(masked_grid_data), axis=(0, 2))
        self.masked_grid_data = masked_grid_data[:, ~masky, :]
        self.masked_gr_x = masked_gr_x[:, ~masky, :]
        self.masked_gr_y = masked_gr_y[:, ~masky, :]
        self.masked_gr_z = masked_gr_z[:, ~masky, :]


class Mayavi3DScene(Editor):
    model = Instance(HasTraits)
    ui = Instance("traitsui.ui.UI")

    name = Str('3D View')

    fields_model = Instance(EMFields)

    # Normal vector of plane
    normal = Array(value=np.array([0, 0, 1]))

    # Origin point of plane
    origin = Array(value=np.array([0, 0, 0]))

    scene = Instance(MlabSceneModel, ())

    data_set_clipper = Instance(DataSetClipper)

    points = List(ArrayClass, value=[ArrayClass(value=np.array([0, 0, -1])), ArrayClass(value=np.array([0, 0, 1]))])
    line = Instance(Surface)
    line_points = Any()

    src = Instance(ArraySource)
    cut = Instance(CutPlane)
    surf = Instance(Surface)

    csf_model = File(value=default_csf_model)
    csf_model_reader = Any()
    csf_surface = Instance(Surface)

    show_full_model = Bool(False)
    full_csf_surface = Instance(Surface)

    log_scale = Bool(True)

    def default_traits_view(self):
        return View(Item('scene', show_label=False, editor=SceneEditor(scene_class=MayaviScene)))

    def create(self, parent):
        self.ui = self.edit_traits(kind='subpanel', parent=parent)
        self.control = self.ui.control

    def destroy(self):
        self.control = None
        if self.ui is not None:
            self.ui.dispose()
        self.ui = None

    @observe('log_scale', post_init=True)
    def toggle_log_scale(self, event):
        if event.new:
            self.surf.parent.scalar_lut_manager.lut.scale = 'log10'
        else:
            self.surf.parent.scalar_lut_manager.lut.scale = 'linear'
        self.scene.mlab.draw()

    @observe('origin', post_init=True)
    def update_origin(self, event):
        if hasattr(self.data_set_clipper, 'widget'):
            self.data_set_clipper.widget.widget.origin = event.new
            self.cut.filters[0].widget.origin = event.new
        self.scene.mlab.draw()

    @observe('normal', post_init=True)
    def update_normal(self, event):
        if hasattr(self.data_set_clipper, 'widget'):
            self.data_set_clipper.widget.widget.normal = event.new
            self.cut.filters[0].widget.normal = event.new
        self.scene.mlab.draw()

    @observe('show_full_model', post_init=True)
    def toggle_full_model(self, event):
        self.csf_surface.visible = not self.show_full_model
        self.full_csf_surface.visible = self.show_full_model

        self.scene.mlab.draw()

    @observe('csf_model', post_init=True)
    def change_cord_model(self, event):
        if self.csf_model_reader is not None:
            self.csf_model_reader.initialize(self.csf_model)

    @observe('scene.activated')
    def initialize_camera(self, info=None):
        if self.csf_surface is not None:
            self.scene.engine.current_object = self.csf_surface
        self.scene.mlab.view(azimuth=-35, elevation=75)

        self.scene.mlab.draw()

    def create_plot(self):
        normal = self.normal

        max_ind  = np.unravel_index(np.nanargmax(self.fields_model.masked_grid_data),
                                   self.fields_model.masked_grid_data.shape)

        self.origin = np.array([self.fields_model.masked_gr_x[max_ind],
                                self.fields_model.masked_gr_y[max_ind],
                                self.fields_model.masked_gr_z[max_ind]])

        self.csf_model_reader = self.scene.engine.open(self.csf_model)
        self.csf_surface = Surface()

        self.data_set_clipper = DataSetClipper()

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

        self.src = self.scene.mlab.pipeline.scalar_field(self.fields_model.masked_gr_x,
                                                         self.fields_model.masked_gr_y,
                                                         self.fields_model.masked_gr_z,
                                                         self.fields_model.masked_grid_data)
        self.cut = self.scene.mlab.pipeline.cut_plane(self.src)
        self.cut.filters[0].widget.normal = normal
        self.cut.filters[0].widget.origin = self.origin
        self.cut.filters[0].widget.enabled = False
        self.surf = self.scene.mlab.pipeline.surface(self.cut, colormap='jet')
        self.surf.actor.actor.use_bounds = False
        self.surf.parent.scalar_lut_manager.lut.nan_color = np.array([0, 0, 0, 0])

        self.scene.mlab.draw()

    @observe(ob.trait('points').list_items().trait('value', optional=True).list_items(optional=True), post_init=True)
    def draw_line(self, event):
        if None in event.new and len(event.old) == len(event.new) and None not in event.old:
            self.points = event.old
            return
        points = np.array([val.value if val is not None else np.array([0, 0, 0]) for val in self.points])

        x = []
        y = []
        z = []

        for point in points:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])

        if not hasattr(self.line, 'mlab_source'):
            self.line = self.scene.mlab.plot3d(x, y, z, tube_radius=0.2, color=(1, 0, 0), figure=self.scene.mayavi_scene)
        else:
            self.line.mlab_source.reset(x=x, y=y, z=z)

        self.scene.mlab.draw()

    def disable_widgets(self):
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
                [np.nanmin(self.fields_model.masked_grid_data),
                 np.nanmax(self.fields_model.masked_grid_data)])
            self.surf.parent.scalar_lut_manager.lut.nan_color = np.array([0, 0, 0, 0])


class SliceFigureModel(Editor):
    model = Instance(HasTraits)
    ui = Instance("traitsui.ui.UI")

    name = Str("Slice Plane")

    fields_model = Instance(EMFields)
    mayavi_scene = Instance(Mayavi3DScene)

    figure = Instance(Figure, ())

    log_scale = Bool(True)

    mycmap = Any()
    norm = Any()
    pcm = Any()
    clb = Any()

    draw_cross = Bool(True)
    points = DelegatesTo('mayavi_scene')
    line_cross = Any()

    def default_traits_view(self):
        return View(
            Group(
                Item('figure', editor=MPLFigureEditor(), show_label=False),
            )
        )

    def create(self, parent):
        self.ui = self.edit_traits(kind='subpanel', parent=parent)
        self.control = self.ui.control

    def destroy(self):
        self.control = None
        if self.ui is not None:
            self.ui.dispose()
        self.ui = None

    def export_slice(self, file_path):
        x, y, data = self._calculate_plane()

        out_df = pd.DataFrame(data=data, index=y, columns=x)

        if file_path.endswith('.xlsx'):
            out_df.to_excel(file_path)
        elif file_path.endswith('.csv') or file_path.endswith('.txt'):
            out_df.to_csv(file_path)

    @observe('log_scale', post_init=True)
    def toggle_log_scale(self, event):
        if event.new:
            self.norm = LogNorm(
                vmin=np.nanmin(self.fields_model.masked_grid_data),
                vmax=np.nanmax(self.fields_model.masked_grid_data)
                                )
        else:
            self.norm = Normalize(
                vmin=np.nanmin(self.fields_model.masked_grid_data),
                vmax=np.nanmax(self.fields_model.masked_grid_data)
            )

        self.clb.update_normal(cm.ScalarMappable(norm=self.norm, cmap=self.mycmap))

        self.update_plot(event)

    def create_plot(self):
        true_x, true_y, true_data = self._calculate_plane()
        self.mycmap = copy.copy(cm.get_cmap('jet'))

        self.figure.clear()

        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(right=0.75, bottom=0.15)

        if self.log_scale:
            self.norm = LogNorm(vmin=np.nanmin(self.fields_model.masked_grid_data),
                                vmax=np.nanmax(self.fields_model.masked_grid_data))
        else:
            self.norm = Normalize(vmin=np.nanmin(self.fields_model.masked_grid_data),
                                  vmax=np.nanmax(self.fields_model.masked_grid_data))

        self.pcm = ax.pcolormesh(true_x, true_y, true_data, shading='nearest', cmap=self.mycmap, norm=self.norm)

        if self.draw_cross:
            self.line_cross = ax.plot([0], [0], 'rx')
        else:
            self.line_cross = ax.plot([0], [0], '')

        ax.set_ylim(bottom=np.nanmin(self.fields_model.masked_gr_y[0, :, 0]) - 2,
                    top=np.nanmax(self.fields_model.masked_gr_y[0, :, 0]) + 2)

        ax.set_xlim(left=np.nanmin(self.fields_model.masked_gr_x[:, 0, 0]) - 2,
                    right=np.nanmax(self.fields_model.masked_gr_x[:, 0, 0]) + 2)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

        ax.set_title('Current Density Magnitude')

        self.clb = self.figure.colorbar(cm.ScalarMappable(norm=self.norm, cmap=self.mycmap), ax=ax)
        self.clb.set_label('Current Density ($A/m^2$)', rotation=270, labelpad=15)

        self.figure.canvas.draw()

    def _calculate_plane(self):
        dataset = self.mayavi_scene.cut.outputs[0].output
        datax, datay, dataz = dataset.points.to_array().T
        scalar_data = dataset.point_data.scalars.to_array()

        true_x = np.unique(np.floor(datax / scale_factor).astype(int)) * scale_factor
        true_y = np.unique(np.floor(datay / scale_factor).astype(int)) * scale_factor

        true_data = np.empty((true_y.size, true_x.size))
        true_data[:] = np.nan

        for i in range(scalar_data.size):
            true_data[arg_find_nearest(true_y, datay[i]), arg_find_nearest(true_x, datax[i])] = scalar_data[i]

        return true_x, true_y, true_data

    @observe('mayavi_scene:origin')
    @observe('mayavi_scene:normal')
    def update_plot(self, event):
        true_x, true_y, true_data = self._calculate_plane()

        self.update_line_cross()

        axes = self.figure.axes[0]
        self.pcm.remove()
        self.pcm = axes.pcolormesh(true_x, true_y, true_data, shading='nearest', cmap=self.mycmap, norm=self.norm)
        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()

    @observe('draw_cross', post_init=True)
    @observe(ob.trait('points').list_items().trait('value', optional=True).list_items(optional=True))
    def update_line_cross(self, event=None):
        if not self.draw_cross and self.line_cross is not None:
            self.line_cross[0].set_marker('')
        elif self.line_cross is not None:
            nx, ny, nz = self.mayavi_scene.normal
            ox, oy, oz = self.mayavi_scene.origin
            plane_z_0 = -1*(nx*ox + ny*oy)/nz + oz

            points = [val.value if val is not None else np.array([0, 0, 0]) for val in self.points]

            p_under = [val for val in points if val[2] <= plane_z_0]
            p_over = [val for val in points if val[2] > plane_z_0]

            if len(p_under) > 0 and len(p_over) > 0:
                p1 = p_under[-1]
                p2 = p_over[0]

                t = (-1*nx*p1[0] - ny*p1[1] - nz*p1[2] + nx*ox + ny*oy + nz*oz) / \
                    (nx*(p2[0] - p1[0]) + ny*(p2[1] - p1[1]) + nz*(p2[2] - p1[2]))
                x = p1[0] + t*(p2[0] - p1[0])
                y = p1[1] + t*(p2[1] - p1[1])

                self.line_cross[0].set_data([x], [y])
                self.line_cross[0].set_marker('x')

        self.figure.canvas.draw()


class LineFigureModel(Editor):
    model = Instance(HasTraits)
    ui = Instance("traitsui.ui.UI")

    name = Str("Line Plot")

    fields_model = Instance(EMFields)

    figure = Instance(Figure, ())

    points = List(ArrayClass, value=[ArrayClass(value=np.array([0, 0, -1])), ArrayClass(value=np.array([0, 0, 1]))])

    interp_func = Instance(RegularGridInterpolator)

    figure_title = Str()
    x_axis_label = Str('Distance Along Line (mm)')
    y_axis_label = Str()

    use_custom_label = Bool(False)

    def default_traits_view(self):
        return View(
            Group(
                Item('figure', editor=MPLFigureEditor(), show_label=False),
            )
        )

    def create(self, parent):
        self.ui = self.edit_traits(kind='subpanel', parent=parent)
        self.control = self.ui.control

    def destroy(self):
        self.control = None
        if self.ui is not None:
            self.ui.dispose()
        self.ui = None

    def export_line(self, file_path):
        line_pos, line_data = self._fill_data()

        out_df = pd.DataFrame(data=line_data, index=line_pos, columns=['y'])

        if file_path.endswith('.xlsx'):
            out_df.to_excel(file_path)
        elif file_path.endswith('.csv') or file_path.endswith('.txt'):
            out_df.to_csv(file_path)
    
    def _calculate_between_pair(self, point_1, point_2, prev_line_pos=0):
        x = np.linspace(point_1[0], point_2[0], 1000)
        y = np.linspace(point_1[1], point_2[1], 1000)
        z = np.linspace(point_1[2], point_2[2], 1000)
        
        line_pos = np.linspace(0, np.linalg.norm(point_2 - point_1), 1000) + prev_line_pos
        line_data = self.interp_func((x, y, z))
        
        return line_pos, line_data

    def _interp_func_default(self):
        func = RegularGridInterpolator(
            (
                self.fields_model.x_vals,
                self.fields_model.y_vals,
                self.fields_model.z_vals
            ),
            self.fields_model.data_arr
        )
        return func

    def _fill_data(self):
        points = [val.value if val is not None else np.array([0, 0, 0]) for val in self.points]

        line_pos = np.array([])
        line_data = np.array([])
        last_pos = 0
        for i, point in enumerate(points[:-1]):
            pos, data = self._calculate_between_pair(point, points[i + 1], prev_line_pos=last_pos)
            line_pos = np.append(line_pos, pos)
            line_data = np.append(line_data, data)
            last_pos = line_pos[-1]

        return line_pos, line_data

    @observe(ob.trait('points').list_items().trait('value', optional=True).list_items(optional=True), post_init=True)
    def create_plot(self, event):
        line_pos, line_data = self._fill_data()

        self.figure.clear()

        ax = self.figure.add_subplot(111)
        ax.plot(line_pos, line_data)
        ax.set_ylabel(self.y_axis_label)
        ax.set_xlabel(self.x_axis_label)

        self.figure.suptitle(self.figure_title)

        self.figure.canvas.draw()

    @observe('fields_model.selected_field_key')
    def _set_figure_labels(self, event):
        if not self.use_custom_label:
            if self.fields_model.selected_field_key.startswith('J'):
                self.figure_title = 'Current Density Magnitude'
                self.y_axis_label = 'Current Density ($A/m^2$)'
            elif self.fields_model.selected_field_key.startswith('EM_E'):
                self.figure_title = 'Electric Field Magnitude'
                self.y_axis_label = 'Electric Field ($V/m$)'
            elif self.fields_model.selected_field_key.startswith('D'):
                self.figure_title = 'Displacement Flux Density Magnitude'
                self.y_axis_label = 'Displacement Flux Density ($C/m^2$)'
            self.create_plot(event)


class StartPage(Editor):
    model = Instance(HasTraits)
    ui = Instance('traitsui.ui.UI')

    name = Str('Start Page')
    task = Any()

    open_data_file_button = Button(label='Open Data File', style='button')

    def default_traits_view(self):
        return View(
                Group(
                        Spring(),
                        Group(
                                Spring(),
                                Item('open_data_file_button', show_label=False),
                                Spring(),
                                orientation='horizontal'
                        ),
                        Spring(),
                        orientation='vertical'
                )
        )

    def create(self, parent):
        self.ui = self.edit_traits(kind='subpanel', parent=parent)
        self.control = self.ui.control

    def destroy(self):
        self.control = None
        if self.ui is not None:
            self.ui.dispose()
        self.ui = None

    @observe('open_data_file_button', post_init=True)
    def open(self, event):
        self.task.open()
