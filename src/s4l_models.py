"""
Traits models for the S4L Visualization application.
"""
# pylint: disable=attribute-defined-outside-init, unused-argument
import copy
import os
from configparser import ConfigParser

import numpy as np
import pandas as pd
import traits.observation.api as ob
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.filters.cut_plane import CutPlane
from mayavi.filters.data_set_clipper import DataSetClipper
from mayavi.modules.surface import Surface
from mayavi.sources.array_source import ArraySource
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.tasks.i_editor import IEditor, MEditor as Editor
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from traits.api import (
    HasTraits, File, Dict, Str, Bool, List, Any, Instance, observe, Array, ListStr, Button,
    DelegatesTo, provides
)
from traitsui.api import View, Item, Group, Spring
from tvtk.pyface.scene_editor import SceneEditor

from .mpl_figure_editor import MPLFigureEditor
from .util import ArrayClass, SCALE_FACTOR, scalar_fields, arg_find_nearest

# pylint: disable=wrong-import-position, wrong-import-order

# We want matplotlib to use a QT backend
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure


class EMFields(HasTraits):
    """ A collection of EM fields output from Sim4Life EM simulations
    """

    # pylint: disable=too-many-instance-attributes

    #: Configuration parser.
    configuration = Instance(ConfigParser)

    #: Current participant ID.
    participant_id = Str()

    #: Path to field data file.
    data_path = File()

    #: Dictionary of data in `data_path`.
    data_dict = Dict()

    #: List of field keys that can be displayed.
    field_keys = ListStr()

    #: The currently selected field key.
    selected_field_key = Str()

    #: X values of grid in data file.
    x_vals = Array()

    #: Y values of grid in data file.
    y_vals = Array()

    #: Z values of grid in data file.
    z_vals = Array()

    #: Raw data of currently selected field from data file.
    data_arr = Array()

    #: X values of regular grid for current field.
    masked_gr_x = Array()

    #: Y values of regular grid for current field.
    masked_gr_y = Array()

    #: Z values of regular grid for current field.
    masked_gr_z = Array()

    #: Data on regular grid for current field.
    masked_grid_data = Array()

    def __init__(self, configuration, data_path, **kwargs):
        self.configuration = configuration
        self.data_path = data_path

        super().__init__(**kwargs)

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
                           not any(field in key for field in scalar_fields)]

        if self.selected_field_key not in self.field_keys:
            self.selected_field_key = self.field_keys[0]
            for key in self.field_keys:
                if key.lower().startswith(self._get_default_value('initial_field').lower()):
                    self.selected_field_key = key
                    break

        self.calculate_field()

    @observe('selected_field_key', post_init=True)
    def calculate_field(self, event=None):  # pylint: disable=unused-argument, too-many-locals
        """
        Calculate the current selected field values and set the grid locations and values

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for selected_field_key
        """
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

        interp_func = RegularGridInterpolator((self.x_vals, self.y_vals, self.z_vals),
                                              self.data_arr)
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

    def _get_default_value(self, option):
        if self.participant_id is not None:
            if self.participant_id not in self.configuration:
                self.configuration[self.participant_id] = {}
            val = self.configuration[self.participant_id][option]
        else:
            val = self.configuration[self.participant_id][option]
        return val


@provides(IEditor)
class Mayavi3DScene(Editor):  # pylint: disable=too-many-instance-attributes
    """
    A Pyface Tasks Editor for holding a Mayavi scene
    """
    #: The model object to view. If not specified, the editor is used instead.
    model = Instance(HasTraits)

    #: The UI object associated with the Traits view, if it has been
    #: constructed.
    ui = Instance("traitsui.ui.UI")

    #: The editor's user-visible name.
    name = Str('3D View')

    #: Configuration parser.
    configuration = Instance(ConfigParser)

    #: Current participant ID.
    participant_id = Str()

    #: The :py:class:`EMFields` instance containing the field data.
    fields_model = Instance(EMFields)

    #: Normal vector of the cut plane
    normal = Array()

    #: Origin point of the cut plane
    origin = Array()

    #: The :py:class:`mayavi.core.ui.api.MlabSceneModel` instance
    #: containing the 3D plot.
    scene = Instance(MlabSceneModel, ())

    #: The mayavi pipeline object containing the cut plane.
    data_set_clipper = Instance(DataSetClipper)

    #: The list of points describing the line for the line figure.
    points = List(ArrayClass, value=[ArrayClass(value=np.array([0, 0, -1])),
                                     ArrayClass(value=np.array([0, 0, 1]))])

    #: The 3D surface object for the line.
    line = Instance(Surface)

    #: The field data source.
    src = Instance(ArraySource)

    #: The field cut plane.
    cut = Instance(CutPlane)

    #: The 3D surface for the field data.
    surf = Instance(Surface)

    #: The path to the spinal cord model file.
    csf_model = File()

    #: The mayavi file reader object to read the spinal cord model file.
    csf_model_reader = Any()

    #: The 3D surface object for the cut spinal cord model.
    csf_surface = Instance(Surface)

    #: Show the full spinal cord model?
    show_full_model = Bool()

    #: The 3D surface object for the full spinal cord model.
    full_csf_surface = Instance(Surface)

    #: Use a logarithmic scale for the field data?
    log_scale = Bool()

    #: Current participant ID.
    participant_id = Str()

    def default_traits_view(self):  # pylint: disable=no-self-use
        """
        Create the default traits View object for the model

        Returns
        -------
        default_traits_view : :py:class:`traitsui.view.View`
            The default traits View object for the model
        """
        return View(Item('scene', show_label=False, editor=SceneEditor(scene_class=MayaviScene)))

    def create(self, parent):
        """
        Create and set the widget(s) for the Editor.

        Parameters
        ----------
        parent : toolkit-specific widget
            The parent widget for the Editor
        """
        self.ui = self.edit_traits(kind='subpanel', parent=parent)  # pylint: disable=invalid-name
        self.control = self.ui.control  # pylint: disable=attribute-defined-outside-init

    def destroy(self):
        """
        Destroy the Editor and clean up after
        """
        self.control = None  # pylint: disable=attribute-defined-outside-init
        if self.ui is not None:
            self.ui.dispose()
        self.ui = None

    @observe('log_scale', post_init=True)
    def toggle_log_scale(self, event):
        """
        Toggle between using a logarithmic scale and a linear scale

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for log_scale
        """
        if event.new:
            self.surf.parent.scalar_lut_manager.lut.scale = 'log10'
        else:
            self.surf.parent.scalar_lut_manager.lut.scale = 'linear'
        self.scene.mlab.draw()

    @observe('origin', post_init=True)
    def update_origin(self, event):
        """
        Update objects when the cut plane origin is changed.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for origin
        """
        if hasattr(self.data_set_clipper, 'widget'):
            self.data_set_clipper.widget.widget.origin = event.new
            self.cut.filters[0].widget.origin = event.new
        self.scene.mlab.draw()

    @observe('normal', post_init=True)
    def update_normal(self, event):
        """
        Update objects when the cut plane normal is changed.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for normal
        """
        if hasattr(self.data_set_clipper, 'widget'):
            self.data_set_clipper.widget.widget.normal = event.new
            self.cut.filters[0].widget.normal = event.new
        self.scene.mlab.draw()

    @observe('show_full_model', post_init=True)
    def toggle_full_model(self, event):
        """
        Toggle between showing the full spinal cord model and showing only below the cut plane.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for show_full_model.
        """
        self.csf_surface.visible = not event.new
        self.full_csf_surface.visible = event.new

        self.scene.mlab.draw()

    def reset_participant_defaults(self):
        self.reset_traits(traits=['csf_model', 'show_full_model', 'log_scale', 'normal', 'origin'])

    @observe('csf_model', post_init=True)
    def change_cord_model(self, event):
        """
        Change the spinal cord model file used for the 3D display.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for csf_model
        """
        if self.csf_model_reader is not None:
            self.csf_model_reader.initialize(event.new)

    @observe('scene.activated')
    def initialize_camera(self, event=None):  # pylint: disable=unused-argument
        """
        Set the camera for the Mayavi scene to a pre-determined perspective.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for scene.activated
        """
        if self.csf_surface is not None:
            self.scene.engine.current_object = self.csf_surface
        self.scene.mlab.view(azimuth=-35, elevation=75)

        self.scene.mlab.draw()

    def create_plot(self):
        """
        Create the 3D objects to be shown.
        """
        normal = self.normal

        max_ind = np.unravel_index(np.nanargmax(self.fields_model.masked_grid_data),
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

    @observe(
            ob.trait('points').list_items().trait('value', optional=True).list_items(optional=True),
            post_init=True)
    def draw_line(self, event):
        """
        Create or update the line described by the points in :ref:`line-attributes`.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for points.
        """
        if None in event.new and len(event.old) == len(event.new) and None not in event.old:
            self.points = event.old
            return
        points = np.array(
                [val.value if val is not None else np.array([0, 0, 0]) for val in self.points])

        x_positions = []
        y_positions = []
        z_positions = []

        for point in points:
            x_positions.append(point[0])
            y_positions.append(point[1])
            z_positions.append(point[2])

        if not hasattr(self.line, 'mlab_source'):
            self.line = self.scene.mlab.plot3d(x_positions, y_positions, z_positions,
                                               tube_radius=0.2, color=(1, 0, 0),
                                               figure=self.scene.mayavi_scene)
        else:
            self.line.mlab_source.reset(x=x_positions, y=y_positions, z=z_positions)

        self.scene.mlab.draw()

    def disable_widgets(self):
        """
        Disable widgets to be hidden and set up color properties.
        """
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

    def _csf_model_default(self):
        try:
            model = self._get_default_value('csf_model')
        except KeyError:
            model = os.path.join(os.getcwd(), 'CSF.vtk')
        return model

    def _show_full_model_default(self):
        full_model = self.configuration.BOOLEAN_STATES[self._get_default_value('full_model').lower()]
        return full_model

    def _log_scale_default(self):
        log_scale = self.configuration.BOOLEAN_STATES[self._get_default_value('log_scale').lower()]
        return log_scale

    def _normal_default(self):
        normal = self._get_default_value('normal')
        return np.fromstring(normal.strip('()'), sep=',')
    
    def _origin_default(self):
        origin = self._get_default_value('origin')
        return np.fromstring(origin.strip('()'), sep=',')

    def _get_default_value(self, option):
        if self.participant_id is not None:
            if self.participant_id not in self.configuration:
                self.configuration[self.participant_id] = {}
            val = self.configuration[self.participant_id][option]
        else:
            val = self.configuration[self.participant_id][option]
        return val


@provides(IEditor)
class SliceFigureModel(Editor):
    """
    A Pyface Tasks Editor to hold the slice figure.
    """
    #: The model object to view. If not specified, the editor is used instead.
    model = Instance(HasTraits)

    #: The UI object associated with the Traits view, if it has been
    #: constructed.
    ui = Instance("traitsui.ui.UI")

    #: The editor's user-visible name.
    name = Str("Slice Plane")

    #: Configuration parser.
    configuration = Instance(ConfigParser)

    #: Current participant ID.
    participant_id = Str()

    #: The :py:class:`EMFields` instance containing the field data.
    fields_model = Instance(EMFields)

    #: The :py:class:`Mayavi3DScene` instance containing the 3D plot.
    mayavi_scene = Instance(Mayavi3DScene)

    #: The :py:class:`matplotlib.figure.Figure` containing the slice figure.
    figure = Instance(Figure, ())

    #: Use a logarithmic scale for the field data?
    log_scale = Bool()

    #: Matplotlib colormap.
    mycmap = Any()

    #: :py:class:`matplotlib.colors.Normalize` instance for the
    #: field data.
    norm = Any()

    #: :py:class:`matplotlib.collections.QuadMesh` containing the plot data.
    pcm = Any()

    #: :py:class:`matplotlib.colorbar.Colorbar` containing the figure's
    #: colorbar.
    clb = Any()

    #: Draw the line cross marker?
    draw_cross = Bool()

    #: The list of points describing the line for the line figure.
    points = DelegatesTo('mayavi_scene')

    #: List of :py:class:`matplotlib.lines.Line2D` representing the line
    #: cross marker
    line_cross = Any()

    #: The slice figure title.
    figure_title = Str()

    #: Data label.
    data_label = Str()

    #: Use user-input labels?
    use_custom_label = Bool(False)

    def default_traits_view(self):  # pylint: disable=no-self-use
        """
        Create the default traits View object for the model

        Returns
        -------
        default_traits_view : :py:class:`traitsui.view.View`
            The default traits View object for the model
        """
        return View(
                Group(
                        Item('figure', editor=MPLFigureEditor(), show_label=False),
                )
        )

    def create(self, parent):
        """
        Create and set the widget(s) for the Editor.

        Parameters
        ----------
        parent : toolkit-specific widget
            The parent widget for the Editor
        """
        self.ui = self.edit_traits(kind='subpanel', parent=parent)  # pylint: disable=invalid-name
        self.control = self.ui.control  # pylint: disable=attribute-defined-outside-init

    def destroy(self):
        """
        Destroy the Editor and clean up after
        """
        self.control = None  # pylint: disable=attribute-defined-outside-init
        if self.ui is not None:
            self.ui.dispose()
        self.ui = None

    def export_slice(self, file_path):
        """
        Export data for slice to Excel or CSV file.

        Parameters
        ----------
        file_path : os.PathLike
            Path to output file
        """
        x_positions, y_positions, data = self._calculate_plane()

        out_df = pd.DataFrame(data=data, index=y_positions, columns=x_positions)

        if file_path.endswith('.xlsx'):
            out_df.to_excel(file_path)
        elif file_path.endswith('.csv') or file_path.endswith('.txt'):
            out_df.to_csv(file_path)

    @observe('log_scale', post_init=True)
    def toggle_log_scale(self, event):
        """
        Toggle between using a logarithmic scale and a linear scale

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for log_scale
        """
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
        """
        Create the slice figure plot.
        """
        true_x, true_y, true_data = self._calculate_plane()
        self.mycmap = copy.copy(cm.get_cmap('jet'))

        self.figure.clear()

        axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(right=0.75, bottom=0.15)

        if self.log_scale:
            self.norm = LogNorm(vmin=np.nanmin(self.fields_model.masked_grid_data),
                                vmax=np.nanmax(self.fields_model.masked_grid_data))
        else:
            self.norm = Normalize(vmin=np.nanmin(self.fields_model.masked_grid_data),
                                  vmax=np.nanmax(self.fields_model.masked_grid_data))

        self.pcm = axes.pcolormesh(true_x, true_y, true_data, shading='nearest', cmap=self.mycmap,
                                   norm=self.norm)

        if self.draw_cross:
            self.line_cross = axes.plot([0], [0], 'rx')
        else:
            self.line_cross = axes.plot([0], [0], '')

        axes.set_ylim(bottom=np.nanmin(self.fields_model.masked_gr_y[0, :, 0]) - 2,
                      top=np.nanmax(self.fields_model.masked_gr_y[0, :, 0]) + 2)

        axes.set_xlim(left=np.nanmin(self.fields_model.masked_gr_x[:, 0, 0]) - 2,
                      right=np.nanmax(self.fields_model.masked_gr_x[:, 0, 0]) + 2)

        axes.set_xlabel('X (mm)')
        axes.set_ylabel('Y (mm)')

        axes.set_title(self.figure_title)

        self.clb = self.figure.colorbar(cm.ScalarMappable(norm=self.norm, cmap=self.mycmap),
                                        ax=axes)
        self.clb.set_label(self.data_label, rotation=270, labelpad=15)

        self.figure.canvas.draw()

    @observe('fields_model.selected_field_key')
    def _set_figure_labels(self, event):
        if not self.use_custom_label:
            if self.fields_model.selected_field_key.startswith('J'):
                self.figure_title = 'Current Density Magnitude'
                self.data_label = 'Current Density ($A/m^2$)'
            elif self.fields_model.selected_field_key.startswith('EM_E'):
                self.figure_title = 'Electric Field Magnitude'
                self.data_label = 'Electric Field ($V/m$)'
            elif self.fields_model.selected_field_key.startswith('D'):
                self.figure_title = 'Displacement Flux Density Magnitude'
                self.data_label = 'Displacement Flux Density ($C/m^2$)'

            if self.line_cross is not None:
                self.update_line_cross(event=event)
            if self.pcm is not None:
                self.update_plot(event=event)

    def _calculate_plane(self):
        dataset = self.mayavi_scene.cut.outputs[0].output
        datax, datay, dataz = dataset.points.to_array().T  # pylint: disable=unused-variable
        scalar_data = dataset.point_data.scalars.to_array()

        true_x = np.unique(np.floor(datax / SCALE_FACTOR).astype(int)) * SCALE_FACTOR
        true_y = np.unique(np.floor(datay / SCALE_FACTOR).astype(int)) * SCALE_FACTOR

        true_data = np.empty((true_y.size, true_x.size))
        true_data[:] = np.nan

        for i in range(scalar_data.size):
            true_data[arg_find_nearest(true_y, datay[i]), arg_find_nearest(true_x, datax[i])] =\
                scalar_data[i]

        return true_x, true_y, true_data

    @observe('mayavi_scene:origin')
    @observe('mayavi_scene:normal')
    def update_plot(self, event):  # pylint: disable=unused-argument
        """
        Update the slice figure when the cut plane origin or normal are changed.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for the cut plane origin or normal
        """
        true_x, true_y, true_data = self._calculate_plane()

        self.update_line_cross()

        axes = self.figure.axes[0]
        self.pcm.remove()
        self.pcm = axes.pcolormesh(true_x, true_y, true_data, shading='nearest', cmap=self.mycmap,
                                   norm=self.norm)

        axes.set_title(self.figure_title)
        self.clb.set_label(self.data_label, rotation=270, labelpad=15)

        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()

    @observe('draw_cross', post_init=True)
    @observe(
            ob.trait('points').list_items().trait('value', optional=True).list_items(optional=True))
    def update_line_cross(self, event=None):  # pylint: disable=too-many-locals, unused-argument
        """
        Update the location of the line cross marker when draw_cross is changed or when the points
        are changed.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for draw_cross or points
        """
        if not self.draw_cross and self.line_cross is not None:
            self.line_cross[0].set_marker('')
        elif self.line_cross is not None:
            normal_x, normal_y, normal_z = self.mayavi_scene.normal
            origin_x, origin_y, origin_z = self.mayavi_scene.origin
            plane_z_0 = -1 * (normal_x * origin_x + normal_y * origin_y) / normal_z + origin_z

            points = [val.value if val is not None else np.array([0, 0, 0]) for val in self.points]

            p_under = [val for val in points if val[2] <= plane_z_0]
            p_over = [val for val in points if val[2] > plane_z_0]

            if len(p_under) > 0 and len(p_over) > 0:
                point_1 = p_under[-1]
                point_2 = p_over[0]

                parametric_t = (-1 * normal_x * point_1[0]
                                - normal_y * point_1[1]
                                - normal_z * point_1[2]
                                + normal_x * origin_x
                                + normal_y * origin_y
                                + normal_z * origin_z) /\
                               (normal_x * (point_2[0] - point_1[0])
                                + normal_y * (point_2[1] - point_1[1])
                                + normal_z * (point_2[2] - point_1[2]))
                x_position = point_1[0] + parametric_t * (point_2[0] - point_1[0])
                y_position = point_1[1] + parametric_t * (point_2[1] - point_1[1])

                self.line_cross[0].set_data([x_position], [y_position])
                self.line_cross[0].set_marker('x')

        self.figure.canvas.draw()

    def _log_scale_default(self):
        log_scale = self.configuration.BOOLEAN_STATES[self._get_default_value('log_scale').lower()]
        return log_scale

    def _draw_cross_default(self):
        line_cross = self.configuration.BOOLEAN_STATES[self._get_default_value('line_cross_marker').lower()]
        return line_cross

    def _get_default_value(self, option):
        if self.participant_id is not None:
            if self.participant_id not in self.configuration:
                self.configuration[self.participant_id] = {}
            val = self.configuration[self.participant_id][option]
        else:
            val = self.configuration[self.participant_id][option]
        return val


@provides(IEditor)
class LineFigureModel(Editor):
    """
    A Pyface Traits Editor to hold the line figure.
    """
    #: The model object to view. If not specified, the editor is used instead.
    model = Instance(HasTraits)

    #: The UI object associated with the Traits view, if it has been
    #: constructed.
    ui = Instance("traitsui.ui.UI")

    #: The editor's user-visible name.
    name = Str("Line Plot")

    #: The :py:class:`EMFields` instance containing the field data.
    fields_model = Instance(EMFields)

    #: The :py:class:`matplotlib.figure.Figure` containing the line figure.
    figure = Instance(Figure, ())

    #: The list of points describing the line for the line figure.
    points = List(ArrayClass, value=[ArrayClass(value=np.array([0, 0, -1])),
                                     ArrayClass(value=np.array([0, 0, 1]))])

    #: The interpolation function to sample the field data for the line
    #: figure.
    interp_func = Instance(RegularGridInterpolator)

    #: The line figure title.
    figure_title = Str()

    #: The line figure x-axis label.
    x_axis_label = Str('Distance Along Line (mm)')

    #: The line figure y-axis label.
    y_axis_label = Str()

    #: Use user-input labels?
    use_custom_label = Bool(False)

    def default_traits_view(self):  # pylint: disable=no-self-use
        """
        Create the default traits View object for the model

        Returns
        -------
        default_traits_view : :py:class:`traitsui.view.View`
            The default traits View object for the model
        """
        return View(
                Group(
                        Item('figure', editor=MPLFigureEditor(), show_label=False),
                )
        )

    def create(self, parent):
        """
        Create and set the widget(s) for the Editor.

        Parameters
        ----------
        parent : toolkit-specific widget
            The parent widget for the Editor
        """
        self.ui = self.edit_traits(kind='subpanel', parent=parent)  # pylint: disable=invalid-name
        self.control = self.ui.control

    def destroy(self):
        """
        Destroy the Editor and clean up after
        """
        self.control = None
        if self.ui is not None:
            self.ui.dispose()
        self.ui = None

    def export_line(self, file_path):
        """
        Export data for line to Excel or CSV file.

        Parameters
        ----------
        file_path : os.PathLike
            Path to output file
        """
        line_pos, line_data = self._fill_data()

        out_df = pd.DataFrame(data=line_data, index=line_pos, columns=['y'])

        if file_path.endswith('.xlsx'):
            out_df.to_excel(file_path)
        elif file_path.endswith('.csv') or file_path.endswith('.txt'):
            out_df.to_csv(file_path)

    def _calculate_between_pair(self, point_1, point_2, prev_line_pos=0):
        x_positions = np.linspace(point_1[0], point_2[0], 1000)
        y_positions = np.linspace(point_1[1], point_2[1], 1000)
        z_positions = np.linspace(point_1[2], point_2[2], 1000)

        line_pos = np.linspace(0, np.linalg.norm(point_2 - point_1), 1000) + prev_line_pos
        line_data = self.interp_func((x_positions, y_positions, z_positions))

        return line_pos, line_data

    def _interp_func_default(self):
        func = RegularGridInterpolator(
                (
                        self.fields_model.x_vals,
                        self.fields_model.y_vals,
                        self.fields_model.z_vals
                ),
                np.nan_to_num(self.fields_model.data_arr),
                bounds_error=False,
                fill_value=None
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

    @observe(
            ob.trait('points').list_items().trait('value', optional=True).list_items(optional=True),
            post_init=True)
    def create_plot(self, event):
        """
        Create or update the line figure.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for points
        """
        line_pos, line_data = self._fill_data()

        self.figure.clear()

        axes = self.figure.add_subplot(111)
        axes.plot(line_pos, line_data)
        axes.set_ylabel(self.y_axis_label)
        axes.set_xlabel(self.x_axis_label)

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


@provides(IEditor)
class StartPage(Editor):
    """
    A Pyface Tasks Editor to hold the opening page
    """
    #: The model object to view. If not specified, the editor is used instead.
    model = Instance(HasTraits)

    #: The UI object associated with the Traits view, if it has been
    #: constructed.
    ui = Instance('traitsui.ui.UI')

    #: The editor's user-visible name.
    name = Str('Start Page')

    #: The task associated with the editor.
    task = Any()

    #: Button to open a new data file.
    open_data_file_button = Button(label='Open Data File', style='button')

    def default_traits_view(self):  # pylint: disable=no-self-use
        """
        Create the default traits View object for the model

        Returns
        -------
        default_traits_view : :py:class:`traitsui.view.View`
            The default traits View object for the model
        """
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
        """
        Create and set the widget(s) for the Editor.

        Parameters
        ----------
        parent : toolkit-specific widget
            The parent widget for the Editor
        """
        self.ui = self.edit_traits(kind='subpanel', parent=parent) # pylint: disable=invalid-name
        self.control = self.ui.control

    def destroy(self):
        """
        Destroy the Editor and clean up after
        """
        self.control = None
        if self.ui is not None:
            self.ui.dispose()
        self.ui = None

    @observe('open_data_file_button', post_init=True)
    def open(self, event):
        """
        Open new data file.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for open_data_file_button
        """
        self.task.open()
