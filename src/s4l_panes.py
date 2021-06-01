"""
Dock Panes for the S4L Visualization application.
"""
# pylint: disable=unused-argument, too-many-instance-attributes
import re
from configparser import ConfigParser

import numpy as np
from numpy import ma
from pyface.api import FileDialog, OK
from pyface.tasks.api import TraitsDockPane
from pyface.ui.qt4.tasks.dock_pane import INVERSE_AREA_MAP
from traits.api import Str, Int, Enum, Array, observe, List, Instance, Button
from traitsui.api import View, Item, Group, EnumEditor, ArrayEditor, ListEditor, InstanceEditor, Spring
from traitsui.key_bindings import KeyBindings, KeyBinding

from .q_range_editor import QRangeEditor
from .s4l_models import EMFields
from .util import ArrayClass


class CustomDockPane(TraitsDockPane):
    """
    Overridden to fix error when dragging a dock pane
    """

    # pylint: disable=attribute-defined-outside-init
    def _receive_dock_area(self, area):
        with self._signal_context():
            if int(area) in INVERSE_AREA_MAP.keys():
                self.dock_area = INVERSE_AREA_MAP[int(area)]


class PlaneAttributes(CustomDockPane):
    """ A pane containing attributes describing a plane in 3d space
    """

    # ------------------------------------------------------------------------
    # TaskPane interface.
    # ------------------------------------------------------------------------

    #: The dock pane's identifier.
    id = 's4l.plane_attributes'

    #: The dock pane's user-visible name.
    name = 'Plane Attributes'

    # ------------------------------------------------------------------------
    # PlaneAttributes interface.
    # ------------------------------------------------------------------------

    #: Configuration parser.
    configuration = Instance(ConfigParser)

    #: Current participant ID.
    participant_id = Str()

    #: The :py:class:`EMFields` instance containing the field data.
    fields_model = Instance(EMFields)

    #: Index of plane location along the slicing direction.
    slice_coord_index = Int(0)

    #: Index of slicing direction (x=0, y=1, z=2).
    dir_idx = Int(2)

    #: Index of min value to show on slider.
    coord_low_index = Int(0)

    #: Index of max value to show on slider.
    coord_high_index = Int(0)

    #: Map of coord indices to coord labels.
    coord_map = Array(value=np.array([0]))

    #: Label of slicing direction.
    coord_label = Str('Z')

    #: Label of min value on slider.
    low_label = Str()

    #: Label of max value on slider.
    high_label = Str()

    #: Type of plane. One of
    #: {"Normal to X", "Normal to Y", "Normal to Z", "Arbitrary Plane"}
    #: Defaults to "Normal to Z".
    plane_type = Enum('Normal to Z',
                      'Normal to X',
                      'Normal to Y',
                      'Arbitrary Plane')

    #: Normal vector of plane.
    normal = Array(dtype=np.float)

    #: Origin point of plane.
    origin = Array(dtype=np.float)

    @observe('plane_type', post_init=True)
    def change_plane_type(self, event):
        """
        Change normal vector when new plane_type is selected.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for plane_type
        """
        if event.new == 'Normal to X':
            self.normal = np.array([1, 0, 0])
            self.dir_idx = 0
        elif event.new == 'Normal to Y':
            self.normal = np.array([0, 1, 0])
            self.dir_idx = 1
        elif event.new == 'Normal to Z':
            self.normal = np.array([0, 0, 1])
            self.dir_idx = 2
        elif event.new == 'Arbitrary Plane':
            self.dir_idx = 2
        self.update_slider_bounds(None)

    @observe('fields_model.masked_gr_z')
    def update_slider_bounds(self, event):
        """
        Update range slider when grid points are updated.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for fields_model.masked_gr_z
        """
        max_ind = np.unravel_index(np.nanargmax(self.fields_model.masked_grid_data),
                                   self.fields_model.masked_grid_data.shape)
        if self.plane_type == 'Normal to X':
            self.coord_label = 'X'
            self.low_label = '{:.2f} mm'.format(self.fields_model.masked_gr_x[0, 0, 0])
            self.high_label = '{:.2f} mm'.format(self.fields_model.masked_gr_x[-1, 0, 0])
            self.coord_map = self.fields_model.masked_gr_x[:, 0, 0]
            self.slice_coord_index = max_ind[0]
        elif self.plane_type == 'Normal to Y':
            self.coord_label = 'Y'
            self.low_label = '{:.2f} mm'.format(self.fields_model.masked_gr_y[0, 0, 0])
            self.high_label = '{:.2f} mm'.format(self.fields_model.masked_gr_y[0, -1, 0])
            self.coord_map = self.fields_model.masked_gr_y[0, :, 0]
            self.slice_coord_index = max_ind[1]
        else:
            self.coord_label = 'Z'
            self.low_label = '{:.2f} mm'.format(self.fields_model.masked_gr_z[0, 0, 0])
            self.high_label = '{:.2f} mm'.format(self.fields_model.masked_gr_z[0, 0, -1])
            self.coord_map = self.fields_model.masked_gr_z[0, 0, :]
            self.slice_coord_index = max_ind[2]

        self.coord_high_index = len(self.coord_map) - 1

    @observe('slice_coord_index', post_init=True)
    def update_coord(self, event):
        """
        Update origin when slider is changed.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change event for slice_coord_index
        """
        coord = self.coord_map[event.new]

        if (self.plane_type == 'Normal to Z' or self.plane_type == 'Arbitrary Plane') and\
                self.origin[2] != coord:
            self.origin = np.array([self.origin[0], self.origin[1], coord])
        elif self.plane_type == 'Normal to Y' and self.origin[1] != coord:
            self.origin = np.array([self.origin[0], coord, self.origin[2]])
        elif self.plane_type == 'Normal to X' and self.origin[0] != coord:
            self.origin = np.array([coord, self.origin[1], self.origin[2]])

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
                        Group(
                                Item(
                                        'slice_coord_index',
                                        label=self.coord_label,
                                        editor=QRangeEditor(
                                                low_name='coord_low_index',
                                                high_name='coord_high_index',
                                                low_label_name='low_label',
                                                high_label_name='high_label',
                                                map_to_values_name='coord_map',
                                                mode='slider',
                                                is_float=False,
                                        ),
                                        padding=15,
                                ),
                        ),
                        Item(
                                'plane_type',
                                editor=EnumEditor(
                                        values={
                                                'Normal to X':     '1:Normal to X',
                                                'Normal to Y':     '2:Normal to Y',
                                                'Normal to Z':     '3:Normal to Z',
                                                'Arbitrary Plane': '4:Arbitrary Plane',
                                        },
                                        format_func=str,
                                        cols=4
                                ),
                                style='custom',
                                show_label=False
                        ),
                        Group(
                                Item('normal', editor=ArrayEditor(width=-60)),
                                Item('origin', editor=ArrayEditor(width=-60)),
                                visible_when='plane_type == "Arbitrary Plane"'
                        )
                )
        )

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


class LineAttributes(CustomDockPane):
    """ A pane containing attributes describing a line in 3d space
    """

    # ------------------------------------------------------------------------
    # TaskPane interface.
    # ------------------------------------------------------------------------

    #: The dock pane's identifier.
    id = 's4l.line_attributes'

    #: The dock pane's user-visible name.
    name = 'Line Attributes'

    # ------------------------------------------------------------------------
    # LineAttributes interface.
    # ------------------------------------------------------------------------

    #: Configuration parser.
    configuration = Instance(ConfigParser)

    #: Current Participant ID
    participant_id = Str()

    #: The list of points describing the line for the line figure.
    points = List(ArrayClass, value=[ArrayClass(value=np.array([0, 0, -1])),
                                     ArrayClass(value=np.array([0, 0, 1]))])

    #: Button to sort list of points.
    sort_points_button = Button(label="Sort", style='button', visible_when='len(points) > 2')

    #: Button to import points from a file.
    import_button = Button(label='Import', style='button')

    #: Button to export points to a file.
    export_button = Button(label='Export', style='button')

    @observe('export_button', post_init=True)
    def export_points(self, event):
        """
        Export points from a file.
        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change handler for export_button.
        """
        point_string = str([tuple(x.value) for x in self.points])

        dialog = FileDialog(
                title='Export Line Points',
                action='save as',
                parent=self.task.window.control,
                wildcard='' + FileDialog.create_wildcard('Text Files', ['*.txt']) + FileDialog.WILDCARD_ALL
        )
        if dialog.open() == OK:
            with open(dialog.path, 'w') as out_file:
                out_file.write(point_string)

    @observe('import_button', post_init=True)
    def import_points(self, event):
        """
        Import points from a file.
        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change handler for import_button.
        """
        dialog = FileDialog(
                title='Import Line Points',
                action='open',
                parent=self.task.window.control,
                wildcard='' + FileDialog.create_wildcard('Text Files', ['*.txt']) + FileDialog.WILDCARD_ALL
        )
        if dialog.open() == OK:
            with open(dialog.path, 'r') as in_file:
                point_string = in_file.read()

            pat = r'(\([^\)]+\))'

            new_points = [np.array([float(s) for s in x.strip("()").split(',')]) for x in re.findall(pat, point_string)]
            new_points = [ArrayClass(value=arr) for arr in new_points]

            self.points = new_points

    @observe('sort_points_button', post_init=True)
    def sort_points(self, event):
        """
        Sort points starting from first point by least distance to previous point.

        Parameters
        ----------
        event : A :py:class:`traits.observation.events.TraitChangeEvent` instance
            The trait change handler for sort_points_button.
        """
        if len(self.points) > 0:
            points = [val if val else ArrayClass() for val in self.points]
            distances = np.zeros((len(points), len(points)))
            for i, point1 in enumerate(points):
                for j, point2 in enumerate(points):
                    distances[i, j] = np.linalg.norm(point1.value - point2.value)

            point_order = [0]
            while len(point_order) < len(points):
                masked = ma.array(distances[point_order[-1], :])
                masked[point_order] = ma.masked
                point_order.extend(ma.where(masked == masked.min())[0])

            tmp_points = [points[i] for i in point_order]

            self.points = tmp_points

    @observe('points.items', post_init=True)
    def _new_point_added(self, event):
        points = [val if val is not None else ArrayClass(value=np.array([0, 0, 0])) for val in
                  self.points]
        if points != self.points:
            self.points = points

    def set_participant_defaults(self):
        """
        Set points to the current participant's default points value from config files.
        """
        if self.participant_id is not None:
            if self.participant_id not in self.configuration:
                self.configuration[self.participant_id] = {}
            default_points = self.configuration[self.participant_id]['points']
        else:
            default_points = self.configuration['DEFAULT']['points']

        self.set_points(default_points)

    def set_points(self, point_string):
        """
        Set points from a string.
        Parameters
        ----------
        point_string : str
            String representing a list of points with the format [(x1, y1, z1), (x2, y2, z2), ...]
        """
        pat = r'(\([^\)]+\))'

        new_points = [np.array([float(s) for s in x.strip("()").split(',')]) for x in re.findall(pat, point_string)]
        new_points = [ArrayClass(value=arr) for arr in new_points]

        self.points = new_points

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
                        Group(
                                Item('import_button', show_label=False),
                                Item('export_button', show_label=False),
                                orientation='horizontal'
                        ),
                        Item('sort_points_button', show_label=False),
                        Item('points',
                             editor=ListEditor(editor=InstanceEditor(), scrollable=False, style='custom'),
                             show_label=False),
                        orientation='vertical'
                )
        )


class ParticipantIDPane(CustomDockPane):
    # ------------------------------------------------------------------------
    # TaskPane interface.
    # ------------------------------------------------------------------------

    #: The dock pane's identifier.
    id = 's4l.participant_id_pane'

    #: The dock pane's user-visible name.
    name = 'Participant ID'

    # ------------------------------------------------------------------------
    # LineAttributes interface.
    # ------------------------------------------------------------------------

    #: Current participant ID.
    participant_id = Str()

    #: ID trait for user input.
    part_box = Str()

    #: Button to set current participant ID to value in box.
    set_id_button = Button(label='Set ID',
                           style='button')

    #: Key bindings for the pane - 'Enter' = clicking set_id_button
    key_bindings = KeyBindings(
            KeyBinding(
                    binding1='Enter',
                    description='Set ID',
                    method_name='set_current_id',
            ),
    )

    @observe('set_id_button', post_init=True)
    def set_current_id(self, event):
        self.participant_id = self.part_box
        self.task.update_participant_id()

    def default_traits_view(self):
        """
        """
        return View(
                Group(
                        Spring(),
                        Item(label="Current Participant ID", style='custom', emphasized=True),
                        Spring(),
                        orientation='horizontal'
                ),
                Group(
                        Spring(),
                        Item('participant_id',
                             show_label=False,
                             style='readonly',
                             visible_when='participant_id != ""'),
                        Item(label="N/A", style='readonly', visible_when='participant_id == ""'),
                        Spring(),
                        orientation='horizontal'
                ),
                Item('_'),
                Group(
                        Item('part_box', label='Participant ID'),
                        Item('set_id_button',
                             show_label=False,
                             tooltip='Sets the current participant ID and resets values to default'),
                        orientation='horizontal'
                ),
                key_bindings=self.key_bindings
        )
