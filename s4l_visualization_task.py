import numpy as np
import pandas as pd
from pyface.tasks.action.task_action import DockPaneAction
from pyface.tasks.task_layout import Splitter, Tabbed

from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator

from pyface.qt import QtCore, QtGui
from pyface.tasks.api import (
    Task,
    TaskLayout,
    VSplitter,
    PaneItem,
    IEditor,
    IEditorAreaPane,
    SplitEditorAreaPane
)
from pyface.tasks.action.api import (
    DockPaneToggleGroup,
    SMenuBar,
    SMenu,
    TaskAction,
)
from pyface.api import (
    FileDialog,
    OK
)
from traits.api import Property, Instance, Array, Int, observe, DelegatesTo
from traits.trait_types import Bool

from s4l_groups import FieldSelectionGroup
from s4l_panes import PlaneAttributes, LineAttributes
from s4l_models import EMFields, Mayavi3DScene, SliceFigureModel, LineFigureModel, StartPage


class S4LVisualizationTask(Task):
    """ A task for visualizing Sim4Life EM fields from scES simulations
    """

    id = "s4l.main_task"
    name = "S4L Visualization"

    # Plane attributes dock pane
    plane_attributes_pane = Instance(PlaneAttributes)

    # Line attributes dock pane
    line_attributes_pane = Instance(LineAttributes)

    active_editor = Property(
        Instance(IEditor), depends_on="editor_area.active_editor"
    )

    editor_area = Instance(IEditorAreaPane)

    start_page = Instance(StartPage)
    fields_model = Instance(EMFields)
    mayavi_scene = Instance(Mayavi3DScene)

    model_initialized = Bool(False)

    toggle_model_action = TaskAction(name='Full Model',
                                     method='toggle_full_model',
                                     style='toggle',
                                     enabled_name='model_initialized')

    new_cord_action = TaskAction(name='New Cord Model',
                                 method='change_cord_model',
                                 enabled_name='model_initialized')

    toggle_scale_action = TaskAction(name='Log Scale',
                                     method='toggle_log_scale',
                                     style='toggle',
                                     checked=True,
                                     enabled_name='model_initialized')

    menu_bar = SMenuBar(
        SMenu(
            TaskAction(name="Open...", method="open", accelerator="Ctrl+O"),
            SMenu(
                TaskAction(name="Export Slice", method="export_slice"),
                TaskAction(name="Export Line", method="export_line"),
                id="File.Export",
                name="&Export",
            ),
            id="File",
            name="&File",
        ),
        SMenu(
            SMenu(
                DockPaneToggleGroup(),
                id='View.Panes',
                name='&Panes'
            ),
            toggle_model_action,
            toggle_scale_action,
            id="View",
            name="&View"
        ),
        SMenu(
            new_cord_action,
            SMenu(
                FieldSelectionGroup(),
                id='Edit.Fields',
                name='&Choose Field'
            ),
            id='Edit',
            name='&Edit',
        ),
    )

    # ------------------------------------------------------------------------
    # 'Task' interface.
    # ------------------------------------------------------------------------

    @observe('editor_area.active_tabwidget')
    def _update_tabwidgets(self, event):
        try:
            self.editor_area.active_tabwidget.setTabsClosable(False)
        except AttributeError:
            pass

    def initialized(self):
        for tabwidget in self.editor_area.tabwidgets():
            tabwidget.setTabsClosable(False)
        self.start_page = StartPage(task=self)
        self.editor_area.add_editor(self.start_page)
        self.editor_area.activate_editor(self.start_page)
        self.activated()

    def activated(self):
        self.editor_area.active_tabwidget.setTabsClosable(False)

    def create_central_pane(self):
        self.editor_area = SplitEditorAreaPane(callbacks={'open': self._new_file})
        return self.editor_area

    def create_dock_panes(self):
        """ Create the attribute editor panes
        """

        self.plane_attributes_pane = PlaneAttributes()
        self.line_attributes_pane = LineAttributes()

        return [self.plane_attributes_pane, self.line_attributes_pane]

    # ------------------------------------------------------------------------
    # 'S4L_Visualization_task' interface.
    # ------------------------------------------------------------------------

    def open(self):
        """ Shows a dialog to open a new data source.
        """
        dialog = FileDialog(
            title='Choose Data File',
            parent=self.window.control,
            wildcard=FileDialog.create_wildcard('Data Files', ['*.mat']) + FileDialog.WILDCARD_ALL
        )
        if dialog.open() == OK:
            if not self.model_initialized:
                self._new_file(dialog.path)

    def export_slice(self):
        """ Exports data for current slice
        """
        dialog = FileDialog(
            title='Export Slice Plane',
            action='save as',
            parent=self.window.control,
            wildcard='' + FileDialog.create_wildcard('Excel Files', ['*.xlsx'])
                        + FileDialog.create_wildcard('CSV Files', ['*.csv', '*.txt'])
                        + FileDialog.WILDCARD_ALL
        )
        if dialog.open() == OK:
            self.slice_figure.export_slice(dialog.path)

    def export_line(self):
        """ Exports data for current line
        """
        dialog = FileDialog(
            title='Export Line Data',
            action='save as',
            parent=self.window.control,
            wildcard='' + FileDialog.create_wildcard('Excel Files', ['*.xlsx'])
                     + FileDialog.create_wildcard('CSV Files', ['*.csv', '*.txt'])
                     + FileDialog.WILDCARD_ALL
        )
        if dialog.open() == OK:
            self.line_figure.export_line(dialog.path)

    def toggle_full_model(self):
        self.mayavi_scene.show_full_model = not self.mayavi_scene.show_full_model

    def toggle_log_scale(self):
        self.mayavi_scene.log_scale = not self.mayavi_scene.log_scale
        self.slice_figure.log_scale = not self.slice_figure.log_scale

    def change_cord_model(self):
        dialog = FileDialog(
            title='Choose Spinal Cord Model',
            parent=self.window.control,
            wildcard=FileDialog.create_wildcard('VTK Model', ['*.vtk']) + FileDialog.WILDCARD_ALL
        )
        if dialog.open() == OK:
            self.mayavi_scene.csf_model = dialog.path

    def reset_camera(self):
        self.mayavi_scene.initialize_camera()

    # ------------------------------------------------------------------------
    # Protected interface.
    # ------------------------------------------------------------------------

    def _new_file(self, filename):
        """
        Changes the data source to the file at the specified path
        :param filename: path to data source file
        """
        self.editor_area.remove_editor(self.start_page)

        self.window.set_layout(
                TaskLayout(
                        bottom=PaneItem('s4l.plane_attributes'),
                        left=PaneItem('s4l.line_attributes'),
                        top_left_corner='left',
                        top_right_corner='right',
                        bottom_left_corner='left',
                        bottom_right_corner='right'
                )
        )

        self.fields_model = EMFields(data_path=filename)
        self.plane_attributes_pane.fields_model = self.fields_model

        self.mayavi_scene = Mayavi3DScene(fields_model=self.fields_model)

        self.mayavi_scene.sync_trait('normal', self.plane_attributes_pane)
        self.mayavi_scene.sync_trait('origin', self.plane_attributes_pane)
        self.line_attributes_pane.sync_trait('points', self.mayavi_scene, mutual=False)

        self.mayavi_scene.create_plot()
        editor = self.mayavi_scene

        self.editor_area.add_editor(editor)
        self.editor_area.activate_editor(editor)
        self.editor_area.active_tabwidget.setTabsClosable(False)
        self.activated()

        self.slice_figure = SliceFigureModel(fields_model=self.fields_model, mayavi_scene=self.mayavi_scene)

        self.slice_figure.create_plot()

        self.editor_area.add_editor(self.slice_figure)
        self.editor_area.activate_editor(self.slice_figure)
        self.editor_area.active_tabwidget.setTabsClosable(False)
        self.activated()

        self.line_figure = LineFigureModel(fields_model=self.fields_model)
        self.line_figure.sync_trait('points', self.line_attributes_pane)

        self.line_figure.create_plot(None)

        self.editor_area.add_editor(self.line_figure)
        self.editor_area.activate_editor(self.line_figure)
        self.editor_area.active_tabwidget.setTabsClosable(False)
        self.activated()
        # self.editor_area.control.split()

        self.mayavi_scene.disable_widgets()

        while self.editor_area.active_tabwidget.parent().is_collapsible():
            self.editor_area.active_tabwidget.parent().collapse()

        layout = Splitter(
            Tabbed(
                PaneItem(1),
                PaneItem(2),
                active_tab=0
            ),
            Tabbed(PaneItem(0), active_tab=0),
        )

        self.editor_area.set_layout(layout)
        self.editor_area.control.setSizes([900, 295])
        self.editor_area.activate_editor(self.slice_figure)
        self.editor_area.active_tabwidget.setTabsClosable(False)

        for tabwidget in self.editor_area.tabwidgets():
            tabwidget.setTabsClosable(False)

        self.model_initialized = True

    @observe('window:closing')
    def _on_close(self, event):
        event.veto = False

    # ------------------------------------------------------------------------
    # Trait property getter/setters
    # ------------------------------------------------------------------------

    def _get_active_editor(self):
        if self.editor_area is not None:
            return self.editor_area.active_editor
        return None

    def _get_field_keys(self):
        if self.fields_model is not None:
            return self.fields_model.field_keys
        return []
