"""
A TraitsUI editor for matplotlib figures.
"""

# pylint: disable=wrong-import-position, no-member

from pyface.qt import QtGui

from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory

import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


class _MPLFigureEditor(Editor):
    scrollable = True

    toolbar = True

    def init(self, parent):
        """Initialize the editor

        Initialize the editor that will hold the matplotlib figure

        Parameters
        ----------
        parent
            The parent widget for the editor
        """
        self.control = self._create_canvas()
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self):
        """
        Creates the canvas that will hold the matplotlib figure

        Returns
        -------
        frame : pyface.qt.QtGui.QWidget
            Widget containing the matplotlib figure
        """
        # matplotlib commands to create a canvas
        if self.toolbar:
            frame = QtGui.QWidget()
            mpl_canvas = FigureCanvas(self.value)
            mpl_canvas.setParent(frame)
            mpl_toolbar = NavigationToolbar2QT(mpl_canvas, frame)

            vbox = QtGui.QVBoxLayout()
            vbox.addWidget(mpl_toolbar)
            vbox.addWidget(mpl_canvas)
            frame.setLayout(vbox)
        else:
            frame = FigureCanvas(self.value)

        return frame


class MPLFigureEditor(BasicEditorFactory):
    """
    An Editor Factory that creates a TraitsUI editor for a matplotlib figure.
    """
    klass = _MPLFigureEditor
