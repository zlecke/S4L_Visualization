from pyface.qt import QtGui

import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory


class _MPLFigureEditor(Editor):
    scrollable = True

    toolbar = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
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
    klass = _MPLFigureEditor
