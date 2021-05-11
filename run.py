"""
Displays field data from Sim4Life electromagnetic low-frequency simulations of
spinal cord electrostimulation.
"""

from pyface.api import GUI
from pyface.tasks.api import TaskWindow

from traits.api import push_exception_handler


from src.s4l_visualization_task import S4LVisualizationTask


def main():
    """ A program for visualizing Sim4Life EM fields from scES simulations
    """
    push_exception_handler(reraise_exceptions=True)
    # Create the GUI (this does NOT start the GUI event loop).
    gui = GUI()

    # Create a Task and add it to a TaskWindow.
    task = S4LVisualizationTask()
    window = TaskWindow(size=(1500, 650))
    window.add_task(task)

    # Show the window.
    window.open()

    # Start the GUI event loop.
    gui.start_event_loop()


if __name__ == "__main__":
    main()
