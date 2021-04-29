from pyface.api import GUI
from pyface.tasks.api import TaskWindow


from s4l_visualization_task import S4LVisualizationTask


def main(argv):
    """ A program for visualizing Sim4Life EM fields from scES simulations
    """
    from traits.api import push_exception_handler
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
    import sys

    main(sys.argv)