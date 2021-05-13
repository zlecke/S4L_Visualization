"""
Pyface Action Groups for the S4L Visualization application
"""
from pyface.tasks.api import Task
from pyface.action.api import Group, Action, ActionItem
from traits.api import List, Property, cached_property, observe, Instance, Bool


class SelectFieldAction(Action):
    """
    A Pyface Action for changing and displaying which field is currently being
    visualized by the S4L Visualization application.
    """
    #: The task with which the action is associated. Set by the framework
    task = Instance(Task)

    #: The action's style.
    #: Can be one of {"push", "radio", "toggle", "widget"}.
    style = 'radio'

    #: Is the action checked? This is only relevant if the action style is
    #: 'radio' or 'toggle'
    checked = Property(Bool, observe='task.fields_model.selected_field_key')

    def perform(self, event=None):
        """ Performs the action.

            Parameters
            ----------
            event : :py:class:`pyface.action.action_event.ActionEvent`
                The event which triggered the action.
            """
        if self.task is not None and self.task.fields_model is not None:
            self.task.fields_model.selected_field_key = self.name

    @cached_property
    def _get_checked(self):
        if self.task is not None and self.task.fields_model is not None:
            return self.name == self.task.fields_model.selected_field_key
        return False

    def _set_checked(self, val):
        pass


class FieldSelectionGroup(Group):
    """
    A Pyface Action Group for selecting which field to display in the
    S4L Visualization application.
    """
    # pylint: disable=unused-argument, attribute-defined-outside-init

    #: The group's identifier.
    id = 'FieldSelectionGroup'

    #: The items in the group.
    items = List()

    #: The task associated with the group.
    task = Property(observe='parent.controller')

    #: The fields available to be displayed
    field_keys = Property(observe="task.fields_model.field_keys")

    @cached_property
    def _get_task(self):
        manager = self.get_manager()

        if manager is None or manager.controller is None:
            return None

        return manager.controller.task

    @cached_property
    def _get_field_keys(self):
        if self.task is None or self.task.fields_model is None:
            return []

        return self.task.fields_model.field_keys

    @observe('field_keys.items')
    def _field_keys_updated(self, event):
        self.destroy()

        items = []
        for key in self.field_keys:
            action = SelectFieldAction(task=self.task, name=key)
            items.append(ActionItem(action=action, parent=self))

        self.items = items

        manager = self.get_manager()
        manager.changed = True

    def get_manager(self):
        """
        Returns the group manager

        Returns
        -------
        manager : :py:class:`pyface.action.ActionManager`
            The ActionManager of the group
        """
        manager = self
        while isinstance(manager, Group):
            manager = manager.parent
        return manager
