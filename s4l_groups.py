from pyface.tasks.api import Task
from pyface.action.api import Group, Action, ActionItem
from traits.api import List, Property, cached_property, observe, Instance, Bool


class SelectFieldAction(Action):
    task = Instance(Task)

    style = 'radio'

    checked = Property(Bool, observe='task.fields_model.selected_field_key')

    def perform(self, event=None):
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
    id = 'FieldSelectionGroup'
    items = List()

    task = Property(observe='parent.controller')

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
        manager = self
        while isinstance(manager, Group):
            manager = manager.parent
        return manager