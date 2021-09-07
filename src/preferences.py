"""
User preferences for the S4L Visualization application.
"""
from traits.api import HasTraits, Instance, Dict, List, Str
from traitsui.api import (
    View, Item, ListEditor, Handler, TextEditor, OKButton, CancelButton
)

from pyface.api import confirm, YES

from configparser import ConfigParser


class DContainer(HasTraits):
    value = Dict

    def __getattr__(self, key):
        if key in self.value:
            return self.value[key]

    def __setattr__(self, key, value):
        if key in self.value:
            self.value[key] = value
        else:
            super().__setattr__(key, value)


class PreferenceSection(HasTraits):
    configuration = Instance(ConfigParser)
    section_name = Str()

    options = Instance(DContainer)

    def _options_default(self):
        d = DContainer()
        d.value = {k: v for k, v in self.configuration.items(self.section_name)}
        return d

    def default_traits_view(self):
        items = [Item('object.options.{}'.format(option),
                      label=option,
                      editor=TextEditor(auto_set=False, enter_set=True)) for option in
                 self.options.value.keys()]
        return View(*items, kind='modal')


class PreferenceHandler(Handler):

    def close(self, info, is_ok):
        if is_ok:
            same = True
            for item in info.object._to_edit:
                for option in item.options.value.keys():
                    if info.object.configuration.get(item.section_name, option) !=\
                            item.options.value[option]:
                        same = False
            if not same:
                save = confirm(None, "Do you want to save changes?", default=YES)
                if save == YES:
                    self._save_configuration_changes(info)
                else:
                    info.object.reset_traits(traits=['_to_edit'])
        else:
            info.object.reset_traits(traits=['_to_edit'])

        return True

    def perform(self, info, action, event):
        if action.id == 'Apply':
            self._save_configuration_changes(info)
        else:
            Handler.perform(self, info, action, event)

    def _save_configuration_changes(self, info):
        options = {item.section_name: item.options.value for item in info.object._to_edit}
        info.object.configuration.read_dict(options)


class PreferenceDialog(HasTraits):
    configuration = Instance(ConfigParser)

    title = Str()

    _to_edit = List(Instance(PreferenceSection))

    def default_traits_view(self):
        return View(Item('_to_edit',
                         editor=ListEditor(use_notebook=True, page_name='.section_name'),
                         style='custom',
                         show_label=False),
                    title=self.title,
                    resizable=True,
                    handler=PreferenceHandler(),
                    buttons=[OKButton, CancelButton])

    def __to_edit_default(self):
        d = []
        for section in self.configuration.sections():
            if section == '':
                continue
            options = DContainer(
                value={option: value for option, value in self.configuration.items(section)
                       if (option in self.configuration._sections[section] and section == "Display")
                       or section != "Display"})
            d.append(PreferenceSection(options=options, section_name=section))
        return d


if __name__ == '__main__':
    config = ConfigParser()
    config.read_file(open('default.ini'))

    pref_diag = PreferenceDialog(configuration=config)
    pref_diag.configure_traits()
    print([(section, config.items(section)) for section in config.sections()])
