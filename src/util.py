import numpy as np
from traits.has_traits import HasStrictTraits
from traits.trait_numeric import Array
from traitsui.editors import ArrayEditor
from traitsui.item import Item
from traitsui.view import View


class ArrayClass(HasStrictTraits):
    value = Array(value=np.array([0, 0, 0]), dtype=np.float64)

    def __init__(self, *args, **kwargs):
        super().__init__()
        if 'value' in kwargs:
            self.value = kwargs['value']
        elif len(args) > 0:
            self.value = args[0]

    def default_traits_view(self):
        return View(Item('value', editor=ArrayEditor(width=-60, auto_set=False, enter_set=True), show_label=False))


scale_factor = 0.001
scalar_fields = ['El__Loss_Density', 'EM_Potential', 'SAR']


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def arg_find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()