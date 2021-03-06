"""
Utility functions and classes for the S4L Visualization application.
"""

import numpy as np
from traits.api import HasStrictTraits, Array
from traitsui.api import ArrayEditor, Item, View


SCALE_FACTOR = 0.001
scalar_fields = ['El__Loss_Density', 'EM_Potential', 'SAR']


def find_nearest(array, value):
    """
    Finds the nearest value to `value` in `array`.

    Parameters
    ----------
    array : array_like
    value : int or float

    Returns
    -------
    closest : int or float
        Closest value in `array` to `value`
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def arg_find_nearest(array, value):
    """
    Find the index of the nearest value to `value` in `array`.

    Parameters
    ----------
    array : array_like
    value : int or float

    Returns
    -------
    idx : int
        Index of nearest value to `value` in `array`.
    """
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


class ArrayClass(HasStrictTraits):
    """
    Wrapper class for traits Array type objects.
    """
    #: A :py:class:`traits.trait_numeric.Array` instance
    value = Array(value=np.array([0, 0, 0]), dtype=np.float64)

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args : iterable
        **kwargs : dict

        Other Parameters
        ----------------
        value : array_like, optional
            Default value is [0, 0, 0]
        """
        super().__init__()
        if 'value' in kwargs:
            self.value = kwargs['value']
        elif len(args) > 0:
            self.value = args[0]

    def _value_default(self): # pylint: disable=no-self-use
        return Array(value=np.array([0, 0, 0]), dtype=np.float64)

    def default_traits_view(self):
        """
        Create the default traits View object for the model.

        Returns
        -------
        default_traits_view : :py:class:`traitsui.view.View`
            The default traits View object for the model.
        """
        return View(Item('value', editor=ArrayEditor(width=-60, auto_set=False, enter_set=True),
                         show_label=False))
