"""
A custom TraitsUI range editor that is a slider that scrubs through a list of
mapped values. Displays current value as #.## mm.
"""
from types import CodeType

from traits.api import (
    CTrait,
    Property,
    Range,
    Enum,
    Int,
    Any,
    Str,
    Bool,
    Undefined,
)
from traitsui.qt4.basic_editor_factory import BasicEditorFactory
from traitsui.qt4.range_editor import SimpleSliderEditor
from traitsui.view import View


class _RangeEditor(SimpleSliderEditor):
    low_label = Str()
    high_label = Str()
    low_label_name = Str()
    high_label_name = Str()
    map_to_values_name = Str()
    map_to_values = Any()

    def init(self, parent):
        factory = self.factory
        super().init(parent)
        if not factory.low_label_name:
            self.low_label = factory.low_label
        if not factory.high_label_name:
            self.high_label = factory.high_label

        self.sync_value(factory.low_label_name, 'low_label', 'from')
        self.sync_value(factory.high_label_name, 'high_label', 'from')
        self.sync_value(factory.map_to_values_name, 'map_to_values', 'from')

        if self.low_label != "":
            self._label_lo.setText(self.low_label)
        if self.high_label != "":
            self._label_hi.setText(self.high_label)

    def _low_changed(self, low):
        if self.value < low:
            if self.factory.is_float:
                self.value = float(low)
            else:
                self.value = int(low)

        if self._label_lo is not None:
            if self.low_label != "":
                self._label_lo.setText(self.low_label)
            else:
                self._label_lo.setText(self.format % low)
            self.update_editor()

    def _high_changed(self, high):
        if self.value > high:
            if self.factory.is_float:
                self.value = float(high)
            else:
                self.value = int(high)

        if self._label_hi is not None:
            if self.high_label != "":
                self._label_hi.setText(self.high_label)
            else:
                self._label_hi.setText(self.format % high)
            self.update_editor()

    def _low_label_changed(self, low_label):
        if self._label_lo is not None:
            self._label_lo.setText(low_label)

    def _high_label_changed(self, high_label):
        if self._label_hi is not None:
            self._label_hi.setText(high_label)

    def update_editor(self):
        super().update_editor()
        if self.map_to_values is not None:
            self.control.text.setText('{:.2f} mm'.format(self.map_to_values[self.value]))

    def _set_value(self, value):
        super()._set_value(value)
        if self.map_to_values is not None:
            self.control.text.setText('{:.2f} mm'.format(self.map_to_values[self.value]))


class QRangeEditor(BasicEditorFactory):
    """
    A custom TraitsUI range editor that is a slider that scrubs through a list of
    mapped values. Displays current value as #.## mm.
    """

    # pylint: disable=too-many-instance-attributes, redefined-builtin, too-many-arguments
    # pylint: disable=access-member-before-definition, attribute-defined-outside-init
    # pylint: disable=arguments-differ, no-self-use, invalid-name

    klass = _RangeEditor

    low_label_name = Str()
    high_label_name = Str()
    map_to_values_name = Str()

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    #: Number of columns when displayed as an enumeration
    cols = Range(1, 20)

    #: Is user input set on every keystroke?
    auto_set = Bool(True)

    #: Is user input set when the Enter key is pressed?
    enter_set = Bool(False)

    #: Label for the low end of the range
    low_label = Str()

    #: Label for the high end of the range
    high_label = Str()

    #: FIXME: This is supported only in the wx backend so far.
    #: The width of the low and high labels
    label_width = Int()

    #: The name of an [object.]trait that defines the low value for the range
    low_name = Str()

    #: The name of an [object.]trait that defines the high value for the range
    high_name = Str()

    #: Formatting string used to format value and labels
    format = Str("%s")

    #: Is the range for floating pointer numbers (vs. integers)?
    is_float = Bool(Undefined)

    #: Function to evaluate floats/ints when they are assigned to an object
    #: trait
    evaluate = Any()

    #: The object trait containing the function used to evaluate floats/ints
    evaluate_name = Str()

    #: Low end of range
    low = Property()

    #: High end of range
    high = Property()

    #: Display mode to use
    mode = Enum(
            "auto", "slider", "xslider", "spinner", "enum", "text", "logslider"
    )

    # -------------------------------------------------------------------------
    #  Traits view definition:
    # -------------------------------------------------------------------------

    traits_view = View(
            [
                    ["low", "high", "|[Range]"],
                    ["low_label{Low}", "high_label{High}", "|[Range Labels]"],
                    [
                            "auto_set{Set automatically}",
                            "enter_set{Set on enter key pressed}",
                            "is_float{Is floating point range}",
                            "-[Options]>",
                    ],
                    ["cols", "|[Number of columns for integer custom style]<>"],
            ]
    )

    def init(self, handler=None):
        """ Performs any initialization needed after all constructor traits
            have been set.
        """

        # pylint: disable=protected-access, eval-used

        if handler is not None:
            if isinstance(handler, CTrait):
                handler = handler.handler

            if self.low_name == "":
                if isinstance(handler._low, CodeType):
                    self.low = eval(handler._low)
                else:
                    self.low = handler._low

            if self.high_name == "":
                if isinstance(handler._low, CodeType):
                    self.high = eval(handler._high)
                else:
                    self.high = handler._high
        else:
            if (self.low is None) and (self.low_name == ""):
                self.low = 0.0

            if (self.high is None) and (self.high_name == ""):
                self.high = 1.0

    def _get_low(self):
        return self._low

    def _set_low(self, low):
        old_low = self._low
        self._low = low = self._cast(low)
        if self.is_float is Undefined:
            self.is_float = isinstance(low, float)

        if (self.low_label == "") or (
                self.low_label == str(old_low)
        ):
            self.low_label = str(low)

    def _get_high(self):
        return self._high

    def _set_high(self, high):
        old_high = self._high
        self._high = high = self._cast(high)
        if self.is_float is Undefined:
            self.is_float = isinstance(high, float)

        if (self.high_label == "") or (
                self.high_label == str(old_high)
        ):
            self.high_label = str(high)

    def _cast(self, value):
        if not isinstance(value, str):
            return value

        try:
            return int(value)
        except ValueError:
            return float(value)

    # -- Private Methods ------------------------------------------------------

    def _get_low_high(self, ui):
        """ Returns the low and high values used to determine the initial range.
        """
        low, high = self.low, self.high

        if (low is None) and (self.low_name != ""):
            low = self.named_value(self.low_name, ui)
            if self.is_float is Undefined:
                self.is_float = isinstance(low, float)

        if (high is None) and (self.high_name != ""):
            high = self.named_value(self.high_name, ui)
            if self.is_float is Undefined:
                self.is_float = isinstance(high, float)

        if self.is_float is Undefined:
            self.is_float = True

        return (low, high, self.is_float)

    # -------------------------------------------------------------------------
    #  'Editor' factory methods:
    # -------------------------------------------------------------------------

    def simple_editor(self, ui, object, name, description, parent):
        """ Generates an editor using the "simple" style.
        Overridden to set the values of the _low_value, _high_value and
        is_float traits.
        """
        self._low_value, self._high_value, self.is_float = self._get_low_high(
                ui
        )
        return super().simple_editor(
                ui, object, name, description, parent
        )

    def custom_editor(self, ui, object, name, description, parent):
        """ Generates an editor using the "custom" style.
        Overridden to set the values of the _low_value, _high_value and
        is_float traits.
        """
        self._low_value, self._high_value, self.is_float = self._get_low_high(
                ui
        )
        return super().custom_editor(
                ui, object, name, description, parent
        )
