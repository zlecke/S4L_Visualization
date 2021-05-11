**********
Dock Panes
**********

Dock panes are tool windows that can be placed along the edges of the application
window that can be moved and/or hidden. They can be placed separately, as seen in
the default layout shown in the screenshots in this guide, or in the same area as
tabbed pages.

.. seealso::

   For info on how to hide a dock pane, see  the :ref:`panes` menu item.

.. _plane-attributes:

Plane Attributes
================

.. image:: /_static/user/plane-attributes.png

The plane attributes dock pane allows the user to change the
plane along which the data being visualized is cut, also known as the "cut plane".

The slider allows the user to scrub through the various slices along the axis normal
to the cut plane.

The 4 radio buttons below the slider allows the user to choose which axis the cut plane
is normal to.

When the cut plane is chosen to be an arbitrary plane, the slider defaults
to slicing along the z-axis as long as the normal vector to the cut plane is non-zero.
Entry boxes for manual entry of the normal vector and origin of the cut plane are also
shown when the cut plane is chosen to be an arbitrary plane.

.. warning::

   The slider and the entry boxes will not always match when in arbitrary plane mode.
   The text entry boxes are always the actual values for the cut plane.

.. _line-attributes:

Line Attributes
===============

.. image:: /_static/user/point-menu.png

The line attributes dock pane allows the user to add, remove, and edit the points
describing the line along which data is sampled to produce the :ref:`line-figure`.

Points can be added or removed by clicking the |point-image| icon next to an
existing point.

The :ref:`line-figure` will sample the data along a linear path between each successive
pair of points in the order shown in the line attributes pane. To ensure that each
successive point is the point closest to the one that came before it, click the "Sort"
button at the top of the pane to sort the points.

.. note::

   Be sure that the first point in the list is either the top most or bottom most point
   in the list before using the Sort button.

.. |point-image| image:: /../slvenv/Lib/site-packages/traitsui/qt4/images/list_editor.png