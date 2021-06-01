********
Menu Bar
********

The menu bar located at the top of the application window contains
many useful buttons and toggles.

.. _file-menu:

File Menu
=========

The File menu contains 2 items: the :ref:`open-item` and the :ref:`export` menu.

.. image:: /_static/user/file-menu.png

.. _open-item:

Open...
-------

.. image:: /_static/user/choose-data-file.png

Allows user to open new data file for visualization.

.. note::

   Data files must be Matlab files with extension ``.mat`` exported
   from a Sim4Life EMLF simulation.

.. _export:

Export
------

The export menu contains 2 items: the :ref:`export-slice` button and the
:ref:`export-line` button.

.. _export-slice:

Export Slice
^^^^^^^^^^^^

.. image:: /_static/user/export-slice.png

Allows the user to export the data from the current :ref:`slice-figure` to a
CSV or Excel file.

.. _export-line:

Export Line
^^^^^^^^^^^

.. image:: /_static/user/export-line.png

Allows the user to export the data from the current :ref:`line-figure` to a
CSV or Excel file.

.. _settings:

Settings
========

.. image:: /_static/user/settings.png

Allows the user to change default settings both globally, through the DEFAULT tab, and
by participant, through a tab with that participant's ID. If a participant's ID is not
present in this dialog, set the current participant's ID in the :ref:`participant-id-pane`.

.. _view-menu:

View Menu
=========

The View Menu contains 4 items: the :ref:`panes` menu, the :ref:`full-model`
toggle, the :ref:`log-scale` toggle, and the :ref:`line-cross-marker` toggle.

.. image:: /_static/user/view-menu.png

.. _panes:

Panes
-----

Allows the user to hide/show the :ref:`line-attributes` and :ref:`plane-attributes`
dock panes.

.. _full-model:

Full Model
----------

Allows the user to toggle between showing the full spinal cord model and showing only the
portion of the spinal cord model below the cut plane in the :doc:`3d-view`.

.. _log-scale:

Log Scale
---------

Allows the user to toggle between log scaling and linear scaling for the :ref:`slice-figure`.

.. _line-cross-marker:

Line Cross Marker
-----------------

Allows the user to show/hide the marker indicating where on the :ref:`slice-figure` the line
described by the points in the :ref:`line-attributes` pane.

.. _edit-menu:

Edit Menu
=========

The edit menu contains 2 items: the :ref:`new-cord-model` button and the
:ref:`choose-field` menu.

.. image:: /_static/user/edit-menu.png

.. _new-cord-model:

New Cord Model
--------------

.. image:: /_static/user/choose-cord-model.png

Allows the user to choose a new spinal cord model file.

.. note::
   The file must be a VTK model file with extension ``.vtk``.

.. _choose-field:

Choose Field
------------

This allows the user to change between fields that are included in the current data file.

.. note::

   Currently, only vector fields can be shown.