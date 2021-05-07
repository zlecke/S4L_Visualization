*****************************
Sim4Life Field Visualizations
*****************************

Displays field data from Sim4Life simulations of spinal cord electrostimulation.

Installation
============
.. include-after

#. Download and install `Python 3.8.4 <https://www.python.org/downloads/release/python-384>`_
#. Clone the repository
#. Save the CSF model vtk file to the base folder of the repository and rename it to ``CSF.vtk``

   OR

   Change the ``default_csf_model`` variable in ``preferences.py`` to be the file path of the CSF model vtk file

   .. literalinclude:: /../src/preferences.py
      :linenos:
      :emphasize-lines: 2
      :caption: src/preferences.py

#. Open a command prompt window and navigate to the repository
#. Run the following commands to set up the environment:

   .. code:: console

      python -m venv slvenv
      slvenv\Scripts\activate
      pip install -r requirements.txt

.. include-before

Usage
=====

To run the application, run the following commands from a command prompt window located in the base folder of the repository:

   .. code:: console

      slvenv\Scripts\activate
      python run.py
