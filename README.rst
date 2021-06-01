*****************************
Sim4Life Field Visualizations
*****************************

Displays field data from Sim4Life simulations of spinal cord electrostimulation.

Installation
============

#. Download and install `Python 3.8.4 <https://www.python.org/downloads/release/python-384>`_
#. Clone the repository
#. Save the CSF model vtk file to the base folder of the repository and rename it to ``CSF.vtk``

   OR

   Change the DEFAULT ``csf_model`` variable in your local ``config.ini`` file to be the file path of the CSF model vtk file

#. Open a command prompt window and navigate to the repository
#. Run the following commands to set up the environment:

   .. code:: console

      python -m venv slvenv
      slvenv\Scripts\activate
      pip install -r requirements.txt

Usage
=====

To run the application, run the following commands from a command prompt window located in the base folder of the repository:

   .. code:: console

      slvenv\Scripts\activate
      python run.py

Documentation
=============

To build the documentation:

#. Install the `Sphinx <https://www.sphinx-doc.org/>`_,
   `Read the Docs Sphinx Theme <https://sphinx-rtd-theme.readthedocs.io/>`_,
   and `Numpydoc <https://numpydoc.readthedocs.io/>`_ packages:

    .. code:: console

      pip install sphinx sphinx-rtd-theme numpydoc

#. Create a new directory $RepoHome$/docs/_build

#. Run the following command from the $RepoHome$/docs/ folder:

   .. code:: console

      make html