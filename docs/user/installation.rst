************
Installation
************

#. Download and install `Python 3.8.4 <https://www.python.org/downloads/release/python-384>`_
#. Clone the repository
#. Save the CSF model vtk file to the base folder of the repository and rename it to ``CSF.vtk``

   OR

   Change the DEFAULT ``csf_model`` variable in your local ``config.ini`` file to be the file path of the CSF model vtk file

   .. literalinclude:: /../config_template.ini
      :linenos:
      :emphasize-lines: 19-20
      :caption: config_template.ini

#. Open a command prompt window and navigate to the repository
#. Run the following commands to set up the environment:

   .. code:: console

      python -m venv slvenv
      slvenv\Scripts\activate
      pip install -r requirements.txt