# Sim4Life Field Visualizations

Displays field data from Sim4Life simulations of spinal cord electrostimulation.

## Installation

1. Download and install [Python 3.8.4](https://www.python.org/downloads/release/python-384)
2. Clone the repository 
3. Save the CSF model vtk file to the base folder of the repository

    OR
    
    Change the `default_csf_model` variable in `preferences.py` to be the file path of the CSF model vtk file
4. Open a command prompt window and navigate to the repository
5. Run the following commands to set up the environment:
    <ol type="a">
      <li><code>python -m venv slvenv</code></li>
      <li><code>slvenv\Scripts\activate</code></li>
      <li><code>pip install -r requirements.txt</code></li>
    </ol>

## Usage
1. Open a command prompt window and navigate to the folder containing the program file
2. Run the following command to activate up the environment:
    ```
    slvenv\Scripts\activate
    ```
3. Run the program with the following command:
    ```
    python S4L_Vis_App.py
    ```