# PyBaMM EIS

PyBaMM EIS rapidly calculates the electrochemical impedance of any battery model defined using PyBaMM, see [the documentation](https://pybamm-eis.readthedocs.io/en/latest/)

## 🔋 Using PyBaMM EIS

The easiest way to use PyBaMM EIS is to compute the impedance of a model of your choice with the default parameters:

```python
import pybammeis
import pybamm
import numpy as np

# DFN with capacitance
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})
eis_sim = pybammeis.EISSimulation(model)
# calculate impedance at log-spaced frequencies
eis_sim.solve(np.logspace(-4, 4, 30))
eis_sim.nyquist_plot()
```

## 💻 About PyBaMM

The example simulations use the package [PyBaMM](www.pybamm.org) (Python Battery Mathematical Modelling). PyBaMM is an open-source battery simulation package
written in Python. Our mission is to accelerate battery modelling research by
providing open-source tools for multi-institutional, interdisciplinary collaboration.
Broadly, PyBaMM consists of
(i) a framework for writing and solving systems
of differential equations,
(ii) a library of battery models and parameters, and
(iii) specialized tools for simulating battery-specific experiments and visualizing the results.
Together, these enable flexible model definitions and fast battery simulations, allowing users to
explore the effect of different battery designs and modeling assumptions under a variety of operating scenarios.

## 🚀 Installation

In order to run the notebooks in this repository you will need to install the `pybamm-eis` package. We recommend installing within a [virtual environment](https://docs.python.org/3/tutorial/venv.html) in order to not alter any python distribution files on your machine.

PyBaMM is available on GNU/Linux, MacOS and Windows. For more detailed instructions on how to install PyBaMM, see [the PyBaMM documentation](https://pybamm.readthedocs.io/en/latest/install/GNU-linux.html#user-install).

### Linux/Mac OS

To install the requirements on Linux/Mac OS use the following terminal commands:

1. Clone the repository
    ```bash
    git clone https://github.com/pybamm-team/pybamm-eis
    ```

2. Change into the `pybamm-eis` directory
    ```bash
    cd pybamm-eis
    ```

3. Create a virtual environment
    ```bash
    virtualenv env
    ```

4. Activate the virtual environment
    ```bash
    source env/bin/activate
    ```

5. Install the `pybammeis` package
    ```bash
    pip install .
    ```

### Windows

To install the requirements on Windows use the following commands:

1. Clone the repository
    ```bash
    git clone https://github.com/pybamm-team/pybamm-eis
    ```

2. Change into the `pybamm-eis` directory
    ```bash
    cd pybamm-eis
    ```

3. Create a virtual environment
    ```bash
    python -m virtualenv env
    ```

4. Activate the virtual environment
    ```bash
    \path\to\env\Scripts\activate
    ```
    where `\path\to\env` is the path to the environment created in step 3
    (e.g. `C:\Users\'Username'\env\Scripts\activate.bat`).

5. Install the `pybammeis` package
    ```bash
    pip install .
    ```

As an alternative, you can set up [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about). This allows you to run a full Linux distribution within Windows.

### Developer

To install as a developer follow the instructions above, replacing the final step with

```bash
pip install -e .
```

This will allow you to edit the code locally.

## 📖 Documentation

API documentation for the `pybammeis` package can be built locally using [Sphinx](https://www.sphinx-doc.org/en/master/). To build the documentation, first change into the `docs` directory, and then execute the following command:

```bash
make html
```

This will generate a number of html files in the `docs/_build/html` directory. To view the documentation, open the file `docs/_build/html/index.html` in a web browser.

## 🛠️ Contributing to PyBaMM

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md) first.

## 🎉 Acknowledgments

This code was developed as part of the Oxford Mathematics Summer Project _"Efficient Linear Algebra Methods to Determine Li-ion Battery Behaviour"_.

Student: Rishit Dhoot
Supervisors: Prof Colin Please and Dr. Robert Timms

## 📫 Get in touch

For any questions, comments, suggestions or bug reports, please see the
[contact page](https://www.pybamm.org/community).

## 📃 License

pybamm-eis is fully open source. For more information about its license, see [LICENSE](https://github.com/pybamm-team/pybamm-eis/blob/main/LICENSE.md).
