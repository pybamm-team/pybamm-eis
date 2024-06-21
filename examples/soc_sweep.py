import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pbeis

# Load model
model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

# Set up parameters with cell SOC as an input parameter
parameter_values = pybamm.ParameterValues("Chen2020")
x0, x100, y100, y0 = pybamm.lithium_ion.get_min_max_stoichiometries(parameter_values)
z = pybamm.InputParameter("SOC")
x = x0 + z * (x100 - x0)
y = y0 - z * (y0 - y100)
c_n_max = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
c_p_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
parameter_values.update(
    {
        "Initial concentration in negative electrode [mol.m-3]": x * c_n_max,
        "Initial concentration in positive electrode [mol.m-3]": y * c_p_max,
    }
)

# Create simulation
eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)

# Choose frequencies and calculate impedance, looping over input parameter values
# and adding the results to a Nyquist plot
frequencies = np.logspace(-4, 4, 30)
_, ax = plt.subplots()
for z in [0.1, 0.5, 0.9]:
    eis_sim.solve(frequencies, inputs={"SOC": z})
    eis_sim.nyquist_plot(ax=ax, label=f"SOC = {z}")
ax.set_xlim([0, 0.08])
ax.set_ylim([0, 0.08])
ax.legend()
plt.show()
