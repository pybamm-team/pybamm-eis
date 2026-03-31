import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybammeis

# Load model
model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
parameter_values = pybamm.ParameterValues("Chen2020")

# Create simulation
eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)

# Choose frequencies and calculate impedance, looping over SOC values
# and adding the results to a Nyquist plot
frequencies = np.logspace(-4, 4, 30)
_, ax = plt.subplots()
for z in [0.1, 0.5, 0.9]:
    eis_sim.solve(frequencies, initial_soc=z)
    eis_sim.nyquist_plot(ax=ax, label=f"SOC = {z}")
ax.set_xlim([0, 0.08])
ax.set_ylim([0, 0.08])
ax.legend()
plt.show()
