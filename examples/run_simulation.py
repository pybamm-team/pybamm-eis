import numpy as np
import pybamm

import pybammeis

# Load model (DFN with capacitance)
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})

# Create simulation
eis_sim = pybammeis.EISSimulation(model)

# Choose frequencies and calculate impedance
frequencies = np.logspace(-4, 4, 30)
eis_sim.solve(frequencies)

# Generate a Nyquist plot
eis_sim.nyquist_plot()
