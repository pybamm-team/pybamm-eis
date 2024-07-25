import numpy as np
import pybamm

import pbeis

# Load model (DFN with capacitance)
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})

# Create simulation
eis_sim = pbeis.EISSimulation(model)

# Choose frequencies and calculate impedance
frequencies = np.logspace(-4, 4, 30)
eis_sim.solve(frequencies)

# Generate a Nyquist plot
eis_sim.nyquist_plot()
