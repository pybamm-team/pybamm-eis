import pybamm
import numpy as np
from eis_simulation import EISSimulation

# Load model (DFN with capacitance)
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN")

# Create simulation
eis_sim = EISSimulation(model)

# Choose frequencies and calculate impedance
frequencies = np.logspace(-4, 4, 30)
eis_sim.solve(frequencies)

# Generate a Nyquist plot
eis_sim.nyquist_plot()
