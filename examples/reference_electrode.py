import numpy as np
import pybamm

import pybammeis

# Load model (DFN with capacitance)
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})

# Insert reference electrode
L_n = model.param.n.L  # "Negative electrode thickness [m]"
L_s = model.param.s.L  # "Separator thickness [m]"
model.insert_reference_electrode(L_n + 0.5 * L_s)

# Create simulation and choose the reference electrode voltage as target
eis_sim = pybammeis.EISSimulation(
    model, target="Negative electrode 3E potential [V]"
)

# Choose frequencies and calculate impedance
frequencies = np.logspace(-4, 4, 30)
eis_sim.solve(frequencies)

# Since it is the negative electrode: flip the sign
eis_sim.solution *= -1

# Generate a Nyquist plot
eis_sim.nyquist_plot()
