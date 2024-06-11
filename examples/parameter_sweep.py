import pbeis
import pybamm
import matplotlib.pyplot as plt

# Load model
model = pybamm.lithium_ion.SPM(
    options={"surface form": "differential", "contact resistance": "true"}
)

# Set up parameters
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    {
        "Negative electrode exchange-current density [A.m-2]": pybamm.InputParameter(
            "j0_n"
        ),
        "Contact resistance [Ohm]": 2 * 1e-3,
    }
)

# Create simulation
eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)

# Choose frequencies and calculate impedance, looping over input parameter values
# and adding the results to a Nyquist plot
frequencies = pbeis.logspace(-4, 4, 30)
_, ax = plt.subplots()
for j0_n in [0.5, 1, 2]:
    eis_sim.solve(frequencies, inputs={"j0_n": j0_n})
    eis_sim.nyquist_plot(ax=ax, label=f"j0_n = {j0_n}")
ax.legend()
ax.set_xlim([0, 0.05])
ax.set_ylim([0, 0.05])
plt.show()
