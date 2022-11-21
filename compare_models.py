import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as timer
from eis_simulation import EISSimulation
from plotting import nyquist_plot

# Load models and parameters
models = [
    pybamm.lithium_ion.SPM(options={"surface form": "differential"}, name="SPM"),
    pybamm.lithium_ion.SPMe(options={"surface form": "differential"}, name="SPMe"),
    pybamm.lithium_ion.MPM(options={"surface form": "algebraic"}, name="MPM"),
    pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN"),
    pybamm.lithium_ion.SPMe(
        {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 2,
        },
        name="SPMePouch",
    ),
    pybamm.lithium_ion.DFN(
        {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 2,
        },
        name="DFNPouch",
    ),
]
parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values = pybamm.get_size_distribution_parameters(
    parameter_values, sd_n=0.2, sd_p=0.4
)

# Loop over models and calculate impedance
frequencies = np.logspace(-4, 4, 30)
impedances = []
for model in models:
    start_time = timer.time()
    eis_sim = EISSimulation(model, parameter_values=parameter_values)
    impedances_freq = eis_sim.solve(
        frequencies,
    )
    end_time = timer.time()
    time_elapsed = end_time - start_time
    print(f"Frequency domain ({model.name}): ", time_elapsed, "s")
    impedances.append(impedances_freq)

# Plot individually
for i, model in enumerate(models):
    _, ax = plt.subplots()
    ax = nyquist_plot(impedances[i], ax=ax)
    plt.suptitle(f"{model.name}")
    plt.savefig(f"figures/{model.name}.pdf", dpi=300)

# Compare
_, ax = plt.subplots()
for i, model in enumerate(models):
    ax = nyquist_plot(impedances[i], ax=ax, label=f"{model.name}", alpha=0.7)
ax.legend()
plt.savefig("figures/compare_models.pdf", dpi=300)
plt.show()
