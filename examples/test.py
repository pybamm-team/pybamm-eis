import pbeis
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as timer

# Set up
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})
parameter_values = pybamm.ParameterValues("Marquis2019")
frequencies = np.logspace(-4, 2, 10)

# Frequency domain
methods = ["direct", "prebicgstab"]
impedances_freqs = []
for method in methods:
    start_time = timer.time()
    eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)
    impedances_freq = eis_sim.solve(frequencies, method)
    end_time = timer.time()
    time_elapsed = end_time - start_time
    print(f"Frequency domain ({method}): ", time_elapsed, "s")
    impedances_freqs.append(impedances_freq)

# Compare
_, ax = plt.subplots()
for i, method in enumerate(methods):
    ax = pbeis.nyquist_plot(
        impedances_freqs[i], ax=ax, label=f"Frequency ({method})", alpha=0.7
    )
ax.legend()
plt.suptitle(f"{model.name}")
plt.savefig(f"figures/{model.name}_time_vs_freq.pdf", dpi=300)
plt.show()
