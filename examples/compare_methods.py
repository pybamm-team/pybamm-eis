import pbeis
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as timer
from scipy.fft import fft

# Set up
model = pybamm.lithium_ion.SPM(options={"surface form": "differential"}, name="SPM")
parameter_values = pybamm.ParameterValues("Marquis2019")
frequencies = np.logspace(-4, 2, 30)

# Time domain
I = 50 * 1e-3
number_of_periods = 20
samples_per_period = 16


def current_function(t):
    return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)


parameter_values["Current function [A]"] = current_function

start_time = timer.time()

sim = pybamm.Simulation(
    model, parameter_values=parameter_values, solver=pybamm.ScipySolver()
)

impedances_time = []
for frequency in frequencies:
    # Solve
    period = 1 / frequency
    dt = period / samples_per_period
    t_eval = np.array(range(0, 1 + samples_per_period * number_of_periods)) * dt
    sol = sim.solve(t_eval, inputs={"Frequency [Hz]": frequency})
    # Extract final two periods of the solution
    time = sol["Time [s]"].entries[-3 * samples_per_period - 1 :]
    current = sol["Current [A]"].entries[-3 * samples_per_period - 1 :]
    voltage = sol["Terminal voltage [V]"].entries[-3 * samples_per_period - 1 :]
    # FFT
    current_fft = fft(current)
    voltage_fft = fft(voltage)
    # Get index of first harmonic
    idx = np.argmax(np.abs(current_fft))
    impedance = -voltage_fft[idx] / current_fft[idx]
    impedances_time.append(impedance)

end_time = timer.time()
time_elapsed = end_time - start_time
print("Time domain method: ", time_elapsed, "s")

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
ax = pbeis.nyquist_plot(impedances_time, ax=ax, label="Time", alpha=0.7)
for i, method in enumerate(methods):
    ax = pbeis.nyquist_plot(
        impedances_freqs[i], ax=ax, label=f"Frequency ({method})", alpha=0.7
    )
ax.legend()
plt.suptitle(f"{model.name}")
plt.savefig(f"figures/{model.name}_time_vs_freq.pdf", dpi=300)
plt.show()
