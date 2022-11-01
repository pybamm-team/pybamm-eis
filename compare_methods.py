import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as timer
from eis_simulation import EISSimulation
from plotting import nyquist_plot
from scipy.fft import fft

# Set up
model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

parameter_values = pybamm.ParameterValues("Chen2020")

frequencies = np.logspace(-4, 2, 30)

# Time domain
I = 50 * 1e-3
number_of_periods = 20
samples_per_period = 16
plot = False  # whether to plot results inside the loop


def current_function(t):
    return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)


parameter_values["Current function [A]"] = current_function

start_time = timer.time()

sim = pybamm.Simulation(
    model, parameter_values=parameter_values, solver=pybamm.ScipySolver()
)

impedances_time = []
for frequency in frequencies:
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
    # Plot
    if plot:
        # sol.plot(["Current [A]", "Terminal voltage [V]"])
        x = np.linspace(0, 1 / dt, len(current_fft))
        _, ax = plt.subplots(2, 2)
        ax[0, 0].plot(time, current)
        ax[0, 1].plot(time, voltage)
        ax[1, 0].plot(x, np.abs(current_fft))
        ax[1, 1].plot(x, np.abs(voltage_fft))
        ax[1, 0].set_xlim([0, frequency * 3])
        ax[1, 1].set_xlim([0, frequency * 3])
        plt.show()

end_time = timer.time()
time_elapsed = end_time - start_time
print("Time domain method: ", time_elapsed, "s")

# Frequency domain
methods = ["direct", "prebicgstab"]
impedances_freqs = []
for method in methods:
    start_time = timer.time()
    eis_sim = EISSimulation(model, parameter_values=parameter_values)
    impedances_freq = eis_sim.solve(frequencies, "prebicgstab")
    end_time = timer.time()
    time_elapsed = end_time - start_time
    print(f"Frequency domain ({method}): ", time_elapsed, "s")
    impedances_freqs.append(impedances_freq)

# Compare
_, ax = plt.subplots()
ax = nyquist_plot(impedances_time, ax=ax, label="Time")
for i, method in enumerate(methods):
    ax = nyquist_plot(impedances_freqs[i], ax=ax, label=f"Frequency ({method})")
ax.legend()
plt.show()
