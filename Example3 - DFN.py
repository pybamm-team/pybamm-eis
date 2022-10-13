import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time
from eis_simulation import EISSimulation
from plotting import nyquist_plot
from scipy.fft import fft

# Set up
model = pybamm.lithium_ion.SPMe(options={"surface form": "differential"})

parameter_values = pybamm.ParameterValues("Chen2020")

frequencies = np.logspace(-4, 2, 30)
frequencies = [1]

# Time domain
I_hat = 50 * 1e-3
I = 2 * I_hat
number_of_periods = 20
samples_per_period = 30


def current_function(t):
    return I * pybamm.sin(pybamm.InputParameter("Frequency [Hz]") * t)


parameter_values["Current function [A]"] = current_function

start_time = time.time()

sim = pybamm.Simulation(
    model, parameter_values=parameter_values, solver=pybamm.CasadiSolver(mode="fast")
)

impedances_time = []
for frequency in frequencies:
    period = 2 * np.pi / frequency
    dt = period / samples_per_period
    t_eval = np.array(range(0, 1 + samples_per_period * number_of_periods)) * dt
    sol = sim.solve(t_eval, inputs={"Frequency [Hz]": frequency})
    # Extract final two periods of the solution
    current = sol["Current [A]"].entries[-2 * samples_per_period - 1 :]
    voltage = sol["Terminal voltage"].entries[-2 * samples_per_period - 1 :]
    # FFT
    current_fft = fft(current)
    voltage_fft = fft(voltage)
    print()
    # Get index of first harmonic
    idx = np.argmax(np.abs(current_fft))
    impedance = -voltage_fft[idx] / current_fft[idx]
    impedances_time.append(impedance)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time domain method: ", time_elapsed, "s")

# Frequency domain
start_time = time.time()
eis_sim = EISSimulation(model, parameter_values=parameter_values)
impedances_freq = eis_sim.solve(frequencies)
end_time = time.time()
time_elapsed = end_time - start_time
print("Frequency domain method: ", time_elapsed, "s")

# Compare
_, ax = plt.subplots()
ax = nyquist_plot(impedances_time, ax=ax, label="Time")
ax = nyquist_plot(impedances_freq, ax=ax, label="Frequency")
print(impedances_time[0].real / impedances_freq[0].real)
ax.legend()
plt.show()
