import pybamm
import numpy as np
import pybammeis
from scipy.fft import fft
import pytest


def test_compare_methods():
    # Set up
    model = pybamm.lithium_ion.SPM(options={"surface form": "differential"}, name="SPM")
    parameter_values = pybamm.ParameterValues("Marquis2019")
    frequencies = np.logspace(-4, 2, 30)

    # Time domain
    I_app = 50 * 1e-3
    number_of_periods = 20
    samples_per_period = 16

    def current_function(t):
        return I_app * pybamm.sin(
            2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t
        )

    parameter_values["Current function [A]"] = current_function

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
        current = sol["Current [A]"].entries[-3 * samples_per_period - 1 :]
        voltage = sol["Voltage [V]"].entries[-3 * samples_per_period - 1 :]
        # FFT
        current_fft = fft(current)
        voltage_fft = fft(voltage)
        # Get index of first harmonic
        idx = np.argmax(np.abs(current_fft))
        impedance = -voltage_fft[idx] / current_fft[idx]
        impedances_time.append(impedance)
    impedances_time = np.array(impedances_time)

    # Frequency domain
    methods = ["direct"]
    impedances_freqs = {}
    for method in methods:
        eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
        impedances_freq = eis_sim.solve(frequencies, method)
        impedances_freqs[method] = impedances_freq

    for method in methods:
        assert np.allclose(
            impedances_time.real, impedances_freqs[method].real, rtol=1e-1
        )
        assert np.allclose(
            impedances_time.imag, impedances_freqs[method].imag, rtol=1e-1
        )


def test_solve_with_inputs():
    model = pybamm.lithium_ion.DFN(
        options={"working electrode": "positive", "surface form": "differential"}
    )
    parameter_values = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
    parameter_values.update(
        {
            "Positive electrode double-layer capacity [F.m-2]": pybamm.InputParameter(
                "C_dl"
            ),
        },
    )
    eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
    frequencies = np.logspace(-4, 4, 30)
    eis_sim.solve(frequencies, inputs={"C_dl": 0.1})


def test_solve_with_initial_soc():
    model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
    parameter_values = pybamm.ParameterValues("Chen2020")
    eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
    frequencies = np.logspace(-2, 2, 10)

    # Solve at two different SOCs
    z_05 = eis_sim.solve(frequencies, initial_soc=0.5)
    z_09 = eis_sim.solve(frequencies, initial_soc=0.9)

    # Results should be different at different SOCs
    assert z_05.shape == (10,)
    assert not np.allclose(z_05, z_09)


def test_initial_soc_in_constructor():
    model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
    parameter_values = pybamm.ParameterValues("Chen2020")
    frequencies = np.logspace(-2, 2, 10)

    # SOC set in constructor
    eis_sim1 = pybammeis.EISSimulation(
        model, parameter_values=parameter_values, initial_soc=0.5
    )
    z1 = eis_sim1.solve(frequencies)

    # SOC set in solve
    eis_sim2 = pybammeis.EISSimulation(model, parameter_values=parameter_values)
    z2 = eis_sim2.solve(frequencies, initial_soc=0.5)

    np.testing.assert_allclose(z1, z2)


def test_initial_soc_voltage_string():
    model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
    parameter_values = pybamm.ParameterValues("Chen2020")
    eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
    frequencies = np.logspace(-2, 2, 10)

    z = eis_sim.solve(frequencies, initial_soc="3.8 V")
    assert z.shape == (10,)


def test_initial_soc_caching():
    """Repeated solves at the same SOC skip redundant initial-condition rebuilds.

    PyBaMM's Simulation.build uses an eSOH fingerprint internally, so calling
    it with the same (initial_soc, inputs) is essentially free. We verify that
    the full EIS result is stable across repeated calls and that changing the
    SOC produces different impedance.
    """
    model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
    parameter_values = pybamm.ParameterValues("Chen2020")
    eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
    frequencies = np.logspace(-2, 2, 5)

    # Solve twice at the same SOC — results must be identical
    z1 = eis_sim.solve(frequencies, initial_soc=0.5)
    z2 = eis_sim.solve(frequencies, initial_soc=0.5)
    np.testing.assert_allclose(z1, z2)

    # Different SOC must give different impedance
    z3 = eis_sim.solve(frequencies, initial_soc=0.9)
    assert not np.allclose(z1, z3)

    # None should use the last-built SOC (no change)
    z4 = eis_sim.solve(frequencies, initial_soc=None)
    np.testing.assert_allclose(z3, z4)


def test_initial_soc_with_inputs():
    """SOC rebuild accounts for input parameters that affect capacity."""
    model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values.update(
        {
            "Negative electrode active material volume fraction": (
                pybamm.InputParameter("eps_n")
            ),
        },
    )
    eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
    frequencies = np.logspace(-2, 2, 5)

    # Same SOC, different active material fraction → different capacity
    # → different initial concentrations → different impedance
    z1 = eis_sim.solve(
        frequencies, initial_soc=0.5, inputs={"eps_n": 0.75}
    )
    z2 = eis_sim.solve(
        frequencies, initial_soc=0.5, inputs={"eps_n": 0.5}
    )
    assert not np.allclose(z1, z2)


def test_bad_method():
    model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})
    eis_sim = pybammeis.EISSimulation(model)
    frequencies = np.logspace(-4, 4, 30)
    with pytest.raises(ValueError, match="'method' must be"):
        eis_sim.solve(frequencies, method="bad_method")
