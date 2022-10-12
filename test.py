import pybamm

model = pybamm.lithium_ion.SPMe(options={"surface form": "differential"})


parameter_values = pybamm.ParameterValues("Chen2020")

experiment = pybamm.Experiment(
    [
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until 10 mA",
            "Rest for 2 hours",
        ),
    ]
)
exp_sim = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
)
exp_sol = exp_sim.solve()
exp_sol.plot()
