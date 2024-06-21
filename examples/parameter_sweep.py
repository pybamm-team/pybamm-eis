import pbeis
import pybamm
import matplotlib.pyplot as plt

# Load model
model = pybamm.lithium_ion.DFN(
    options={"working electrode": "positive", "surface form": "differential"}
)

# Load parameters
parameter_values = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")


def j0(c_e, c_s_surf, c_s_max, T):
    j0_ref = pybamm.Parameter(
        "Positive electrode reference exchange-current density [A.m-2]"
    )
    c_e_init = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")

    return (
        j0_ref
        * (c_e / c_e_init) ** 0.5
        * (c_s_surf / c_s_max) ** 0.5
        * (1 - c_s_surf / c_s_max) ** 0.5
    )


parameter_values.update(
    {
        "Positive electrode reference exchange-current density [A.m-2]": 5,
        "Positive electrode exchange-current density [A.m-2]": pybamm.InputParameter(
            "j0_ref"
        ),
        "Positive electrode double-layer capacity [F.m-2]": pybamm.InputParameter(
            "C_dl"
        ),
    },
    check_already_exists=False,
)

# Create simulation
eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)

# Choose frequencies and calculate impedance, looping over input parameter values
# and adding the results to a Nyquist plot
frequencies = pbeis.logspace(-4, 4, 30)

j0_refs = [1, 5]
C_dls = [0.1, 10]
markers = ["o", "x"]
colors = ["r", "b"]

_, ax = plt.subplots()
for i, j0_ref in enumerate(j0_refs):
    for j, C_dl in enumerate(C_dls):
        eis_sim.solve(frequencies, inputs={"j0_ref": j0_ref, "C_dl": C_dl})
        eis_sim.nyquist_plot(
            ax=ax,
            marker=markers[j],
            color=colors[i],
            alpha=0.5,
            label=f"j0_ref = {j0_ref}, C_dl = {C_dl}",
        )
ax.legend()
plt.show()
