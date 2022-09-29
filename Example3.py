# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 01:15:01 2022

@author: rish3
"""

import pybamm
import numpy as np
from EIS_from_model import EIS, nyquist_plot

model = pybamm.BaseModel()

x = pybamm.SpatialVariable("x", domain="rod", coord_sys="cartesian")
c_hat = pybamm.Variable("Concentration (FT)", domain="rod")
D = pybamm.Parameter("Diffusivity")
c0 = pybamm.Parameter("Initial concentration")

j_hat = pybamm.Parameter("Applied flux (FT)")
j_hat_var = pybamm.Variable("Applied flux variable")

N = - D * pybamm.grad(c_hat)  # The flux 
dcdt = - pybamm.div(N)  # The right hand side of the PDE
model.rhs = {c_hat: dcdt}  # Add to model

model.algebraic = {j_hat_var: j_hat_var - j_hat}

model.boundary_conditions = {
    c_hat: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (-j_hat_var / D, "Neumann"),
    }
}

model.initial_conditions = {c_hat: c0, j_hat_var: j_hat}

model.variables = {
    "Concentration (FT)": c_hat, 
    "Applied flux (FT)": j_hat,
    "Applied flux variable": j_hat_var,
}

geometry = {"rod": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}

# Define parameter values object
param = pybamm.ParameterValues(
    {
        "Applied flux (FT)": 1,
        "Diffusivity": 1,
        "Initial concentration": 1,
    },
)

param.process_model(model)
param.process_geometry(geometry)

submesh_types = {"rod": pybamm.Uniform1DSubMesh}
var_pts = {x: 8}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {"rod": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)

disc.process_model(model)

solver = pybamm.CasadiSolver()
solver.set_up(model)

y0 = model.concatenated_initial_conditions.entries  # vector of initial conditions
J = model.jac_rhs_algebraic_eval(0, y0, []).sparse()  #  call the Jacobian and return a (sparse) matrix

variable = model.variables["Applied flux variable"]
variable_y_indices = np.arange(variable.first_point, variable.last_point)
variable_y_indices

from scipy.sparse import csc_matrix

row = variable_y_indices
col = np.array([0])
data = np.array([-1])
b = csc_matrix((data, (row, col)), shape=y0.shape).todense()

M = model.mass_matrix.entries


answers, ws, timer = EIS(M, J, b, 1, 1000, 10, method = 'prebicgstab')
nyquist_plot(answers)
print(timer)