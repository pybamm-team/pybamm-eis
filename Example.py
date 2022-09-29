# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 20:08:31 2022

@author: rish3
"""

import pybamm
import numpy as np
from EIS_from_model import EIS, nyquist_plot
import scipy.fft

model = pybamm.BaseModel()

x = pybamm.SpatialVariable("x", domain="rod", coord_sys="cartesian")
c_hat = pybamm.Variable("Concentration (FT)", domain="rod")
D = pybamm.Parameter("Diffusivity")
c0 = pybamm.Parameter("Initial concentration")

j_hat = pybamm.Parameter("Applied flux (FT)")

N = - D * pybamm.grad(c_hat)  # The flux 
dcdt = - pybamm.div(N)  # The right hand side of the PDE
model.rhs = {c_hat: dcdt}  # Add to model

model.boundary_conditions = {
    c_hat: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (-j_hat / D, "Neumann"),
    }
}

model.initial_conditions = {c_hat: c0}

model.variables = {
    "Concentration (FT)": c_hat, 
    "Applied flux (FT)": j_hat,
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
var_pts = {x: 100}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {"rod": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)

disc.process_model(model)

solver = pybamm.ScipySolver()
solver.set_up(model)

y0 = model.concatenated_initial_conditions.entries  # vector of initial conditions
J = model.jac_rhs_algebraic_eval(0, y0, []).sparse()  #  call the Jacobian and return a (sparse) matrix

b = model.rhs_algebraic_eval(0, y0, [])
M = model.mass_matrix.entries

answers, ws, timer = EIS(M, J, b, 1, 1000, 10, method = 'direct')
nyquist_plot(answers)
print(timer)


#(-0.12602100399465593+0.12634292282544524j)