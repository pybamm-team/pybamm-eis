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
c = pybamm.Variable("Concentration", domain="rod")
D = pybamm.Parameter("Diffusivity")
c0 = pybamm.Parameter("Initial concentration")

j = pybamm.FunctionParameter("Applied flux", {"Time": pybamm.t})

N = - D * pybamm.grad(c)  # The flux 
dcdt = - pybamm.div(N)  # The right hand side of the PDE
model.rhs = {c: dcdt}  # Add to model

model.boundary_conditions = {
    c: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (-j / D, "Neumann"),
    }
}

model.initial_conditions = {c: c0}

model.variables = {
    "Concentration": c, 
    "Surface concentration": pybamm.surf(c),
    "Applied flux": j,
    "Time": pybamm.t,
}

geometry = {"rod": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}

def applied_flux_function(A, omega):
    "Flux must return a function of time only"
    def applied_flux(t):
        return A * pybamm.sin(2 * np.pi * omega * t)
    
    return applied_flux


# Choose amplitude and frequency 
A = 2
omega = 5

# Define parameter values object
param = pybamm.ParameterValues(
    {
        "Applied flux": applied_flux_function(A, omega),
        "Applied flux": 1,
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

answers, ws, timer = EIS(model, 5, 5, 1, method = 'thomas')
nyquist_plot(answers)              
print(timer)
##SOLVE in time domain
# Choose solver
solver = pybamm.ScipySolver()

# Example: simulate for 10/omega seconds
simulation_time = 10/omega  # end time in seconds
npts = int(60 * simulation_time * omega)  # need enough timesteps to resolve output
t_eval = np.linspace(0, simulation_time, npts)
solution = solver.solve(model, t_eval)


pybamm.dynamic_plot(solution, ["Concentration", "Surface concentration", "Applied flux"])

t = solution["Time"].entries  # array of size `Nt`
c = solution["Surface concentration"].entries  # array of size `Nx` by `Nt`

#Fouries transform skipping the first 50 entries
c_hat = scipy.fft.fft(c[50:])
#??
print(answers)
print(c_hat)