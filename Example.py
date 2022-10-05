# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 20:08:31 2022

@author: rish3
"""

import pybamm
import numpy as np
from EIS_from_model import nyquist_plot
import scipy.fft
import time

start_freq = 1
end_freq = 100
num_points = 7
start_time = time.time()

answers = []
ws = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), num_points))


for omega in ws:
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
        
    
    # Choose amplitude 
    A = 2
    
    # Define parameter values object
    param = pybamm.ParameterValues(
        {
            "Applied flux": applied_flux_function(A, omega),
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
    ##SOLVE in time domain
    # Choose solver
    solver = pybamm.ScipySolver()
    
    # Example: simulate for 10/omega seconds
    samples_per_period = 128
    num_periods = 5
    
    skip_periods = int(np.floor(4*omega))
    
    dt = 1/omega/samples_per_period
    t_eval = np.array(range(0, 1+samples_per_period*(num_periods+skip_periods))) * dt
    solution = solver.solve(model, t_eval)
    
    
    #pybamm.dynamic_plot(solution, ["Concentration", "Surface concentration", "Applied flux"])
    
    t = solution["Time"].entries  # array of size `Nt`
    c = solution["Surface concentration"].entries  # array of size `Nx` by `Nt`
    
    #Fouries transform skipping the first 2 periods
    c_hat = scipy.fft.fft(c[skip_periods*samples_per_period:]-1)
    x = np.linspace(0, 1/dt, samples_per_period*num_periods)
    i = A * np.sin(2 * np.pi * omega * t)
    i_hat = scipy.fft.fft(i[skip_periods*samples_per_period:])
    
    
    import matplotlib.pyplot as plt
    plt.plot(t[skip_periods*samples_per_period:], c[skip_periods*samples_per_period:]-1)
    plt.show()
    plt.plot(t[skip_periods*samples_per_period:], i[skip_periods*samples_per_period:])
    plt.show()
    plt.plot(x[:200], np.abs(c_hat[:200]))
    plt.show()
    plt.plot(x[:200], np.abs(i_hat[:200]))
    plt.show()
    
    index = np.argmax(np.abs(i_hat))
    
    z = c_hat[index]/i_hat[index]
    print(x[index])
    answers.append(z)

end_time = time.time()
timer = end_time - start_time
print(timer)
nyquist_plot(answers)
print(answers)
print(ws)