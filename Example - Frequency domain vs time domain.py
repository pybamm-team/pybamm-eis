# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 20:08:31 2022

@author: rish3
"""

'''
In this example we find the EIS model of a simple diffusion problem using 
the time domain and the frequency domain for comparison.
'''
import pybamm
import numpy as np
from EIS_from_model import nyquist_plot, EIS
import scipy.fft
import scipy.sparse
import time

#First we solve in the time domain

start_freq = 0.01
end_freq = 100
num_points = 15
start_time = time.time()

answers = []
ws = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), num_points))


for omega in ws:
    #Set up the model and simulate with Pybamm
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
            return A * pybamm.sin(omega * t)
        
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
    
    dt = 2 * np.pi/omega/samples_per_period
    t_eval = np.array(range(0, 1+samples_per_period*(num_periods+skip_periods))) * dt
    solution = solver.solve(model, t_eval)
    
    
    #pybamm.dynamic_plot(solution, ["Concentration", "Surface concentration", "Applied flux"])
    
    t = solution["Time"].entries  # array of size `Nt`
    c = solution["Surface concentration"].entries  # array of size `Nx` by `Nt`
    
    #Fourier Transform to find impedence
    c_hat = scipy.fft.fft(c[skip_periods*samples_per_period:]-1)
    x = np.linspace(0, 1/dt, samples_per_period*num_periods)
    i = A * np.sin(omega * t)
    i_hat = scipy.fft.fft(i[skip_periods*samples_per_period:])
    
    
    
    '''
    #The following is useful for plots of current and voltage and their
    Fourier transforms if desired
    
    import matplotlib.pyplot as plt
    plt.plot(t[skip_periods*samples_per_period:], c[skip_periods*samples_per_period:]-1)
    plt.show()
    plt.plot(t[skip_periods*samples_per_period:], i[skip_periods*samples_per_period:])
    plt.show()
    plt.plot(x[:200], np.abs(c_hat[:200]))
    plt.show()
    plt.plot(x[:200], np.abs(i_hat[:200]))
    plt.show()
    '''
    
    index = np.argmax(np.abs(i_hat))
    
    z = -c_hat[index]/i_hat[index]
    answers.append(z)

end_time = time.time()
timer = end_time - start_time
print(timer)
#create a plot of the answers
nyquist_plot(answers)
print(answers)
print(ws)

#Now we solve in the frequency domain, demonstrating the difference in speed 
#while obtaining the same answers.

#set up the model
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

#Get the matrices and vector b from the model
y0 = model.concatenated_initial_conditions.entries  # vector of initial conditions
J = model.jac_rhs_algebraic_eval(0, y0, []).sparse()  #  call the Jacobian and return a (sparse) matrix

b = model.rhs_algebraic_eval(0, y0, [])
M = model.mass_matrix.entries

#Add an extra row with just one entry. This is important as voltage must be
# the second to last entry and current last when using EIS_from_model.
extra_entry = scipy.sparse.csr_matrix([[0]])
M = scipy.sparse.block_diag([M, extra_entry])
extra_entry[0, 0] = 1
J = scipy.sparse.block_diag([J, extra_entry])
size = np.shape(J)[0]
b = np.reshape(np.append(b, 1), [size, 1])

#Solve using a direct method and plot the answers
answers, ws, timer = EIS(M, J, b, 0.01, 100, 15, method = 'direct')
nyquist_plot(answers)
print(timer)
print(answers)
print(ws)
