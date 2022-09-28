# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 01:30:52 2022

@author: rish3
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt
from EIS_from_model import EIS, nyquist_plot

model = pybamm.lithium_ion.SPM()

def set_up_model_for_eis(model, inplace=True):
    """
    Set up model so that current and voltage are states. 
    This formulation is suitable for EIS calculations in 
    the frequency domain. 
    
    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        Model to set up for EIS.
    inplace: bool, optional
        If True, modify the model in place. Otherwise, return a
        new model. Default is True.
    """ 
    pybamm.logger.info(
        "Start setting up {} for EIS".format(model.name)
    )    
    
    # Set up inplace vs not inplace
    if inplace:
        # any changes to model attributes will change new_model attributes
        # since they point to the same object
        new_model = model
    else:
        # create a copy of the model
        new_model = model.new_copy()

    # Create a voltage variable
    V_cell = pybamm.Variable("Terminal voltage variable")
    new_model.variables["Terminal voltage variable"] = V_cell
    V = new_model.variables["Terminal voltage [V]"]

    # Add an algebraic equation for the voltage variable
    new_model.algebraic[V_cell] = V_cell - V
    new_model.initial_conditions[
        V_cell
    ] = new_model.param.p.U_ref - new_model.param.n.U_ref

    # Now make current density a variable
    # To do so, we replace all instances of the current density in the
    # model with a current density variable, which is obtained from the
    # FunctionControl submodel

    # Create the FunctionControl submodel and extract variables
    external_circuit_variables = (
        pybamm.external_circuit.FunctionControl(
            model.param, None, model.options, control="algebraic"
        ).get_fundamental_variables()
    )

    # Perform the replacement
    symbol_replacement_map = {
        new_model.variables[name]: variable
        for name, variable in external_circuit_variables.items()
    }
    # Don't replace initial conditions, as these should not contain
    # Variable objects
    replacer = pybamm.SymbolReplacer(
        symbol_replacement_map, process_initial_conditions=False
    )
    replacer.process_model(new_model, inplace=True)

    # Update the algebraic equation and initial conditions for
    # FunctionControl
    # This creates an algebraic equation for the current,
    # together with the appropriate guess for the initial condition.
    # External circuit submodels are always equations on the current
    i_cell = new_model.variables["Current density variable"]
    I = new_model.variables["Current [A]"]
    I_applied = pybamm.FunctionParameter(
        "Current function [A]", {"Time [s]": pybamm.t * new_model.param.timescale}
    )
    new_model.algebraic[i_cell] = I - I_applied
    new_model.initial_conditions[
        i_cell
    ] = new_model.param.current_with_time
    
    pybamm.logger.info(
        "Finish setting up {} for EIS".format(model.name)
    )    
    
    return new_model

new_model = set_up_model_for_eis(model, inplace=False)

models = [model, new_model]

sols = []

parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Current function [A]"] = 2.5

var_pts = {
            "x_n": 5,
            "x_s": 5,
            "x_p": 5,
            "r_n": 5,
            "r_p": 5,
}

for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)
    sol = sim.solve([0, 3600])
    sols.append(sol)
    
pybamm.dynamic_plot(sols)

model = sols[1].all_models[0]  # get the discretised model from the second simulation
inds = {"Current density variable": None, "Terminal voltage variable": None}
for key in inds.keys():
    variable = model.variables[key]
    variable_y_indices = np.arange(variable.first_point, variable.last_point)
    inds[key] = variable_y_indices
    
    
I_typ = parameter_values.evaluate(model.param.I_typ)
sols[1].y[inds["Current density variable"], :] * I_typ

model = sols[1].all_models[0]
y0 = model.concatenated_initial_conditions.entries  # vector of initial conditions
J = model.jac_rhs_algebraic_eval(0, y0, []).sparse()  #  call the Jacobian and return a (sparse) matrix
plt.spy(J)
plt.show()

'''
row = inds["Current density variable"]
col = np.array([0])
data = np.array([-1])  
b = csc_matrix((data, (row, col)), shape=y0.shape).todense()
'''

b = np.zeros(y0.shape[0])
b[-1] = -1

M = model.mass_matrix.entries
answers, ws, timer = EIS(M, J, b, 1, 1000, 100, method = 'bicgstab')
nyquist_plot(answers)
print(timer)

