import time

import numpy as np
import pybamm
from casadi import vertcat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import pybammeis


class EISSimulation:
    """
    A Simulation class for easy building and running of PyBaMM EIS simulations
    using a frequency domain approach.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated
    parameter_values: :class:`pybamm.ParameterValues` (optional)
        Parameters and their corresponding numerical values.
    geometry: :class:`pybamm.Geometry` (optional)
        The geometry upon which to solve the model
    submesh_types: dict (optional)
        A dictionary of the types of submesh to use on each subdomain
    var_pts: dict (optional)
        A dictionary of the number of points used by each spatial variable
    spatial_methods: dict (optional)
        A dictionary of the types of spatial method to use on each
        domain (e.g. pybamm.FiniteVolume)
    """

    def __init__(
        self,
        model,
        parameter_values=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
    ):
        # Set attributes
        self.model_name = model.name
        self.set_up_time = None
        self.solve_time = None
        timer = pybamm.Timer()

        # Set up the model for EIS
        pybamm.logger.info(f"Start setting up {self.model_name} for EIS")
        self.model = self.set_up_model_for_eis(model)

        # Create and build a simulation to conviniently build the model
        parameter_values = parameter_values or model.default_parameter_values
        parameter_values["Current function [A]"] = 0
        sim = pybamm.Simulation(
            self.model,
            geometry=geometry,
            parameter_values=parameter_values,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
        )
        sim.build()
        self.built_model = sim.built_model

        # Get scale factor for the impedance (PyBaMM model may have set scales for the
        # voltage and current variables)
        V_scale = getattr(self.model.variables["Voltage [V]"], "scale", 1)
        I_scale = getattr(self.model.variables["Current [A]"], "scale", 1)
        self.z_scale = parameter_values.evaluate(V_scale / I_scale)

        # Set setup time
        self.set_up_time = timer.time()
        pybamm.logger.info(f"Finished setting up {self.model_name} for EIS")
        pybamm.logger.info(f"Set-up time: {self.set_up_time}")

    def set_up_model_for_eis(self, model):
        """
        Set up model so that current and voltage are states.
        This formulation is suitable for EIS calculations in
        the frequency domain.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Model to set up for EIS.
        """
        pybamm.logger.info(f"Start setting up {self.model_name} for EIS")

        # Make a new copy of the model
        new_model = model.new_copy()

        # Create a voltage variable
        V_cell = pybamm.Variable("Voltage variable [V]")
        new_model.variables["Voltage variable [V]"] = V_cell
        V = new_model.variables["Voltage [V]"]
        # Add an algebraic equation for the voltage variable
        new_model.algebraic[V_cell] = V_cell - V
        new_model.initial_conditions[V_cell] = new_model.param.ocv_init

        # Now make current density a variable
        # To do so, we replace all instances of the current in the model with a current
        # variable, which is obtained from the FunctionControl submodel

        # Create the FunctionControl submodel and extract variables
        external_circuit_variables = pybamm.external_circuit.FunctionControl(
            model.param, None, model.options, control="algebraic"
        ).get_fundamental_variables()

        # TODO: remove SymbolReplacer and use PyBaMM's "set up for experiment"

        # Perform the replacement
        symbol_replacement_map = {
            new_model.variables[name]: variable
            for name, variable in external_circuit_variables.items()
        }
        # Don't replace initial conditions, as these should not contain
        # Variable objects
        replacer = pybammeis.SymbolReplacer(
            symbol_replacement_map, process_initial_conditions=False
        )
        replacer.process_model(new_model, inplace=True)

        # Add an algebraic equation for the current variable
        # External circuit submodels are always equations on the current
        I_var = new_model.variables["Current variable [A]"]
        I_ = new_model.variables["Current [A]"]
        I_applied = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t}
        )
        new_model.algebraic[I_var] = I_ - I_applied
        new_model.initial_conditions[I_var] = 0

        pybamm.logger.info(f"Finish setting up {self.model_name} for EIS")

        return new_model

    def _build_matrix_problem(self, inputs_dict=None):
        """
        Build the mass matrix, Jacobian, vector of initial conditions, and forcing
        vector for the impedance problem.

        Parameters
        ----------
        inputs_dict : dict, optional
            Any input parameters to pass to the model when solving
        """
        pybamm.logger.info(f"Start constructing matrix problem for {self.model_name}")

        # If necessary convert inputs to a casadi vector
        model = self.built_model
        inputs_dict = inputs_dict or {}
        if model.convert_to_format == "casadi":
            inputs = vertcat(*[x for x in inputs_dict.values()])
        else:
            inputs = inputs_dict

        # Extract mass matrix and Jacobian
        solver = pybamm.BaseSolver()
        solver.set_up(model, inputs=inputs_dict)
        M = model.mass_matrix.entries
        self.y0 = model.concatenated_initial_conditions.evaluate(0, inputs=inputs_dict)
        J = model.jac_rhs_algebraic_eval(
            0, self.y0, inputs
        ).sparse()  # call the Jacobian and return a (sparse) matrix
        # Convert to csc for efficiency in later methods
        self.M = csc_matrix(M)
        self.J = csc_matrix(J)
        # Add forcing on the current density variable, which is the
        # final entry by construction
        self.b = np.zeros_like(self.y0)
        self.b[-1] = -1

        pybamm.logger.info(f"Finish constructing matrix problem for {self.model_name}")

    def solve(self, frequencies, method="direct", inputs=None):
        """
        Compute the impedance at the given frequencies by solving problem

        .. math::
            i \\omega \tau M x = J x + b

        where i is the imagianary unit, \\omega is the frequency, \tau is the model
        timescale, M is the mass matrix, J is the Jacobian, x is the state vector,
        and b gives a periodic forcing in the current.

        Parameters
        ----------
        frequencies : array-like
            The frequencies at which to compute the impedance.
        method : str, optional
            The method used to calculate the impedance. Can be 'direct', 'prebicgstab',
            or 'bicgstab'. Default is 'direct'.
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        solution : array-like
            The impedances at the given frequencies.
        """

        pybamm.logger.info(f"Start calculating impedance for {self.model_name}")
        timer = pybamm.Timer()

        self._build_matrix_problem(inputs_dict=inputs)

        if method == "direct":
            zs = []
            for frequency in frequencies:
                A = 1.0j * 2 * np.pi * frequency * self.M - self.J
                lu = splu(A)
                x = lu.solve(self.b)
                # The model is set up such that the voltage is the penultimate
                # entry and the current density variable is the final entry
                z = -x[-2][0] / x[-1][0]
                zs.append(z)
        elif method in ["prebicgstab", "bicgstab"]:
            zs = self.iterative_method(frequencies, method=method)
        else:
            raise ValueError(
                "'method' must be 'direct', 'prebicgstab' or 'bicgstab', ",
                f"but is '{method}'",
            )

        self.solution = np.array(zs) * self.z_scale

        # Store solve time as an attribute
        self.solve_time = timer.time()
        pybamm.logger.info(f"Finished calculating impedance for {self.model_name}")
        pybamm.logger.info(f"Solve time: {self.solve_time}")

        return self.solution

    def iterative_method(self, frequencies, method="prebicgstab"):
        """
        Compute the impedance at the given frequencies by solving problem

        .. math::
            i \\omega \tau M x = J x + b

        using an iterative method, where i is the imagianary unit, \\omega
        is the frequency, \tau is the model timescale, M is the mass matrix,
        J is the Jacobian, x is the state vector, and b gives a periodic
        forcing in the current.

        Parameters
        ----------
        frequencies : array-like
            The frequencies at which to compute the impedance.
        method : str, optional
            The method used to calculate the impedance. Can be:
            'bicgstab' - use bicgstab with no preconditioner
            'prebicgstab' - use bicgstab with a preconditioner, this is
            the default.

        Returns
        -------
        zs : array-like
            The impedances at the given frequencies.
        """
        # Allocate solve times for preconditioner
        if method == "prebicgstab":
            lu_time = 0
            solve_time = 0

        # Loop over frequencies
        zs = []
        sol = self.b
        iters_per_frequency = []

        for frequency in frequencies:
            # Reset per-frequency iteration counter
            num_iters = 0

            # Construct the matrix A(frequency)
            A = 1.0j * 2 * np.pi * frequency * self.M - self.J

            def callback(xk):
                """
                Increments the number of iterations in the call to the 'method'
                functions.
                """
                nonlocal num_iters
                num_iters += 1

            if method == "bicgstab":
                sol = pybammeis.bicgstab(A, self.b, start_point=sol, callback=callback)
            elif method == "prebicgstab":
                # Update preconditioner based on solve time
                if lu_time <= solve_time:
                    lu_start_time = time.process_time()
                    lu = splu(A)
                    sol = lu.solve(self.b)
                    lu_time = time.process_time() - lu_start_time

                # Solve
                solve_start_time = time.process_time()
                sol = pybammeis.prebicgstab(
                    A, self.b, lu, start_point=sol, callback=callback
                )
                solve_time = time.process_time() - solve_start_time

            else:
                raise ValueError

            # Store number of iterations at this frequency
            iters_per_frequency.append(num_iters)

            # The model is set up such that the voltage is the penultimate
            # entry and the current density variable is the final entry
            z = -sol[-2][0] / sol[-1][0]
            zs.append(z)

        return zs

    def nyquist_plot(self, **kwargs):
        """
        A method to quickly creates a nyquist plot using the results of the simulation.
        Calls :meth:`pybammeis.nyquist_plot`.

        Parameters
        ----------
        kwargs
            Keyword arguments, passed to pybammeis.nyquist_plot.
        """
        return pybammeis.nyquist_plot(self.solution, **kwargs)
