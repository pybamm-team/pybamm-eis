import pybamm
import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix


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
        # Set up the model for EIS
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

        # Extract mass matrix and Jacobian
        solver = pybamm.BaseSolver()
        solver.set_up(self.built_model)
        self.M = self.built_model.mass_matrix.entries
        self.y0 = self.built_model.concatenated_initial_conditions.entries
        self.J = self.built_model.jac_rhs_algebraic_eval(
            0, self.y0, []
        ).sparse()  # call the Jacobian and return a (sparse) matrix
        # Add forcing on the current density variable, which is the
        # final entry by construction
        self.b = np.zeros_like(self.y0)
        self.b[-1] = -1
        # Store time and current scales
        self.timescale = self.built_model.timescale_eval
        self.current_scale = sim.parameter_values.evaluate(model.param.I_typ)

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
        pybamm.logger.info("Start setting up {} for EIS".format(model.name))

        # Make a new copy of the model
        new_model = model.new_copy()

        # Create a voltage variable
        V_cell = pybamm.Variable("Terminal voltage variable")
        new_model.variables["Terminal voltage variable"] = V_cell
        V = new_model.variables["Terminal voltage [V]"]
        # Add an algebraic equation for the voltage variable
        new_model.algebraic[V_cell] = V_cell - V
        new_model.initial_conditions[V_cell] = (
            new_model.param.p.U_ref - new_model.param.n.U_ref
        )

        # Now make current density a variable
        # To do so, we replace all instances of the current density in the
        # model with a current density variable, which is obtained from the
        # FunctionControl submodel

        # Create the FunctionControl submodel and extract variables
        external_circuit_variables = pybamm.external_circuit.FunctionControl(
            model.param, None, model.options, control="algebraic"
        ).get_fundamental_variables()

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

        # Add an algebraic equation for the current density variable
        # External circuit submodels are always equations on the current
        i_cell = new_model.variables["Current density variable"]
        I = new_model.variables["Current [A]"]
        I_applied = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t * new_model.param.timescale}
        )
        new_model.algebraic[i_cell] = I - I_applied
        new_model.initial_conditions[i_cell] = 0

        pybamm.logger.info("Finish setting up {} for EIS".format(model.name))

        return new_model

    def solve(self, frequencies, method="direct"):
        """
        Compute the impedance at the given frequencies by solving problem

        .. math::
            i \omega \tau M x = J x + b

        where i is the imagianary unit, \omega is the frequency, \tau is the model
        timescale, M is the mass matrix, J is the Jacobian, x is the state vector,
        and b gives a periodic forcing in the current.

        Parameters
        ----------
        frequencies : array-like
            The frequencies at which to compute the impedance.
        method : str, optional
            The method used to calculate the impedance. Can be 'direct'.
            Default is 'direct'.

        Returns
        -------
        solution : array-like
            The impedances at the given frequencies.
        """

        # TODO: add other methods
        if method == "direct":
            # Convert to csc for efficiency
            M = csc_matrix(self.M)
            J = csc_matrix(self.J)
            impedances = []
            for frequency in frequencies:
                A = 1.0j * 2 * np.pi * frequency * self.timescale * M - J
                lu = splu(A)
                x = lu.solve(self.b)
                # The model is set up such that the voltage is the penultimate
                # entry and the current density variable is the final entry
                # Note: current density variable is dimensionless so we need
                # include the current scale from the model to get the actual current
                z = -x[-2][0] / x[-1][0] / self.current_scale
                impedances.append(z)
        else:
            raise NotImplementedError("'method' must be 'direct'.")

        self.solution = np.array(impedances)

        return self.solution
