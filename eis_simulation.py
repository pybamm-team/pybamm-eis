import pybamm
import numpy as np
import numerical_methods as nm
import preconditioners
import time
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix
from utils import SymbolReplacer


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
        M = self.built_model.mass_matrix.entries
        self.y0 = self.built_model.concatenated_initial_conditions.entries
        J = self.built_model.jac_rhs_algebraic_eval(
            0, self.y0, []
        ).sparse()  # call the Jacobian and return a (sparse) matrix
        # Convert to csc for efficiency in later methods
        self.M = csc_matrix(M)
        self.J = csc_matrix(J)
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
        replacer = SymbolReplacer(
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
            The method used to calculate the impedance. Can be 'direct', 'prebicgstab',
            'bicgstab' or 'cg'. Default is 'direct'.

        Returns
        -------
        solution : array-like
            The impedances at the given frequencies.
        """

        if method == "direct":
            zs = []
            for frequency in frequencies:
                A = 1.0j * 2 * np.pi * frequency * self.timescale * self.M - self.J
                lu = splu(A)
                x = lu.solve(self.b)
                # The model is set up such that the voltage is the penultimate
                # entry and the current density variable is the final entry
                z = -x[-2][0] / x[-1][0]
                zs.append(z)
        elif method in ["prebicgstab", "bicgstab", "cg"]:
            zs, frequencies = self.iterative_method(frequencies, method=method)
        else:
            raise NotImplementedError(
                "'method' must be 'direct', 'prebicgstab', 'bicgstab' or 'cg', ",
                f"but is '{method}'",
            )

        # Note: the current density variable is dimensionless so we need
        # to scale by the current scale from the model to get true impedance
        self.solution = np.array(zs) / self.current_scale

        return self.solution

    def iterative_method(
        self, frequencies, method="prebicgstab", preconditioner=preconditioners.ELU
    ):
        """
        Compute the impedance at the given frequencies by solving problem

        .. math::
            i \omega \tau M x = J x + b

        using an iterative method, where i is the imagianary unit, \omega
        is the frequency, \tau is the model timescale, M is the mass matrix,
        J is the Jacobian, x is the state vector, and b gives a periodic
        forcing in the current.

        Parameters
        ----------
        frequencies : array-like
            The frequencies at which to compute the impedance.
        method : str, optional
            The method used to calculate the impedance. Can be:
            'cg' - conjugate gradient - only use for Hermitian matrices
            'bicgstab' - use bicgstab with no preconditioner
            'prebicgstab' - use bicgstab with a preconditioner, this is the
            default.
        preconditioner: function, optional
            A function that calculates a preconditioner from A, M, J and the previous
            preconditioner. Returns L, U, triangular. Only relevent when using prebicgstab.
            Default is ELU (which is normally the best). Return None for U if only L is
            being used.

        Returns
        -------
        solution : array-like
            The impedances at the given frequencies.
        ws : list
            Frequencies evaluated at.

        """
        start_freq = frequencies[0]
        end_freq = frequencies[-1]

        solution = []
        ws = []
        w = start_freq
        next_freq_pos = 1
        
        L = None
        U = None
        LUt = 0
        t = 1
        st = None

        start_point = self.b

        
        iters = []
        while w <= end_freq:
            if w >= 0.99*frequencies[next_freq_pos]:
                next_freq_pos += 1
            
            w_log_increment_max = np.log(frequencies[next_freq_pos]) - np.log(w)
            A = 1.0j * 2 * np.pi * w * self.timescale * self.M - self.J
            num_iters = 0
            stored_vals = []
            ns = []

            if method == "prebicgstab":
                et = time.process_time()
                if st:
                    t = et - st

                if LUt <= t:
                    LUstart_time = time.process_time()
                    L = splu(A)
                    start_point = L.solve(self.b)
                    LUt = time.process_time() - LUstart_time

                st = time.process_time()

            def callback(xk):
                nonlocal num_iters
                num_iters += 1
                stored_vals.append(xk[-1])
                ns.append(num_iters)

            if method == "bicgstab":
                c = nm.bicgstab(A, self.b, start_point=start_point, callback=callback)
            elif method == "prebicgstab":
                c = nm.prebicgstab(
                    A, self.b, L, U, start_point=start_point, callback=callback
                )
            elif method == "cg":
                c = nm.conjugate_gradient(
                    A, self.b, start_point=start_point, callback=callback
                )

            # The model is set up such that the voltage is the penultimate
            # entry and the current density variable is the final entry
            V = c[-2]
            I = c[-1]
            Z = -V / I
            solution.append(Z)

            # find the errors between the value of v after each iteration of
            # bicgstab and the final value
            es = np.abs(np.array(stored_vals) - V)

            # create a list of the number of iterations. This combines with the
            # previous to give corresponding lists with the error from the answer
            # and the number of iterations that were subsequently taken.
            ns = num_iters + 1 - np.array(ns)

            old_c = np.array(c)
            if len(solution) == 1:
                w_log_increment = float(w_log_increment_max)
                start_point = c
            else:
                old_increment = float(w_log_increment)
                kappa = np.abs(V - start_point[-2]) / w_log_increment**2
                if kappa == 0:
                    kappa = 0.001
                ys = []
                for j, e in enumerate(es):
                    y = (
                        2
                        * ns[j]
                        / (
                            -w_log_increment
                            + np.sqrt((w_log_increment) ** 2 + 4 * (e + 0.01) / kappa)
                        )
                    )
                    ys.append(y)
                y_min = min(ys)
                if ys[-1] == y_min:
                    n_val = ns[-1] + 1
                    w_log_increment = min(n_val / y_min[0], w_log_increment_max)
                else:
                    w_log_increment = min(
                        ns[ys.index(y_min)] / y_min[0], w_log_increment_max
                    )

                start_point = c + w_log_increment / old_increment * (c - old_c)

            multiplier = np.exp(w_log_increment)

            ws.append(w)
            iters.append(num_iters)

            w = w * multiplier
        return solution, ws
