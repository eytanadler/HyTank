"""
@File    :   boil_off.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : Components for computing and integrationg the boil-off and heater ODEs
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import warnings
from copy import deepcopy

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import scipy.interpolate
import scipy.integrate
import openmdao.api as om
from openmdao.core.analysis_error import AnalysisError

# ==============================================================================
# Extension modules
# ==============================================================================
from hytank.H2_properties import HydrogenProperties
from hytank.utilities.constants import GRAV_CONST, UNIVERSAL_GAS_CONST, MOLEC_WEIGHT_H2
from hytank.utilities import Integrator


# Thermophysical hydrogen properties to use in the model
H2_prop = HydrogenProperties()

# Sometimes OpenMDAO's bound enforcement doesn't work properly,
# so enforce these bounds within compute methods to avoid divide
# by zero errors.
LIQ_HEIGHT_FRAC_LOWER_ENFORCE = 1e-7
LIQ_HEIGHT_FRAC_UPPER_ENFORCE = 1.0 - 1e-7


class BoilOff(om.Group):
    """
    Time-integrated properties of the ullage and bulk liquid due to heat
    and mass flows into, out of, and within the liquid hydrogen tank.
    The model used is heavily based on work in Eugina Mendez Ramos's thesis
    (http://hdl.handle.net/1853/64797). See Chapter 4 and Appendix E
    for more relevant details.

    Due to geometric computations, this model can get tempermental when the
    tank is nearly empty or nearly full. If used in an optimization problem,
    it may help to constrain the tank fill level to be greater than 1% or so.

    WARNING: Do not modify or connect anything to the initial integrated delta state values
             ("integ.delta_m_gas_initial", "integ.delta_m_liq_initial", etc.). They must
             remain zero for the initial tank state to be the expected value. Set the initial
             tank condition using the BoilOff options.

    Inputs
    ------
    radius : float
        Inner radius of the cylinder. This value does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    m_dot_gas_out : float
        Mass flow rate of gaseous hydrogen out of the ullage; could be for venting
        or gaseous hydrogen consumption (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid hydrogen out of the tank; this is where fuel being consumed
        is bookkept, assuming it is removed from the tank as a liquid (vector, kg/s)
    Q_gas : float
        Heat flow rate through the tank walls into the ullage (vector, W)
    Q_liq : float
        Heat flow rate through the tank walls into the bulk liquid (vector, W)
    P_heater : float
        Power added to resistive heater in the bulk liquid (vector, W)

    Outputs
    -------
    m_gas : float
        Mass of the gaseous hydrogen in the tank ullage (vector, kg)
    m_liq : float
        Mass of liquid hydrogen in the tank (vector, kg)
    T_gas : float
        Temperature of the gaseous hydrogen in the ullage (vector, K)
    T_liq : float
        Temperature of the bulk liquid hydrogen (vector, K)
    P_gas : float
        Pressure of the gas in the ullage (vector, Pa)
    fill_level : float
        Fraction of tank volume filled with liquid (vector)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    fill_level_init : float
        Initial fill level (in range 0-1) of the tank, default 0.95
        to leave space for boil-off gas; 5% adopted from Millis et al. 2009 (scalar, dimensionless)
    ullage_T_init : float
        Initial temperature of gas in ullage, default 21 K (scalar, K)
    ullage_P_init : float
        Initial pressure of gas in ullage, default 150,000 Pa; ullage pressure must be higher than ambient
        to prevent air leaking in and creating a combustible mixture (scalar, Pa)
    liquid_T_init : float
        Initial temperature of bulk liquid hydrogen, default 20 K (scalar, K)
    end_cap_depth_ratio : float
        End cap depth divided by cylinder radius. 1 gives hemisphere, 0.5 gives 2:1 semi ellipsoid.
        Must be in the range 0-1 (inclusive). By default 1.0.
    heater_Q_add_init : float
        Initial heat being added to the tank by the heater. Can overwrite this value by setting
        the initial value input in the integrator. By default 0.0 (scalar, W)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("fill_level_init", default=0.95, desc="Initial fill level")
        self.options.declare("ullage_T_init", default=21.0, desc="Initial ullage temp (K)")
        self.options.declare("ullage_P_init", default=1.5e5, desc="Initial ullage pressure (Pa)")
        self.options.declare("liquid_T_init", default=20.0, desc="Initial bulk liquid temp (K)")
        self.options.declare(
            "end_cap_depth_ratio", lower=0.0, upper=1.0, default=1.0, desc="End cap depth / cylinder radius"
        )
        self.options.declare("heater_Q_add_init", default=0.0, types=float, desc="Initial heat input from heater")

    def setup(self):
        nn = self.options["num_nodes"]

        # Compute the time derivatives of the states as a function of the states and other inputs
        self.add_subsystem(
            "ode",
            FullODE(num_nodes=nn, end_cap_depth_ratio=self.options["end_cap_depth_ratio"]),
            promotes_inputs=[
                "radius",
                "length",
                "P_heater",
                "m_dot_gas_out",
                "m_dot_liq_out",
                "Q_gas",
                "Q_liq",
                "m_gas",
                "m_liq",
                "T_gas",
                "T_liq",
            ],
            promotes_outputs=[
                "fill_level",
                "P_gas",
            ],
        )

        # The initial tank states are specified indirectly by the fill_level_init, ullage_T_init, ullage_P_init,
        # and liquid_T_init options, along with the input tank radius and length. We can't connect a component
        # directly to the integrator's inputs because those initial values are linked between phases. Thus, we
        # use a bit of a trick where we actually integrate the change in the state values since the beginning
        # of the mission and then add their correct initial values in the add_init_state_values component.
        integ = self.add_subsystem(
            "integ",
            Integrator(num_nodes=nn, diff_units="s", time_setup="duration", method="bdf3"),
        )
        integ.add_integrand("delta_m_gas", rate_name="m_dot_gas", units="kg", val=0, start_val=0)
        integ.add_integrand("delta_m_liq", rate_name="m_dot_liq", units="kg", val=0, start_val=0)
        integ.add_integrand("delta_T_gas", rate_name="T_dot_gas", units="K", val=0, start_val=0)
        integ.add_integrand("delta_T_liq", rate_name="T_dot_liq", units="K", val=0, start_val=0)
        integ.add_integrand("delta_V_gas", rate_name="V_dot_gas", units="m**3", val=0, start_val=0)
        integ.add_integrand(
            "Q_add", rate_name="Q_add_dot", units="W", val=0, start_val=self.options["heater_Q_add_init"]
        )

        self.add_subsystem(
            "add_init_state_values",
            InitialTankStateModification(
                num_nodes=nn,
                fill_level_init=self.options["fill_level_init"],
                ullage_T_init=self.options["ullage_T_init"],
                ullage_P_init=self.options["ullage_P_init"],
                liquid_T_init=self.options["liquid_T_init"],
                end_cap_depth_ratio=self.options["end_cap_depth_ratio"],
            ),
            promotes_inputs=["radius", "length"],
            promotes_outputs=["m_liq", "m_gas", "T_liq", "T_gas"],
        )

        # Connect the integrated delta states to the component that increments them by their computed initial values
        self.connect("integ.delta_m_gas", "add_init_state_values.delta_m_gas")
        self.connect("integ.delta_m_liq", "add_init_state_values.delta_m_liq")
        self.connect("integ.delta_T_gas", "add_init_state_values.delta_T_gas")
        self.connect("integ.delta_T_liq", "add_init_state_values.delta_T_liq")
        self.connect("integ.delta_V_gas", "add_init_state_values.delta_V_gas")
        self.connect("integ.Q_add", "ode.Q_add")

        # Connect the ODE to the integrator
        self.connect("ode.m_dot_gas", "integ.m_dot_gas")
        self.connect("ode.m_dot_liq", "integ.m_dot_liq")
        self.connect("ode.T_dot_gas", "integ.T_dot_gas")
        self.connect("ode.T_dot_liq", "integ.T_dot_liq")
        self.connect("ode.V_dot_gas", "integ.V_dot_gas")
        self.connect("ode.Q_add_dot", "integ.Q_add_dot")

        self.connect("add_init_state_values.V_gas", "ode.V_gas")

        # Set a solver specifically for this component in an attempt to increase robustness
        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options["solve_subsystems"] = False
        self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["restart_from_successful"] = True
        self.nonlinear_solver.options["maxiter"] = 30
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["atol"] = 1e-9
        self.nonlinear_solver.options["rtol"] = 1e-12
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
            bound_enforcement="scalar", alpha=1.0, iprint=0, print_bound_enforce=False
        )

        # Do a bit of a hack to know which solver is calling this
        # group's guess_nonlinear. This information is used to know
        # whether or not to run the guess_nonlinear IVP solver.
        self._in_my_solve_nl = False

    def _solve_nonlinear(self):
        """
        Overriding this function so _in_my_solve_nl will be True
        when this group's guess_nonlinear is called by this group's
        solver and False when this group's guess_nonlienar is called
        by any other solver in the model.
        """
        self._in_my_solve_nl = True
        try:
            super()._solve_nonlinear()
        finally:
            self._in_my_solve_nl = False

    def _mpi_print_stuff(self, text):
        """
        Print arbitrary text with the appropriate prefix and indentation
        for the Newton solver that belongs to this group.
        """
        if self.nonlinear_solver.options["iprint"] > 0 and (
            self.nonlinear_solver._system().comm.rank == 0 or os.environ.get("USE_PROC_FILES")
        ):
            prefix = self.nonlinear_solver._solver_info.prefix

            print(f"{prefix}{text}")

    def guess_nonlinear(self, inputs, outputs, resids):
        """
        Guessing reasonable initial states for this system is incredibly
        important because solving the initial value problem all at once
        with a Newton solver very easily jumps outside the bounds of the
        hydrogen property surrogate models to unreasonable states. This
        uses a two-pronged approach to guess the initial states:

            1. If guess_nonlinear is being called by the Newton solver
               that belongs to this group, try using the SciPy initial
               value solver at this phase's initial conditions to solve
               the problem given the initial conditions and inputs. If
               this succeeds, it gives very good initial guesses. This
               solver is far more robust for solving this problem than
               the all-at-once Newton. If it fails, raise an error to
               skip the Newton solve of this iteration since it's very
               unlikely to work.

            2. If guess_nonlinear is being called by something else,
               just set the states to a constant value of their initial
               condition in this phase.
        """
        # If the model is already converged, don't change anything
        norm = resids.get_norm()
        zero_resids = np.all(resids.asarray() < 1e-14)
        if norm < 1e-2 and not zero_resids:
            return

        # Initial tank properties as specified by this group's options
        r = inputs["ode.level_calc.radius"].item()
        L = inputs["ode.level_calc.length"].item()
        fill_init = self.options["fill_level_init"]
        T_gas_init = self.options["ullage_T_init"]
        P_gas_init = self.options["ullage_P_init"]
        T_liq_init = self.options["liquid_T_init"]

        # Check that the initial temperatures are on the correct side of the saturation point
        T_sat = H2_prop.sat_gh2_T(P_gas_init)
        if T_liq_init > T_sat:
            warnings.warn(
                f"Initial liquid temperature of {T_liq_init} K is above the saturation "
                f"temperature of {T_sat} K. The solver is unlikely to converge under these conditions."
            )
        if T_gas_init < T_sat:
            warnings.warn(
                f"Initial ullage temperature of {T_gas_init} K is below the saturation "
                f"temperature of {T_sat} K. The solver is unlikely to converge under these conditions."
            )

        # Compute the initial gas mass from the given initial pressure
        V_tank = 4 / 3 * np.pi * r**3 * self.options["end_cap_depth_ratio"] + np.pi * r**2 * L
        V_gas_init = V_tank * (1 - fill_init)
        m_gas_init = H2_prop.gh2_rho(P_gas_init, T_gas_init).item() * V_gas_init
        m_liq_init = (V_tank - V_gas_init) * H2_prop.lh2_rho(T_liq_init)

        def get_ode_problem(num_nodes=1):
            # Set up a problem with the ODE that can be used in an initial value problem solver
            p = om.Problem()
            p.model = FullODE(num_nodes=num_nodes, end_cap_depth_ratio=self.options["end_cap_depth_ratio"])

            # Set model options so that this ODE has the same set options as the group's components
            ode_options = self.ode.boil_off_ode.options
            heater_ode_options = self.ode.heater_ode.options
            p.model_options["*"] = {
                "heater_boil_frac": ode_options["heater_boil_frac"],
                "heat_transfer_C_const": ode_options["heat_transfer_C_const"],
                "heat_transfer_n_const": ode_options["heat_transfer_n_const"],
                "sigmoid_fac": ode_options["sigmoid_fac"],
                "heater_rate_const": heater_ode_options["heater_rate_const"],
            }

            p.setup()

            # If the heater fractions are set to "input", set them appropriately
            if ode_options["heater_boil_frac"] == "input":
                p.set_val("boil_off_ode.heater_boil_frac", inputs["ode.boil_off_ode.heater_boil_frac"])
            if heater_ode_options["heater_rate_const"] == "input":
                p.set_val("heater_ode.heater_rate_const", inputs["ode.heater_ode.heater_rate_const"])

            # Use a Newton solver for the implicit component in FullODE, otherwise just needs a forward pass
            p.model.liq_height_calc.linear_solver = om.DirectSolver()
            p.model.liq_height_calc.nonlinear_solver = om.NewtonSolver()
            p.model.liq_height_calc.nonlinear_solver.options["solve_subsystems"] = False
            # Turning on err_on_non_converge prevents solve_ivp from accepting unconverged values
            p.model.liq_height_calc.nonlinear_solver.options["err_on_non_converge"] = True
            p.model.liq_height_calc.nonlinear_solver.options["maxiter"] = 5
            p.model.liq_height_calc.nonlinear_solver.options["iprint"] = 0
            p.model.liq_height_calc.nonlinear_solver.options["atol"] = 1e-9
            p.model.liq_height_calc.nonlinear_solver.options["rtol"] = 1e-12
            p.model.liq_height_calc.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
                bound_enforcement="scalar", alpha=1.0, iprint=0, print_bound_enforce=False
            )

            # Set the tank geometry parameters
            p.set_val("radius", r, units="m")
            p.set_val("length", L, units="m")

            return p

        # ==============================================================================
        # Try solving an initial value problem to get guesses for the states
        # ==============================================================================
        # Run the initial value problem solver only when this guess_nonlinear is called by this group's solver
        if self._in_my_solve_nl:
            p = get_ode_problem(num_nodes=1)

            # Get initial values of the integrated states within this phase
            u_init = np.array(
                [
                    m_gas_init + inputs["integ.delta_m_gas_initial"].item(),
                    m_liq_init + inputs["integ.delta_m_liq_initial"].item(),
                    T_gas_init + inputs["integ.delta_T_gas_initial"].item(),
                    T_liq_init + inputs["integ.delta_T_liq_initial"].item(),
                    V_gas_init + inputs["integ.delta_V_gas_initial"].item(),
                    inputs["integ.Q_add_initial"].item(),
                ]
            )

            # Create splines for the heat and mass flow inputs so they can be
            # computed at any time within the domain
            input_splines = {}
            t_span = (0, inputs["integ.duration"].item())  # sec
            t = np.linspace(*t_span, self.options["num_nodes"])
            for i_name in ["m_dot_gas_out", "m_dot_liq_out", "Q_liq", "Q_gas", "P_heater"]:
                input_splines[i_name] = scipy.interpolate.Akima1DInterpolator(t, inputs[i_name])

            # Function to compute the state time derivatives as a function of the states and time
            def state_deriv_function(t, u):
                # Set state inputs
                p.set_val("m_gas", u[0], units="kg")
                p.set_val("m_liq", u[1], units="kg")
                p.set_val("T_gas", u[2], units="K")
                p.set_val("T_liq", u[3], units="K")
                p.set_val("V_gas", u[4], units="m**3")
                p.set_val("Q_add", u[5], units="W")

                # Set control inputs using splines
                p.set_val("m_dot_gas_out", input_splines["m_dot_gas_out"](t), units="kg/s")
                p.set_val("m_dot_liq_out", input_splines["m_dot_liq_out"](t), units="kg/s")
                p.set_val("Q_gas", input_splines["Q_gas"](t), units="W")
                p.set_val("Q_liq", input_splines["Q_liq"](t), units="W")
                p.set_val("P_heater", input_splines["P_heater"](t), units="W")

                p.run_model()

                # Get the time derivatives of the states
                return np.array(
                    [
                        p.get_val("m_dot_gas", units="kg/s").item(),
                        p.get_val("m_dot_liq", units="kg/s").item(),
                        p.get_val("T_dot_gas", units="K/s").item(),
                        p.get_val("T_dot_liq", units="K/s").item(),
                        p.get_val("V_dot_gas", units="m**3/s").item(),
                        p.get_val("Q_add_dot", units="W/s").item(),
                    ]
                )

            # Function to compute the Jacobian of the ODE (df_i / du_j)
            def jac_function(t, u):
                # Run the model first
                _ = state_deriv_function(t, u)

                # Compute the total derivatives
                return p.compute_totals(
                    of=["m_dot_gas", "m_dot_liq", "T_dot_gas", "T_dot_liq", "V_dot_gas", "Q_add_dot"],
                    wrt=["m_gas", "m_liq", "T_gas", "T_liq", "V_gas", "Q_add"],
                    return_format="array",
                )

            self._mpi_print_stuff("Solving boil-off IVP to generate initial guesses...")
            sol = scipy.integrate.solve_ivp(
                state_deriv_function,
                t_span,
                u_init,
                method="BDF",
                jac=jac_function,
                t_eval=t,
                rtol=1e-3,
                atol=1e-6,
            )

            # If the IVP solver worked, set the values and get outta here
            if sol.success:
                self._mpi_print_stuff("    ...succeeded")

                # States
                outputs["m_gas"] = m_gas = sol.y[0]
                outputs["m_liq"] = m_liq = sol.y[1]
                outputs["T_gas"] = T_gas = sol.y[2]
                outputs["T_liq"] = T_liq = sol.y[3]
                outputs["add_init_state_values.V_gas"] = V_gas = sol.y[4]
                outputs["integ.Q_add"] = Q_add = sol.y[5]

                outputs["integ.delta_m_gas"] = outputs["m_gas"] - m_gas_init
                outputs["integ.delta_m_liq"] = outputs["m_liq"] - m_liq_init
                outputs["integ.delta_T_gas"] = outputs["T_gas"] - T_gas_init
                outputs["integ.delta_T_liq"] = outputs["T_liq"] - T_liq_init
                outputs["integ.delta_V_gas"] = V_gas - V_gas_init

                outputs["integ.delta_m_gas_final"] = outputs["integ.delta_m_gas"][-1]
                outputs["integ.delta_m_liq_final"] = outputs["integ.delta_m_liq"][-1]
                outputs["integ.delta_T_gas_final"] = outputs["integ.delta_T_gas"][-1]
                outputs["integ.delta_T_liq_final"] = outputs["integ.delta_T_liq"][-1]
                outputs["integ.delta_V_gas_final"] = outputs["integ.delta_V_gas"][-1]
                outputs["integ.Q_add_final"] = outputs["integ.Q_add"][-1]

                # Other properties that can be computed from states
                outputs["fill_level"] = 1 - V_gas / V_tank
                outputs["P_gas"] = H2_prop.gh2_P(m_gas / V_gas, T_gas)

                # Determine the geometric properties and state rates throughout this phase
                p_geo = get_ode_problem(num_nodes=self.options["num_nodes"])

                p_geo.set_val("m_gas", m_gas, units="kg")
                p_geo.set_val("m_liq", m_liq, units="kg")
                p_geo.set_val("T_gas", T_gas, units="K")
                p_geo.set_val("T_liq", T_liq, units="K")
                p_geo.set_val("V_gas", V_gas, units="m**3")
                p_geo.set_val("Q_add", Q_add, units="W")

                # Set control inputs using splines
                p_geo.set_val("m_dot_gas_out", input_splines["m_dot_gas_out"](t), units="kg/s")
                p_geo.set_val("m_dot_liq_out", input_splines["m_dot_liq_out"](t), units="kg/s")
                p_geo.set_val("Q_gas", input_splines["Q_gas"](t), units="W")
                p_geo.set_val("Q_liq", input_splines["Q_liq"](t), units="W")
                p_geo.set_val("P_heater", input_splines["P_heater"](t), units="W")

                p_geo.run_model()

                outputs["ode.interface_params.A_interface"] = p_geo.get_val("interface_params.A_interface")
                outputs["ode.interface_params.L_interface"] = p_geo.get_val("interface_params.L_interface")
                outputs["ode.interface_params.A_dry"] = p_geo.get_val("interface_params.A_dry")
                outputs["ode.interface_params.A_wet"] = p_geo.get_val("interface_params.A_wet")
                outputs["ode.liq_height_calc.h_liq_frac"] = p_geo.get_val("liq_height_calc.h_liq_frac")

                outputs["ode.boil_off_ode.m_dot_gas"] = p_geo.get_val("boil_off_ode.m_dot_gas")
                outputs["ode.boil_off_ode.m_dot_liq"] = p_geo.get_val("boil_off_ode.m_dot_liq")
                outputs["ode.boil_off_ode.T_dot_gas"] = p_geo.get_val("boil_off_ode.T_dot_gas")
                outputs["ode.boil_off_ode.T_dot_liq"] = p_geo.get_val("boil_off_ode.T_dot_liq")
                outputs["ode.boil_off_ode.V_dot_gas"] = p_geo.get_val("boil_off_ode.V_dot_gas")
                outputs["ode.heater_ode.Q_add_dot"] = p_geo.get_val("heater_ode.Q_add_dot")

                return

            self._mpi_print_stuff("    ...failed")

            # If it didn't work, raise an error to avoid running the Newton solver.
            # The Newton solver almost never works if this IVP solver fails, so don't
            # waste time and risk blowing up the solution.
            raise AnalysisError("Boil-off initial value problem solver guess failed")

        self._mpi_print_stuff("Skipping boil-off initial guess process")

        # ==============================================================================
        # If IVP didn't work, set all the states to a constant value
        # ==============================================================================
        # Include the initial integrator values in case this group lives in the middle of a mission
        outputs["m_gas"] = m_gas_init + inputs["integ.delta_m_gas_initial"]
        outputs["m_liq"] = m_liq_init + inputs["integ.delta_m_liq_initial"]
        outputs["T_gas"] = T_gas_init + inputs["integ.delta_T_gas_initial"]
        outputs["T_liq"] = T_liq_init + inputs["integ.delta_T_liq_initial"]
        outputs["add_init_state_values.V_gas"] = V_gas_init + inputs["integ.delta_V_gas_initial"]

        outputs["integ.delta_m_gas"] = inputs["integ.delta_m_gas_initial"]
        outputs["integ.delta_m_liq"] = inputs["integ.delta_m_liq_initial"]
        outputs["integ.delta_T_gas"] = inputs["integ.delta_T_gas_initial"]
        outputs["integ.delta_T_liq"] = inputs["integ.delta_T_liq_initial"]
        outputs["integ.delta_V_gas"] = inputs["integ.delta_V_gas_initial"]


class FullODE(om.Group):
    """
    This group combines all the necessary components for the computation
    of the state derivatives as a function of the states. This includes not only
    the boil-off ODE, but also the geometric computations for liquid height,
    fill level, and tank areas and lengths.

    Inputs
    ------
    radius : float
        Inner radius of the cylinder. This value does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    m_dot_gas_out : float
        Mass flow rate of gaseous hydrogen out of the ullage; could be for venting
        or gaseous hydrogen consumption (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid hydrogen out of the tank; this is where fuel being consumed
        is bookkept, assuming it is removed from the tank as a liquid (vector, kg/s)
    Q_gas : float
        Heat flow rate through the tank walls into the ullage (vector, W)
    Q_liq : float
        Heat flow rate through the tank walls into the bulk liquid (vector, W)
    Q_add : float
        Additional heat added directly to the bulk liquid by a heater this heat
        is assumed to go directly to boiling the liquid, rather than also heating
        the bulk liquid as Q_liq does (vector, W)
    m_gas : float
        Mass of the gaseous hydrogen in the tank ullage (vector, kg)
    m_liq : float
        Mass of liquid hydrogen in the tank (vector, kg)
    T_gas : float
        Temperature of the gaseous hydrogen in the ullage (vector, K)
    T_liq : float
        Temperature of the bulk liquid hydrogen (vector, K)
    V_gas : float
        Volume of the ullage (vector, m^3)

    Outputs
    -------
    m_dot_gas : float
        Rate of change of ullage gas mass (vector, kg/s)
    m_dot_liq : float
        Rate of change of bulk liquid mass (vector, kg/s)
    T_dot_gas : float
        Rate of change of ullage gas temperature (vector, K/s)
    T_dot_liq : float
        Rate of change of bulk liquid temperature (vector, K/s)
    V_dot_gas : float
        Rate of change of ullage volume (vector, m^3/s)
    P_gas : float
        Pressure of the gas in the ullage (vector, Pa)
    fill_level : float
        Fraction of tank volume filled with liquid (vector)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    end_cap_depth_ratio : float
        End cap depth divided by cylinder radius. 1 gives hemisphere, 0.5 gives 2:1 semi ellipsoid.
        Must be in the range 0-1 (inclusive). By default 1.0.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare(
            "end_cap_depth_ratio", lower=0.0, upper=1.0, default=1.0, desc="End cap depth / cylinder radius"
        )

    def setup(self):
        nn = self.options["num_nodes"]
        depth_ratio = self.options["end_cap_depth_ratio"]

        # Compute the fill level in the tank
        self.add_subsystem(
            "level_calc",
            BoilOffFillLevelCalc(num_nodes=nn, end_cap_depth_ratio=depth_ratio),
            promotes_inputs=["radius", "length", "V_gas"],
            promotes_outputs=["fill_level"],
        )

        # Compute the required geometric properties
        self.add_subsystem(
            "liq_height_calc",
            LiquidHeight(num_nodes=nn, end_cap_depth_ratio=depth_ratio),
            promotes_inputs=["radius", "length"],
        )
        self.add_subsystem(
            "interface_params",
            BoilOffGeometry(num_nodes=nn, end_cap_depth_ratio=depth_ratio),
            promotes_inputs=["radius", "length"],
        )
        self.connect("fill_level", "liq_height_calc.fill_level")
        self.connect("liq_height_calc.h_liq_frac", "interface_params.h_liq_frac")

        # Compute the heat added to the liquid by the heater
        self.add_subsystem(
            "heater_ode",
            HeaterODE(num_nodes=nn),
            promotes_inputs=["P_heater", "Q_add"],
            promotes_outputs=["Q_add_dot"],
        )

        # Compute the ODE equations to be integrated
        self.add_subsystem(
            "boil_off_ode",
            LH2BoilOffODE(num_nodes=nn),
            promotes_inputs=[
                "m_dot_gas_out",
                "m_dot_liq_out",
                "Q_gas",
                "Q_liq",
                "Q_add",
                "m_gas",
                "m_liq",
                "T_gas",
                "T_liq",
                "V_gas",
            ],
            promotes_outputs=["m_dot_gas", "m_dot_liq", "T_dot_gas", "T_dot_liq", "V_dot_gas", "P_gas"],
        )
        self.connect("interface_params.A_interface", "boil_off_ode.A_interface")
        self.connect("interface_params.L_interface", "boil_off_ode.L_interface")

        # Set defaults for inputs promoted from multiple sources
        self.set_input_defaults("radius", 1.0, units="m")
        self.set_input_defaults("length", 0.5, units="m")


class LiquidHeight(om.ImplicitComponent):
    """
    Implicitly compute the height of liquid in the tank.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `. ~~~~~~~~~~~~~~~~~~ ,'      -.- h  -->  h_liq_frac = h / (2 * radius)
         ` -------------- '         -'-

    Inputs
    ------
    fill_level : float
        Fraction of tank volume filled with liquid (vector)
    radius : float
        Inner radius of the cylinder. This value does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)

    Outputs
    -------
    h_liq_frac : float
        Height of the liquid in the tank nondimensionalized by the height of
        the tank; 1.0 indicates the height is two radii (at the top of the tank)
        and 0.0 indicates the height is zero (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    end_cap_depth_ratio : float
        End cap depth divided by cylinder radius. 1 gives hemisphere, 0.5 gives 2:1 semi ellipsoid.
        Must be in the range 0-1 (inclusive). By default 1.0.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare(
            "end_cap_depth_ratio", lower=0.0, upper=1.0, default=1.0, desc="End cap depth / cylinder radius"
        )

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("fill_level", shape=(nn,))
        self.add_input("radius", units="m")
        self.add_input("length", units="m")

        self.add_output("h_liq_frac", val=0.5, shape=(nn,), lower=1e-3, upper=1.0 - 1e-3)

        arng = np.arange(nn)
        self.declare_partials("h_liq_frac", ["radius", "length"], rows=arng, cols=np.zeros(nn))
        self.declare_partials("h_liq_frac", ["h_liq_frac", "fill_level"], rows=arng, cols=arng)

    def apply_nonlinear(self, inputs, outputs, residuals):
        fill = inputs["fill_level"]
        r = inputs["radius"]
        L = inputs["length"]
        h_frac = outputs["h_liq_frac"]
        h_frac[h_frac <= LIQ_HEIGHT_FRAC_LOWER_ENFORCE] = LIQ_HEIGHT_FRAC_LOWER_ENFORCE
        h_frac[h_frac >= LIQ_HEIGHT_FRAC_UPPER_ENFORCE] = LIQ_HEIGHT_FRAC_UPPER_ENFORCE
        h = h_frac * 2 * r

        # For the current guess of the liquid height, compute the
        # volume of fluid in the hemispherical and cylindrical
        # portions of the tank
        V_caps = np.pi * h**2 / 3 * (3 * r - h) * self.options["end_cap_depth_ratio"]

        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment
        V_cyl = r**2 / 2 * (th - np.sin(th)) * L

        # Total tank volume
        V_tank = 4 / 3 * np.pi * r**3 * self.options["end_cap_depth_ratio"] + np.pi * r**2 * L

        # Residual is difference between liquid volume given current
        # height guess and actual liquid volume computed with fill level
        residuals["h_liq_frac"] = V_caps + V_cyl - V_tank * fill

    def linearize(self, inputs, outputs, J):
        fill = inputs["fill_level"]
        r = inputs["radius"]
        L = inputs["length"]
        h_frac = outputs["h_liq_frac"]
        h_frac[h_frac <= LIQ_HEIGHT_FRAC_LOWER_ENFORCE] = LIQ_HEIGHT_FRAC_LOWER_ENFORCE
        h_frac[h_frac >= LIQ_HEIGHT_FRAC_UPPER_ENFORCE] = LIQ_HEIGHT_FRAC_UPPER_ENFORCE
        h = h_frac * 2 * r

        # Compute partials of spherical volume w.r.t. inputs and height
        Vcaps_r = np.pi * h**2 * self.options["end_cap_depth_ratio"]
        Vcaps_h = (2 * np.pi * h * r - np.pi * h**2) * self.options["end_cap_depth_ratio"]

        # Compute partials of cylindrical volume w.r.t. inputs and height
        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment
        th_r = -2 / np.sqrt(1 - (1 - h / r) ** 2) * h / r**2
        th_h = 2 / np.sqrt(1 - (1 - h / r) ** 2) / r

        Vcyl_r = (
            r * (th - np.sin(th)) * L  # pV_cyl / pr
            + r**2 / 2 * (1 - np.cos(th)) * L * th_r  # pV_cyl / pth * pth / pr
        )
        Vcyl_h = r**2 / 2 * (1 - np.cos(th)) * L * th_h
        Vcyl_L = r**2 / 2 * (th - np.sin(th))

        # Total tank volume
        V_tank = 4 / 3 * np.pi * r**3 * self.options["end_cap_depth_ratio"] + np.pi * r**2 * L
        Vtank_r = 4 * np.pi * r**2 * self.options["end_cap_depth_ratio"] + 2 * np.pi * r * L
        Vtank_L = np.pi * r**2

        J["h_liq_frac", "radius"] = Vcaps_r + Vcyl_r - Vtank_r * fill + (Vcaps_h + Vcyl_h) * 2 * outputs["h_liq_frac"]
        J["h_liq_frac", "length"] = Vcyl_L - Vtank_L * fill
        J["h_liq_frac", "fill_level"] = -V_tank
        J["h_liq_frac", "h_liq_frac"] = (Vcaps_h + Vcyl_h) * 2 * r

    def guess_nonlinear(self, inputs, outputs, resids):
        # If the model is already converged, don't change anything
        norm = resids.get_norm()
        if norm < 1e-2 and norm != 0.0:
            return

        # Guess the height initially using a linear approximation of height w.r.t. fill level
        outputs["h_liq_frac"] = inputs["fill_level"]


class BoilOffGeometry(om.ExplicitComponent):
    """
    Compute areas and volumes in the tank from fill level.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `. ~~~~~~~~~~~~~~~~~~ ,'      -.- h  -->  h_liq_frac = h / (2 * radius)
         ` -------------- '         -'-

    Inputs
    ------
    h_liq_frac : float
        Height of the liquid in the tank nondimensionalized by the height of
        the tank; 1.0 indicates the height is two radii (at the top of the tank)
        and 0.0 indicates the height is zero (vector, dimensionless)
    radius : float
        Inner radius of the cylinder. This value does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)

    Outputs
    -------
    A_interface : float
        Area of the surface of the liquid in the tank. This is the area of
        the interface between the ullage and bulk liquid portions, hence
        the name (vector, m^2)
    L_interface : float
        Characteristic length of the interface between the ullage and the
        bulk liquid (vector, m)
    A_wet : float
        The area of the tank's surface touching the bulk liquid (vector, m^2)
    A_dry : float
        The area of the tank's surface touching the ullage (vector, m^2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    end_cap_depth_ratio : float
        End cap depth divided by cylinder radius. 1 gives hemisphere, 0.5 gives 2:1 semi ellipsoid.
        Must be in the range 0-1 (inclusive). By default 1.0.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare(
            "end_cap_depth_ratio", lower=0.0, upper=1.0, default=1.0, desc="End cap depth / cylinder radius"
        )

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("h_liq_frac", shape=(nn,))
        self.add_input("radius", units="m")
        self.add_input("length", units="m")

        self.add_output("A_interface", units="m**2", shape=(nn,), lower=1e-5, val=3.0)
        self.add_output("L_interface", units="m", shape=(nn,), lower=1e-5, val=1.0)
        self.add_output("A_wet", units="m**2", shape=(nn,), lower=1e-5, val=5.0)
        self.add_output("A_dry", units="m**2", shape=(nn,), lower=1e-5, val=5.0)

        arng = np.arange(nn)
        self.declare_partials(["*"], "h_liq_frac", rows=arng, cols=arng)
        self.declare_partials(["*"], ["radius", "length"], rows=arng, cols=np.zeros(nn))

        # Prevent h_liq_frac input from being evaluated at 0 or 1 (made a variable
        # here so can be turned off for unit testing)
        self.adjust_h_liq_frac = True

    def compute(self, inputs, outputs):
        r = inputs["radius"]
        L = inputs["length"]
        h_frac = inputs["h_liq_frac"]
        d_end = self.options["end_cap_depth_ratio"]
        if self.adjust_h_liq_frac:
            h_frac[h_frac <= LIQ_HEIGHT_FRAC_LOWER_ENFORCE] = LIQ_HEIGHT_FRAC_LOWER_ENFORCE
            h_frac[h_frac >= LIQ_HEIGHT_FRAC_UPPER_ENFORCE] = LIQ_HEIGHT_FRAC_UPPER_ENFORCE
        h = h_frac * 2 * r

        # Ellipticity of oblate spheroid (add 1e-9 term to avoid divide by zero and NaN
        # errors in next term when d_end = 0 or d_end = 1)
        e = np.sqrt(1 - d_end**2)
        if e == 0.0:
            e += 1e-9
        elif e == 1.0:
            e -= 1e-9
        A_spheroid = r**2 * np.pi * (2 + d_end**2 / e * np.log((1 + e) / (1 - e)))  # surface area

        # Total area of the tank
        A_tank = A_spheroid + 2 * np.pi * r * L

        # Some useful geometric parameters
        c = 2 * np.sqrt(2 * r * h - h**2)  # chord length of circular segment
        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment

        # Interface area
        outputs["A_interface"] = np.pi * (c / 2) ** 2 * d_end + c * L
        outputs["L_interface"] = c  # take the chord as the characteristic length

        # Approximate the wetted cap area in a way that is exact when d_end is 0 (flat end caps)
        # or 1 (hemispherical end caps). Also do it such that the surface area when full (theta = 2 pi)
        # is always equal to the exact end cap surface area. This is achieved by linearly interpolating
        # "shape functions" that describe the shape of the surface area vs. theta curves when d_end
        # is 0 and 1. These shape functions are normalized such that they equal 1 when theta is 2 pi.
        # The linear interpolation between these two shape functions is then multiplied by the
        # surface area of an oblate spheroid. The oblate spheroid surface area expression is exact
        # for all d_end values between 0 and 1. This achieves the desired behavior.
        A_cap_wet = A_spheroid * (0.5 * (1 - np.cos(th / 2)) * d_end + (th - np.sin(th)) / (2 * np.pi) * (1 - d_end))

        # Wet and dry areas
        outputs["A_wet"] = A_cap_wet + th * r * L
        outputs["A_dry"] = A_tank - outputs["A_wet"]

    def compute_partials(self, inputs, J):
        r = inputs["radius"]
        L = inputs["length"]
        h_frac = inputs["h_liq_frac"]
        d_end = self.options["end_cap_depth_ratio"]
        if self.adjust_h_liq_frac:
            h_frac[h_frac <= LIQ_HEIGHT_FRAC_LOWER_ENFORCE] = LIQ_HEIGHT_FRAC_LOWER_ENFORCE
            h_frac[h_frac >= LIQ_HEIGHT_FRAC_UPPER_ENFORCE] = LIQ_HEIGHT_FRAC_UPPER_ENFORCE
        h = h_frac * 2 * r

        # Ellipticity of oblate spheroid (add 1e-9 term to avoid divide by zero and NaN
        # errors in next term when d_end = 0 or d_end = 1)
        e = np.sqrt(1 - d_end**2)
        if e == 0.0:
            e += 1e-9
        elif e == 1.0:
            e -= 1e-9
        A_spheroid = r**2 * np.pi * (2 + d_end**2 / e * np.log((1 + e) / (1 - e)))  # surface area
        Aspheroid_r = 2 * A_spheroid / r

        # Derivatives of chord and central angle of segment w.r.t. height and radius
        c = 2 * np.sqrt(2 * r * h - h**2)  # chord length of circular segment
        c_r = 2 * h / np.sqrt(2 * r * h - h**2)
        c_h = (2 * r - 2 * h) / np.sqrt(2 * r * h - h**2)

        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment
        th_r = -2 / np.sqrt(1 - (1 - h / r) ** 2) * h / r**2
        th_h = 2 / np.sqrt(1 - (1 - h / r) ** 2) / r

        Acapwet_th = A_spheroid * (0.25 * np.sin(th / 2) * d_end + (1 - np.cos(th)) / (2 * np.pi) * (1 - d_end))

        J["A_interface", "h_liq_frac"] = c_h * (np.pi * c / 2 * d_end + L) * 2 * r
        J["A_interface", "radius"] = c_r * (np.pi * c / 2 * d_end + L) + J["A_interface", "h_liq_frac"] / r * h_frac
        J["A_interface", "length"] = c

        J["L_interface", "h_liq_frac"] = c_h * 2 * r
        J["L_interface", "radius"] = c_r + J["L_interface", "h_liq_frac"] / r * h_frac
        J["L_interface", "length"] *= 0.0

        J["A_wet", "h_liq_frac"] = (Acapwet_th + r * L) * th_h * 2 * r
        J["A_wet", "radius"] = (
            (Acapwet_th + r * L) * th_r
            + Aspheroid_r * (0.5 * (1 - np.cos(th / 2)) * d_end + (th - np.sin(th)) / (2 * np.pi) * (1 - d_end))
            + th * L
            + J["A_wet", "h_liq_frac"] / r * h_frac
        )
        J["A_wet", "length"] = th * r

        J["A_dry", "h_liq_frac"] = -J["A_wet", "h_liq_frac"]
        J["A_dry", "radius"] = Aspheroid_r + 2 * np.pi * L - J["A_wet", "radius"]
        J["A_dry", "length"] = 2 * np.pi * r - J["A_wet", "length"]


class HeaterODE(om.ExplicitComponent):
    """
    Computes the heat input into the tank assuming a resistive heater with thermal inertia.
    The thermal inertia is captured using the following ODE:

        dQ/dt = C (P - Q)

    where Q is the heat input into the tank, P is the electrical power provided to the heater,
    and C is a constant. The ODE says that the heat added to the tank approaches the power added
    to the heater at a rate proportional to the difference between the two. This is roughly
    based in heat transfer physics, where the constant encapsulates the heater's mass, heat
    transfer coefficient, and other parameters.

    Inputs
    ------
    P_heater : float
        Power provided to the heater element (vector, W)
    Q_add : float
        Heat input into the LH2 tank (vector, W)
    heater_rate_const : float
        If the heater_rate_const option is set to \"input\", the constant in the ODE is provided
        as an input, otherwise this input does not exist (scalar, 1/s)

    Outputs
    -------
    Q_add_dot : float
        Time derivative of heat added to tank (vector, W/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    heater_rate_const : float
        Constant of proportionality in the ODE that encapsulates heater mass, heat transfer
        coefficient, and other related parameters. A smaller value corresponds to a slower
        heater response time, by default 1e-3 (scalar, 1/s)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("heater_rate_const", default=1e-3, types=(float, str), desc="Constant in heater ODE")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("P_heater", shape=(nn,), val=0.0, units="W")
        self.add_input("Q_add", shape=(nn,), val=0.0, units="W")
        self.add_output("Q_add_dot", shape=(nn,), val=0.0, units="W/s")

        arng = np.arange(nn)
        self.declare_partials("Q_add_dot", ["P_heater", "Q_add"], rows=arng, cols=arng)

        if self.options["heater_rate_const"] == "input":
            self.add_input("heater_rate_const", val=1e-3, units="1/s")
            self.declare_partials("Q_add_dot", "heater_rate_const")

    def compute(self, inputs, outputs):
        C = self.options["heater_rate_const"]
        if C == "input":
            C = inputs["heater_rate_const"]

        outputs["Q_add_dot"] = C * (inputs["P_heater"] - inputs["Q_add"])

    def compute_partials(self, inputs, J):
        C = self.options["heater_rate_const"]
        if C == "input":
            C = inputs["heater_rate_const"]

        J["Q_add_dot", "P_heater"] = C
        J["Q_add_dot", "Q_add"] = -C

        if self.options["heater_rate_const"] == "input":
            J["Q_add_dot", "heater_rate_const"] = inputs["P_heater"] - inputs["Q_add"]


class LH2BoilOffODE(om.ExplicitComponent):
    """
    Compute the derivatives of the state values for the liquid hydrogen boil-off process
    given the current states values and other related inputs. The states are the mass of
    gaseous and liquid hydrogen, the temperature of the gas and liquid, and the volume
    of gas (volume of the ullage).

    This portion of the code leans on much of the work from Eugina Mendez Ramos's thesis
    (http://hdl.handle.net/1853/64797). See Chapter 4 and Appendix E for more relevant details.

    Inputs
    ------
    m_gas : float
        Mass of the gaseous hydrogen in the tank ullage (vector, kg)
    m_liq : float
        Mass of liquid hydrogen in the tank (vector, kg)
    T_gas : float
        Temperature of the gaseous hydrogen in the ullage (vector, K)
    T_liq : float
        Temperature of the bulk liquid hydrogen (vector, K)
    V_gas : float
        Volume of the ullage (vector, m^3)
    m_dot_gas_out : float
        Mass flow rate of gaseous hydrogen out of the ullage; could be for venting
        or gaseous hydrogen consumption (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid hydrogen out of the tank; this is where fuel being consumed
        is bookkept, assuming it is removed from the tank as a liquid (vector, kg/s)
    Q_gas : float
        Heat flow rate through the tank walls into the ullage (vector, W)
    Q_liq : float
        Heat flow rate through the tank walls into the bulk liquid (vector, W)
    Q_add : float
        Additional heat added directly to the bulk liquid by a heater this heat
        is assumed to go directly to boiling the liquid, rather than also heating
        the bulk liquid as Q_liq does (vector, W)
    A_interface : float
        Area of the surface of the liquid in the tank. This is the area of
        the interface between the ullage and bulk liquid portions, hence
        the name (vector, m^2)
    L_interface : float
        Characteristic length of the interface between the ullage and the
        bulk liquid (vector, m)

    Outputs
    -------
    m_dot_gas : float
        Rate of change of ullage gas mass (vector, kg/s)
    m_dot_liq : float
        Rate of change of bulk liquid mass (vector, kg/s)
    T_dot_gas : float
        Rate of change of ullage gas temperature (vector, K/s)
    T_dot_liq : float
        Rate of change of bulk liquid temperature (vector, K/s)
    V_dot_gas : float
        Rate of change of ullage volume (vector, m^3/s)
    P_gas : float
        Pressure in the ullage (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    heater_boil_frac : float
        The fraction of the heat from the heater that directly induces boil-off. The remaining heat (1 - heater_boil_frac)
        goes to heating the bulk liquid. This value must be between 0 and 1 (inclusive). This option can also be set to
        \"input\", which makes it an input rather than an option so an optimizer can tune it. By default 0.75
    heat_transfer_C_const : float
        Multiplier on the Nusselt number used when computing the convective heat transfer coefficient. By default
        0.27, which is for convection above a cooler horizontal surface (see W. H. McAdams, Heat Transmission 1954).
    heat_transfer_n_const : float
        Exponent on the Prandtl-Grashof product in the Nusselt number calculation. By default 0.25, which is for
        convection above a cooler horizontal surface (see W. H. McAdams, Heat Transmission 1954).
    sigmoid_fac : float
        Factor that multiplies exponent of sigmoid function to change its steepness. The sigmoid function is used to
        turn on and off the bulk boiling and cloud condensation when the liquid or ullage reaches saturation
        conditions. A greater value corresponds to a steeper sigmoid function, which is more accurate but can make
        solving or optimizing more numerically challenging. By default 100.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare(
            "heater_boil_frac", default=0.75, desc="Fraction of heat from heater that goes straight to boiling"
        )
        self.options.declare("heat_transfer_C_const", default=0.27, types=float, desc="Convective C coefficient")
        self.options.declare("heat_transfer_n_const", default=0.25, types=float, desc="Convective n coefficient")
        self.options.declare(
            "sigmoid_fac", default=100.0, desc="Multiplier on exponent in sigmoid for bulk boil and cloud condensation"
        )

    def setup(self):
        # Check options
        self.heater_boil_frac_input = False
        if self.options["heater_boil_frac"] == "input":
            self.heater_boil_frac_input = True
        elif self.options["heater_boil_frac"] > 1.0 or self.options["heater_boil_frac"] < 0.0:
            raise ValueError("heater_boil_frac option must be between 0 and 1 (inclusive)")

        nn = self.options["num_nodes"]
        self.add_input("m_gas", units="kg", shape=(nn,))
        self.add_input("m_liq", units="kg", shape=(nn,))
        self.add_input("T_gas", units="K", shape=(nn,))
        self.add_input("T_liq", units="K", shape=(nn,))
        self.add_input("V_gas", units="m**3", shape=(nn,))
        self.add_input("m_dot_gas_out", units="kg/s", shape=(nn,), val=0.0)
        self.add_input("m_dot_liq_out", units="kg/s", shape=(nn,), val=0.0)
        self.add_input("Q_gas", units="W", shape=(nn,), val=0.0)
        self.add_input("Q_liq", units="W", shape=(nn,), val=0.0)
        self.add_input("Q_add", units="W", shape=(nn,), val=0.0)
        self.add_input("A_interface", units="m**2", shape=(nn,))
        self.add_input("L_interface", units="m", shape=(nn,))
        if self.heater_boil_frac_input:
            self.add_input("heater_boil_frac", val=0.75)

        self.add_output("m_dot_gas", units="kg/s", shape=(nn,), ref=1e-3, val=0.0)
        self.add_output("m_dot_liq", units="kg/s", shape=(nn,), ref=1e-3, val=0.0)
        self.add_output("T_dot_gas", units="K/s", shape=(nn,), ref=1e-3, val=0.0)
        self.add_output("T_dot_liq", units="K/s", shape=(nn,), ref=1e-3, val=0.0)
        self.add_output("V_dot_gas", units="m**3/s", shape=(nn,), ref=1e-5, val=0.0)
        self.add_output("P_gas", units="Pa", shape=(nn,), ref=1e5, val=1e5, lower=5e4, upper=12.5e5)

        arng = np.arange(nn)
        method = "exact"
        self.declare_partials(
            ["m_dot_gas", "m_dot_liq", "V_dot_gas", "T_dot_gas", "T_dot_liq"],
            [
                "Q_liq",
                "Q_gas",
                "Q_add",
                "A_interface",
                "L_interface",
                "T_gas",
                "T_liq",
                "V_gas",
                "m_dot_gas_out",
                "m_dot_liq_out",
                "m_gas",
                "m_liq",
            ],
            rows=arng,
            cols=arng,
            method=method,
        )
        self.declare_partials("P_gas", ["m_gas", "T_gas", "V_gas"], rows=arng, cols=arng, method=method)
        if self.heater_boil_frac_input:
            self.declare_partials("*", "heater_boil_frac", method="fd", step=1e-6)

        # Use this to check if the compute method has been called already with the same inputs
        self.inputs_cache = None

        # Limit on values that go in the exponent of the sigmoid functions
        self.exp_limit = 50.0

        # Hydrogen property surrogate models to use. Add this as a class attribute so that it can
        # be changed during testing to the one from Mendez Ramos's thesis, which enables
        # complex step derivative checking.
        self.H2 = H2_prop

    def _process_inputs(self, inputs):
        """
        Adds a small perturbation to any input states that have values of zero or less and
        shouldn't (either because it's nonphysical or causes divide by zero errors). Returns
        a new dictionary with modified inputs.

        See OpenMDAO issue #2824 for more details on why this might happen despite putting
        bounds on these outputs.
        """
        adjusted_inputs = deepcopy(dict(inputs))
        new_val = 1e-10
        inputs_to_change = ["m_gas", "m_liq", "T_gas", "T_liq", "V_gas", "A_interface", "L_interface"]

        for var in inputs_to_change:
            adjusted_inputs[var][adjusted_inputs[var] <= new_val] = new_val

        return adjusted_inputs

    def compute(self, inputs_orig, outputs):
        inputs = self._process_inputs(inputs_orig)

        # Unpack the states from the inputs
        m_gas = inputs["m_gas"]
        m_liq = inputs["m_liq"]
        T_gas = inputs["T_gas"]
        T_liq = inputs["T_liq"]
        V_gas = inputs["V_gas"]
        self.inputs_cache = deepcopy(dict(inputs))

        m_dot_gas_out = inputs["m_dot_gas_out"]  # gas released for venting or consumption
        m_dot_liq_out = inputs["m_dot_liq_out"]  # liquid leaving the tank (e.g., for fuel to the engines)

        # Heat flows into bulk liquid and ullage
        Q_gas = inputs["Q_gas"]
        Q_liq = inputs["Q_liq"]
        Q_add = inputs["Q_add"]  # heat added to bulk liquid with heater

        A_int = inputs["A_interface"]  # area of the surface of the bulk liquid (the interface)
        L_int = inputs["L_interface"]  # characteristic length of the interface

        heater_boil_frac = self.options["heater_boil_frac"]
        if self.heater_boil_frac_input:
            heater_boil_frac = inputs["heater_boil_frac"]

        # =============================== Compute physical properties ===============================
        # Compute pressure from density and temperature using the hydrogen property surrogate model
        self.P_gas = P_gas = self.H2.gh2_P(m_gas / V_gas, T_gas)

        # Ullage gas properties
        self.h_gas = self.H2.gh2_h(P_gas, T_gas)  # enthalpy
        self.u_gas = self.H2.gh2_u(P_gas, T_gas)  # internal energy
        self.cv_gas = self.H2.gh2_cv(P_gas, T_gas)  # specific heat at constant volume

        # Bulk liquid properties
        self.h_liq = self.H2.lh2_h(T_liq)  # enthalpy
        self.u_liq = self.H2.lh2_u(T_liq)  # internal energy
        self.cp_liq = self.H2.lh2_cp(T_liq)  # specific heat at constant pressure
        self.rho_liq = self.H2.lh2_rho(T_liq)  # density
        self.P_liq = self.H2.lh2_P(T_liq)  # pressure

        # Temperature of the interface assumes saturated hydrogen with same pressure as the ullage
        self.T_int = self.H2.sat_gh2_T(P_gas)  # use saturated GH2 temperature

        # Saturated gas properties at the ullage saturation temperature. Evaluate the saturated gas
        # properties at just T_sat_gas rather than mean film used in Mendez-Ramos because it won't
        # violate the surrogate ranges and hinder robustness. It has a minimal effect on the results.
        self.cp_sat_gas = self.H2.sat_gh2_cp(self.T_int)  # specific heat at constant pressure
        self.visc_sat_gas = self.H2.sat_gh2_viscosity(self.T_int)  # viscosity
        self.k_sat_gas = self.H2.sat_gh2_k(self.T_int)  # thermal conductivity
        self.beta_sat_gas = self.H2.sat_gh2_beta(self.T_int)  # coefficient of thermal expansion
        self.rho_sat_gas = self.H2.sat_gh2_rho(self.T_int)  # density

        # ==================== Compute heat transfer between ullage and interface ====================
        # Compute the heat transfer coefficient for the heat transfer from the ullage to the interface.
        # Evaluate the heat transfer coefficient between the ullage and interface using the saturated gas
        # properties at the mean film temperature (average of ullage and interface temps) because the ullage
        # near the interface is close to saturated.
        # Use constants specified in options
        self.C_const = self.options["heat_transfer_C_const"]
        self.n_const = self.options["heat_transfer_n_const"]

        # Compute the fluid properties for heat transfer
        self.prandtl = self.cp_sat_gas * self.visc_sat_gas / self.k_sat_gas
        self.grashof = (
            GRAV_CONST
            * self.beta_sat_gas
            * self.rho_sat_gas**2
            * np.sqrt((T_gas - self.T_int) ** 2)  # use sqrt of square as absolute value shorthand so complex-safe
            * L_int**3
            / self.visc_sat_gas**2
        )
        self.prandtl[np.real(self.prandtl) < 0] = 0.0
        self.grashof[np.real(self.grashof) < 0] = 0.0
        self.nusselt = self.C_const * (self.prandtl * self.grashof) ** self.n_const
        self.heat_transfer_coeff_gas_int = self.k_sat_gas / L_int * self.nusselt

        # Heat from the environment that goes to heating the walls is likely be small (only a few percent),
        # so we'll ignore it (see Van Dresar paper).
        self.Q_gas_int = Q_gas_int = self.heat_transfer_coeff_gas_int * A_int * (T_gas - self.T_int)

        # ============================================ ODEs ============================================
        # Evaluate the ODEs without any influences from bulk boiling or cloud condensation
        # Mass flows
        self.m_dot_boil_off = (Q_gas_int + Q_add * heater_boil_frac) / (self.h_gas - self.h_liq)
        self.m_dot_liq = -self.m_dot_boil_off - m_dot_liq_out
        self.m_dot_gas = self.m_dot_boil_off - m_dot_gas_out

        # Volume changes
        self.V_dot_liq = self.m_dot_liq / self.rho_liq
        self.V_dot_gas = -self.V_dot_liq

        # Temperature rates
        self.T_dot_liq = (
            Q_liq
            + Q_add * (1 - heater_boil_frac)
            - self.P_gas * self.V_dot_liq
            + self.m_dot_liq * (self.h_liq - self.u_liq)
        ) / (m_liq * self.cp_liq)
        self.T_dot_gas = (
            Q_gas - self.Q_gas_int - P_gas * self.V_dot_gas + self.m_dot_gas * (self.h_gas - self.u_gas)
        ) / (m_gas * self.cv_gas)

        # ================================ Bulk boiling ================================
        # Bulk boiling of liquid occurs when ullage pressure drops to liquid vapor pressure
        self.d_P_liq_d_T = self.H2.lh2_P(T_liq, deriv=True)
        self.m_dot_bulk_boil = (
            MOLEC_WEIGHT_H2
            * V_gas
            / (UNIVERSAL_GAS_CONST * T_gas)
            * (
                self.d_P_liq_d_T * self.T_dot_liq
                - UNIVERSAL_GAS_CONST
                / MOLEC_WEIGHT_H2
                * (
                    self.m_dot_gas * T_gas / V_gas
                    + m_gas * self.T_dot_gas / V_gas
                    - m_gas * T_gas * self.V_dot_gas / V_gas**2
                )
            )
        )

        # Do abs in a way that tracks which signs are flipped so we can accommodate it in the partials
        self.m_dot_bb_sign = np.sign(self.m_dot_bulk_boil)
        self.m_dot_bulk_boil *= self.m_dot_bb_sign

        # Add bulk liquid boiling to the ODEs at time steps where the
        # ullage pressure is at or below the liquid vapor pressure.
        # Use sigmoid function to smoothly turn on/off bulk boil.
        # Adaptively scale the sigmoid based on pressure values to
        # stretch it out so it's not so sharp.
        self.P_diff = self.options["sigmoid_fac"] * (P_gas - self.P_liq) / self.P_liq
        self.idx_P_overflow = (
            np.abs(self.P_diff) > self.exp_limit
        )  # any magnitudes > 50 are limited to prevent overflows in exp
        self.P_diff[self.idx_P_overflow] = self.exp_limit * np.sign(self.P_diff[self.idx_P_overflow])
        self.bulk_boil_multiplier = 1 / (1 + np.exp(self.P_diff))

        # ============================= Cloud condensation =============================
        # Cloud (a.k.a. fog-type or homogeneous) condensation occurs when ullage temp drops to saturation temp
        self.d_T_sat_g_d_P = self.H2.sat_gh2_T(P_gas, deriv=True)
        self.m_dot_cloud_cond = (
            P_gas
            * V_gas
            * MOLEC_WEIGHT_H2
            / (UNIVERSAL_GAS_CONST * T_gas**2)
            * (
                self.d_T_sat_g_d_P
                * UNIVERSAL_GAS_CONST
                / MOLEC_WEIGHT_H2
                * (
                    self.m_dot_gas * T_gas / V_gas
                    + m_gas * self.T_dot_gas / V_gas
                    - m_gas * T_gas * self.V_dot_gas / V_gas**2
                )
                - self.T_dot_gas
            )
        )

        # Do abs in a way that tracks which signs are flipped so we can accommodate it in the partials
        self.m_dot_cc_sign = np.sign(self.m_dot_cloud_cond)
        self.m_dot_cloud_cond *= self.m_dot_cc_sign

        # Add cloud condensation to the ODEs at time steps where the
        # ullage temperature is at or below the ullage saturation point.
        # Use sigmoid function to smoothly turn on/off condensation.
        # Adaptively scale the sigmoid based on temperature values to
        # stretch it out so it's not so sharp.
        self.T_diff = self.options["sigmoid_fac"] * (T_gas - self.T_int) / self.T_int
        self.idx_T_overflow = (
            np.abs(self.T_diff) > self.exp_limit
        )  # any magnitudes > 50 are limited to prevent overflows in exp
        self.T_diff[self.idx_T_overflow] = self.exp_limit * np.sign(self.T_diff[self.idx_T_overflow])
        self.cloud_cond_multiplier = 1 / (1 + np.exp(self.T_diff))

        # ================ Add bulk boiling and cloud condensation to ODEs ================
        # Mass flow additions from bulk boiling and cloud condensation
        self.m_dot_bbcc = (
            self.m_dot_bulk_boil * self.bulk_boil_multiplier - self.m_dot_cloud_cond * self.cloud_cond_multiplier
        )
        self.m_dot_liq_incl_bbcc = self.m_dot_liq - self.m_dot_bbcc
        self.m_dot_gas_incl_bbcc = self.m_dot_gas + self.m_dot_bbcc

        # Volume changes from added boil off mass flows
        self.V_dot_gas_incl_bbcc = self.V_dot_gas + self.m_dot_bbcc / self.rho_liq

        # Temperature changes from added boil off mass flows
        self.T_dot_liq_incl_bbcc = self.T_dot_liq + (
            self.P_gas * self.m_dot_bbcc / self.rho_liq - self.m_dot_bbcc * (self.h_liq - self.u_liq)
        ) / (m_liq * self.cp_liq)
        self.T_dot_gas_incl_bbcc = self.T_dot_gas + (
            -P_gas * self.m_dot_bbcc / self.rho_liq + self.m_dot_bbcc * (self.h_gas - self.u_gas)
        ) / (m_gas * self.cv_gas)

        # Heat for bulk boiling comes from liquid
        self.Q_liq_bulk_boil = -(self.h_gas - self.h_liq) * self.m_dot_bulk_boil * self.bulk_boil_multiplier
        self.T_dot_liq_incl_bbcc += self.Q_liq_bulk_boil / (m_liq * self.cp_liq)

        # Heat from condensation goes to gas
        self.Q_gas_cloud_cond = (self.h_gas - self.h_liq) * self.m_dot_cloud_cond * self.cloud_cond_multiplier
        self.T_dot_gas_incl_bbcc += self.Q_gas_cloud_cond / (m_gas * self.cv_gas)

        # ============================= We got em! =============================
        outputs["m_dot_gas"] = self.m_dot_gas_incl_bbcc
        outputs["m_dot_liq"] = self.m_dot_liq_incl_bbcc
        outputs["T_dot_gas"] = self.T_dot_gas_incl_bbcc
        outputs["T_dot_liq"] = self.T_dot_liq_incl_bbcc
        outputs["V_dot_gas"] = self.V_dot_gas_incl_bbcc

        # Ullage pressure (useful for other stuff)
        outputs["P_gas"] = P_gas

    def compute_partials(self, inputs_orig, J):
        """
        Start by computing all the derivatives without bulk boiling or
        cloud condensation. Then add their influence. Fun stuff.
        """
        inputs = self._process_inputs(inputs_orig)

        # Check that the compute method has been called with the same inputs
        if self.inputs_cache is None:
            self.compute(inputs, {})
        else:
            for name in inputs.keys():
                try:
                    if np.any(inputs[name] != self.inputs_cache[name]):
                        raise ValueError()
                except:
                    self.compute(inputs, {})
                    break

        # Reset any nonzero entries in Jacobian to zero
        for key in J.keys():
            # But skip any output w.r.t. itself (they're in
            # there because of how explicit components work)
            if key[0] == key[1]:
                continue
            # Also skip any keys that are finite differenced
            # because this would make their derivatives wrong
            if "heater_boil_frac" in key[1]:
                continue
            J[key] *= 0.0

        # Unpack the states from the inputs
        m_gas = inputs["m_gas"]
        m_liq = inputs["m_liq"]
        T_gas = inputs["T_gas"]
        T_liq = inputs["T_liq"]
        V_gas = inputs["V_gas"]

        # Heat input
        Q_add = inputs["Q_add"]

        # ============================== Compute geometric quantities ==============================
        A_int = inputs["A_interface"]  # area of the surface of the bulk liquid (the interface)
        L_int = inputs["L_interface"]  # characteristic length of the interface

        heater_boil_frac = self.options["heater_boil_frac"]
        if self.heater_boil_frac_input:
            heater_boil_frac = inputs["heater_boil_frac"]

        # Variable naming convention: d{of variable}__{wrt variable}
        # Outputs are abbreviatted as mg, ml, Tg, Tl, Vg, and P

        # ==============================================================================
        # m_dot_gas (no bulk boiling or cloud condensation yet)
        # ==============================================================================
        # Influence of m_dot_boil_off on computing m_dot_gas
        dmg__m_dot_boil_off = 1.0

        # Interface and heated boil-off effect on m_dot_boil_off
        dmg__Q_gas_int = dmg__m_dot_boil_off / (self.h_gas - self.h_liq)
        dmg__Q_add = dmg__m_dot_boil_off * heater_boil_frac / (self.h_gas - self.h_liq)
        dmg__h_gas = -dmg__m_dot_boil_off * (self.Q_gas_int + Q_add * heater_boil_frac) / (self.h_gas - self.h_liq) ** 2
        dmg__h_liq = dmg__m_dot_boil_off * (self.Q_gas_int + Q_add * heater_boil_frac) / (self.h_gas - self.h_liq) ** 2

        # -------------- Derivative of Q_gas_int w.r.t. inputs --------------
        # Call the heat_transfer_coeff_gas_int variable htc. There are three components to
        # the heat transfer coefficient: k_sat_gas, Nusselt number, and L_int

        # Total derivatives of T_int w.r.t. inputs, which will be needed for both the
        # k_sat_gas and Nusselt number portions
        dT_int__P_gas = self.H2.sat_gh2_T(self.P_gas, deriv=True)
        dP_gas__rho_gas, dP_gas__T_gas = self.H2.gh2_P(m_gas / V_gas, T_gas, deriv=True)
        dP_gas__m_gas = dP_gas__rho_gas / V_gas
        dP_gas__V_gas = -dP_gas__rho_gas * m_gas / V_gas**2
        dT_int__m_gas = dT_int__P_gas * dP_gas__m_gas
        dT_int__T_gas = dT_int__P_gas * dP_gas__T_gas
        dT_int__V_gas = dT_int__P_gas * dP_gas__V_gas

        # Dump the pressure ones
        J["P_gas", "m_gas"] = dP_gas__m_gas
        J["P_gas", "T_gas"] = dP_gas__T_gas
        J["P_gas", "V_gas"] = dP_gas__V_gas

        # k_sat_gas_portion, which depends only on T_int
        dhtc__k_sat_gas = self.nusselt / L_int
        dk_sat_gas__T_int = self.H2.sat_gh2_k(self.T_int, deriv=True)

        # Nusselt number portion
        dhtc__Nu = self.k_sat_gas / L_int

        # Use mask to avoid NaNs caused by
        NaN_mask = np.logical_and(self.prandtl != 0.0, self.grashof != 0.0)
        dhtc__Pr = np.zeros_like(self.prandtl)
        dhtc__Gr = np.zeros_like(self.grashof)
        dhtc__Pr[NaN_mask] = (
            self.n_const
            * self.C_const
            * (self.prandtl[NaN_mask] * self.grashof[NaN_mask]) ** (self.n_const - 1)
            * self.grashof[NaN_mask]
            * dhtc__Nu[NaN_mask]
        )
        dhtc__Gr[NaN_mask] = (
            self.n_const
            * self.C_const
            * (self.prandtl[NaN_mask] * self.grashof[NaN_mask]) ** (self.n_const - 1)
            * self.prandtl[NaN_mask]
            * dhtc__Nu[NaN_mask]
        )

        dPr__T_int = (
            self.visc_sat_gas / self.k_sat_gas * self.H2.sat_gh2_cp(self.T_int, deriv=True)
            + self.cp_sat_gas / self.k_sat_gas * self.H2.sat_gh2_viscosity(self.T_int, deriv=True)
            - self.cp_sat_gas * self.visc_sat_gas / self.k_sat_gas**2 * self.H2.sat_gh2_k(self.T_int, deriv=True)
        )

        dGr__T_int = (  # p(Gr)/p(beta) * p(beta)/p(T_int)
            GRAV_CONST
            * self.rho_sat_gas**2
            * np.abs(T_gas - self.T_int)
            * L_int**3
            / self.visc_sat_gas**2
            * self.H2.sat_gh2_beta(self.T_int, deriv=True)
        )
        dGr__T_int += (  # p(Gr)/p(rho) * p(rho)/p(T_int)
            2
            * GRAV_CONST
            * self.beta_sat_gas
            * self.rho_sat_gas
            * np.abs(T_gas - self.T_int)
            * L_int**3
            / self.visc_sat_gas**2
            * self.H2.sat_gh2_rho(self.T_int, deriv=True)
        )
        dGr__T_int += (  # p(Gr)/p(viscosity) * p(viscosity)/p(T_int)
            -2
            * GRAV_CONST
            * self.beta_sat_gas
            * self.rho_sat_gas**2
            * np.abs(T_gas - self.T_int)
            * L_int**3
            / self.visc_sat_gas**3
            * self.H2.sat_gh2_viscosity(self.T_int, deriv=True)
        )
        abs_val_mult = np.ones(self.options["num_nodes"])  # derivative of abs(T_gas - T_int) w.r.t. T_gas
        abs_val_mult[(T_gas - self.T_int) < 0] = -1.0
        dGr__T_int += (  # p(Gr)/p(T_int)
            GRAV_CONST
            * self.beta_sat_gas
            * self.rho_sat_gas**2
            * (-abs_val_mult)
            * L_int**3
            / self.visc_sat_gas**2
        )

        dGr__T_gas = (  # p(Gr)/p(T_gas)
            GRAV_CONST * self.beta_sat_gas * self.rho_sat_gas**2 * abs_val_mult * L_int**3 / self.visc_sat_gas**2
        )
        dGr__L_int = (  # p(Gr)/p(L_int)
            3
            * GRAV_CONST
            * self.beta_sat_gas
            * self.rho_sat_gas**2
            * np.abs(T_gas - self.T_int)
            * L_int**2
            / self.visc_sat_gas**2
        )

        # Zero out anywhere the Prandtl or Grashof number is negative
        dPr__T_int[np.real(self.prandtl) < 0] *= 0.0
        dGr__T_int[np.real(self.grashof) < 0] *= 0.0
        dGr__T_gas[np.real(self.grashof) < 0] *= 0.0
        dGr__L_int[np.real(self.grashof) < 0] *= 0.0

        dhtc__T_int = dhtc__k_sat_gas * dk_sat_gas__T_int + dhtc__Pr * dPr__T_int + dhtc__Gr * dGr__T_int

        # Total derivatives of the heat transfer coefficient
        dhtc__m_gas = dhtc__T_int * dT_int__m_gas
        dhtc__T_gas = dhtc__T_int * dT_int__T_gas + dhtc__Gr * dGr__T_gas
        dhtc__V_gas = dhtc__T_int * dT_int__V_gas
        dhtc__L_int = dhtc__Gr * dGr__L_int - self.k_sat_gas / L_int**2 * self.nusselt

        # Get total derivatives of Q_gas_int w.r.t. inputs
        dQ_gas_int__htc = A_int * (T_gas - self.T_int)
        pQ_gas_int__T_int = -self.heat_transfer_coeff_gas_int * A_int  # just the partial derivative w.r.t. T_int
        dQ_gas_int__m_gas = dQ_gas_int__htc * dhtc__m_gas + pQ_gas_int__T_int * dT_int__m_gas
        dQ_gas_int__T_gas = (
            dQ_gas_int__htc * dhtc__T_gas + pQ_gas_int__T_int * dT_int__T_gas + self.heat_transfer_coeff_gas_int * A_int
        )
        dQ_gas_int__V_gas = dQ_gas_int__htc * dhtc__V_gas + pQ_gas_int__T_int * dT_int__V_gas
        dQ_gas_int__L_int = dQ_gas_int__htc * dhtc__L_int
        dQ_gas_int__A_int = self.heat_transfer_coeff_gas_int * (T_gas - self.T_int)

        # -------------- Finish off the m_dot_boil_off derivatives (excl bulk boiling and condensation) --------------
        dh_gas__P_gas, dh_gas__T_gas = self.H2.gh2_h(self.P_gas, T_gas, deriv=True)
        dmg__m_gas = dmg__Q_gas_int * dQ_gas_int__m_gas + dmg__h_gas * dh_gas__P_gas * dP_gas__m_gas
        dmg__T_gas = dmg__Q_gas_int * dQ_gas_int__T_gas + dmg__h_gas * (dh_gas__T_gas + dh_gas__P_gas * dP_gas__T_gas)
        dmg__V_gas = dmg__Q_gas_int * dQ_gas_int__V_gas + dmg__h_gas * dh_gas__P_gas * dP_gas__V_gas
        dmg__T_liq = dmg__h_liq * self.H2.lh2_h(T_liq, deriv=True)
        dmg__L_int = dmg__Q_gas_int * dQ_gas_int__L_int
        dmg__A_int = dmg__Q_gas_int * dQ_gas_int__A_int
        # dmg__Q_add already done

        # Dump em
        J["m_dot_gas", "m_gas"] = dmg__m_gas
        J["m_dot_gas", "T_gas"] = dmg__T_gas
        J["m_dot_gas", "V_gas"] = dmg__V_gas
        J["m_dot_gas", "T_liq"] = dmg__T_liq
        J["m_dot_gas", "Q_add"] = dmg__Q_add
        J["m_dot_gas", "L_interface"] = dmg__L_int
        J["m_dot_gas", "A_interface"] = dmg__A_int
        J["m_dot_gas", "m_dot_gas_out"] = -1.0

        # ==============================================================================
        # m_dot_liq (no bulk boiling or cloud condensation yet)
        # ==============================================================================
        J["m_dot_liq", "m_gas"] = -J["m_dot_gas", "m_gas"]
        J["m_dot_liq", "T_gas"] = -J["m_dot_gas", "T_gas"]
        J["m_dot_liq", "V_gas"] = -J["m_dot_gas", "V_gas"]
        J["m_dot_liq", "T_liq"] = -J["m_dot_gas", "T_liq"]
        J["m_dot_liq", "Q_add"] = -J["m_dot_gas", "Q_add"]
        J["m_dot_liq", "L_interface"] = -J["m_dot_gas", "L_interface"]
        J["m_dot_liq", "A_interface"] = -J["m_dot_gas", "A_interface"]
        J["m_dot_liq", "m_dot_liq_out"] = -1.0

        # ==============================================================================
        # V_dot_gas (no bulk boiling or cloud condensation yet)
        # ==============================================================================
        drho_liq__T_liq = self.H2.lh2_rho(T_liq, deriv=True)
        J["V_dot_gas", "m_gas"] = -J["m_dot_liq", "m_gas"] / self.rho_liq
        J["V_dot_gas", "T_gas"] = -J["m_dot_liq", "T_gas"] / self.rho_liq
        J["V_dot_gas", "V_gas"] = -J["m_dot_liq", "V_gas"] / self.rho_liq
        J["V_dot_gas", "T_liq"] = (
            -J["m_dot_liq", "T_liq"] / self.rho_liq + self.m_dot_liq / self.rho_liq**2 * drho_liq__T_liq
        )
        J["V_dot_gas", "Q_add"] = -J["m_dot_liq", "Q_add"] / self.rho_liq
        J["V_dot_gas", "L_interface"] = -J["m_dot_liq", "L_interface"] / self.rho_liq
        J["V_dot_gas", "A_interface"] = -J["m_dot_liq", "A_interface"] / self.rho_liq
        J["V_dot_gas", "m_dot_liq_out"] = -J["m_dot_liq", "m_dot_liq_out"] / self.rho_liq

        # ==============================================================================
        # T_dot_gas (no_bulk_boiling or cloud condensation yet)
        # ==============================================================================
        # Derivatives originating from partial w.r.t. Q_gas
        J["T_dot_gas", "Q_gas"] = 1 / (m_gas * self.cv_gas)

        # Derivatives originating from partial w.r.t. Q_gas_int
        J["T_dot_gas", "m_gas"] = -dQ_gas_int__m_gas / (m_gas * self.cv_gas)
        J["T_dot_gas", "T_gas"] = -dQ_gas_int__T_gas / (m_gas * self.cv_gas)
        J["T_dot_gas", "V_gas"] = -dQ_gas_int__V_gas / (m_gas * self.cv_gas)
        J["T_dot_gas", "L_interface"] = -dQ_gas_int__L_int / (m_gas * self.cv_gas)
        J["T_dot_gas", "A_interface"] = -dQ_gas_int__A_int / (m_gas * self.cv_gas)

        # Derivatives originating from partial w.r.t. P_gas
        dT_dot_gas__P_gas = -self.V_dot_gas / (m_gas * self.cv_gas)
        J["T_dot_gas", "m_gas"] += dT_dot_gas__P_gas * dP_gas__m_gas
        J["T_dot_gas", "T_gas"] += dT_dot_gas__P_gas * dP_gas__T_gas
        J["T_dot_gas", "V_gas"] += dT_dot_gas__P_gas * dP_gas__V_gas

        # Derivatives originating from partial w.r.t. V_dot_gas
        dT_dot_gas__V_dot_gas = -self.P_gas / (m_gas * self.cv_gas)
        J["T_dot_gas", "m_gas"] += dT_dot_gas__V_dot_gas * J["V_dot_gas", "m_gas"]
        J["T_dot_gas", "T_gas"] += dT_dot_gas__V_dot_gas * J["V_dot_gas", "T_gas"]
        J["T_dot_gas", "V_gas"] += dT_dot_gas__V_dot_gas * J["V_dot_gas", "V_gas"]
        J["T_dot_gas", "T_liq"] = dT_dot_gas__V_dot_gas * J["V_dot_gas", "T_liq"]
        J["T_dot_gas", "Q_add"] = dT_dot_gas__V_dot_gas * J["V_dot_gas", "Q_add"]
        J["T_dot_gas", "L_interface"] += dT_dot_gas__V_dot_gas * J["V_dot_gas", "L_interface"]
        J["T_dot_gas", "A_interface"] += dT_dot_gas__V_dot_gas * J["V_dot_gas", "A_interface"]
        J["T_dot_gas", "m_dot_liq_out"] = dT_dot_gas__V_dot_gas * J["V_dot_gas", "m_dot_liq_out"]

        # Derivatives originating from partial w.r.t. m_dot_gas
        dT_dot_gas__m_dot_gas = (self.h_gas - self.u_gas) / (m_gas * self.cv_gas)
        J["T_dot_gas", "m_gas"] += dT_dot_gas__m_dot_gas * J["m_dot_gas", "m_gas"]
        J["T_dot_gas", "T_gas"] += dT_dot_gas__m_dot_gas * J["m_dot_gas", "T_gas"]
        J["T_dot_gas", "V_gas"] += dT_dot_gas__m_dot_gas * J["m_dot_gas", "V_gas"]
        J["T_dot_gas", "T_liq"] += dT_dot_gas__m_dot_gas * J["m_dot_gas", "T_liq"]
        J["T_dot_gas", "Q_add"] += dT_dot_gas__m_dot_gas * J["m_dot_gas", "Q_add"]
        J["T_dot_gas", "L_interface"] += dT_dot_gas__m_dot_gas * J["m_dot_gas", "L_interface"]
        J["T_dot_gas", "A_interface"] += dT_dot_gas__m_dot_gas * J["m_dot_gas", "A_interface"]
        J["T_dot_gas", "m_dot_gas_out"] = dT_dot_gas__m_dot_gas * J["m_dot_gas", "m_dot_gas_out"]

        # Derivatives originating from partial w.r.t. h_gas
        dT_dot_gas__h_gas = self.m_dot_gas / (m_gas * self.cv_gas)
        J["T_dot_gas", "m_gas"] += dT_dot_gas__h_gas * (dh_gas__P_gas * dP_gas__m_gas)
        J["T_dot_gas", "T_gas"] += dT_dot_gas__h_gas * (dh_gas__P_gas * dP_gas__T_gas + dh_gas__T_gas)
        J["T_dot_gas", "V_gas"] += dT_dot_gas__h_gas * (dh_gas__P_gas * dP_gas__V_gas)

        # Derivatives originating from partial w.r.t. u_gas
        du_gas__P_gas, du_gas__T_gas = self.H2.gh2_u(self.P_gas, T_gas, deriv=True)
        dT_dot_gas__u_gas = -self.m_dot_gas / (m_gas * self.cv_gas)
        J["T_dot_gas", "m_gas"] += dT_dot_gas__u_gas * (du_gas__P_gas * dP_gas__m_gas)
        J["T_dot_gas", "T_gas"] += dT_dot_gas__u_gas * (du_gas__P_gas * dP_gas__T_gas + du_gas__T_gas)
        J["T_dot_gas", "V_gas"] += dT_dot_gas__u_gas * (du_gas__P_gas * dP_gas__V_gas)

        # Derivatives originating from partial w.r.t. m_gas
        J["T_dot_gas", "m_gas"] += -self.T_dot_gas / m_gas

        # Derivatives originating from partial w.r.t. cv_gas
        dcv_gas__P_gas, dcv_gas__T_gas = self.H2.gh2_cv(self.P_gas, T_gas, deriv=True)
        dT_dot_gas__cv_gas = -self.T_dot_gas / self.cv_gas
        J["T_dot_gas", "m_gas"] += dT_dot_gas__cv_gas * (dcv_gas__P_gas * dP_gas__m_gas)
        J["T_dot_gas", "T_gas"] += dT_dot_gas__cv_gas * (dcv_gas__P_gas * dP_gas__T_gas + dcv_gas__T_gas)
        J["T_dot_gas", "V_gas"] += dT_dot_gas__cv_gas * (dcv_gas__P_gas * dP_gas__V_gas)

        # ==============================================================================
        # T_dot_liq (no_bulk_boiling or cloud condensation yet)
        # ==============================================================================
        # Derivatives originating from partial w.r.t. Q_liq
        J["T_dot_liq", "Q_liq"] = 1 / (m_liq * self.cp_liq)

        # Derivatives originating from partial w.r.t. Q_add
        J["T_dot_liq", "Q_add"] = (1 - heater_boil_frac) / (m_liq * self.cp_liq)

        # Derivatives originating from partial w.r.t. P_gas
        dT_dot_liq__P_gas = -self.V_dot_liq / (m_liq * self.cp_liq)
        J["T_dot_liq", "m_gas"] = dT_dot_liq__P_gas * dP_gas__m_gas
        J["T_dot_liq", "T_gas"] = dT_dot_liq__P_gas * dP_gas__T_gas
        J["T_dot_liq", "V_gas"] = dT_dot_liq__P_gas * dP_gas__V_gas

        # Derivatives originating from partial w.r.t. V_dot_liq
        dT_dot_liq__V_dot_liq = -self.P_gas / (m_liq * self.cp_liq)
        J["T_dot_liq", "m_gas"] += dT_dot_liq__V_dot_liq * (-J["V_dot_gas", "m_gas"])
        J["T_dot_liq", "T_gas"] += dT_dot_liq__V_dot_liq * (-J["V_dot_gas", "T_gas"])
        J["T_dot_liq", "V_gas"] += dT_dot_liq__V_dot_liq * (-J["V_dot_gas", "V_gas"])
        J["T_dot_liq", "T_liq"] = dT_dot_liq__V_dot_liq * (-J["V_dot_gas", "T_liq"])
        J["T_dot_liq", "Q_add"] += dT_dot_liq__V_dot_liq * (-J["V_dot_gas", "Q_add"])
        J["T_dot_liq", "L_interface"] = dT_dot_liq__V_dot_liq * (-J["V_dot_gas", "L_interface"])
        J["T_dot_liq", "A_interface"] = dT_dot_liq__V_dot_liq * (-J["V_dot_gas", "A_interface"])
        J["T_dot_liq", "m_dot_liq_out"] = dT_dot_liq__V_dot_liq * (-J["V_dot_gas", "m_dot_liq_out"])

        # Derivatives originating from partial w.r.t. m_dot_liq
        dT_dot_liq__m_dot_liq = (self.h_liq - self.u_liq) / (m_liq * self.cp_liq)
        J["T_dot_liq", "m_gas"] += dT_dot_liq__m_dot_liq * J["m_dot_liq", "m_gas"]
        J["T_dot_liq", "T_gas"] += dT_dot_liq__m_dot_liq * J["m_dot_liq", "T_gas"]
        J["T_dot_liq", "V_gas"] += dT_dot_liq__m_dot_liq * J["m_dot_liq", "V_gas"]
        J["T_dot_liq", "T_liq"] += dT_dot_liq__m_dot_liq * J["m_dot_liq", "T_liq"]
        J["T_dot_liq", "Q_add"] += dT_dot_liq__m_dot_liq * J["m_dot_liq", "Q_add"]
        J["T_dot_liq", "L_interface"] += dT_dot_liq__m_dot_liq * J["m_dot_liq", "L_interface"]
        J["T_dot_liq", "A_interface"] += dT_dot_liq__m_dot_liq * J["m_dot_liq", "A_interface"]
        J["T_dot_liq", "m_dot_liq_out"] += dT_dot_liq__m_dot_liq * J["m_dot_liq", "m_dot_liq_out"]

        # Derivatives originating from partial w.r.t. h_liq
        dh_liq__T_liq = self.H2.lh2_h(T_liq, deriv=True)
        J["T_dot_liq", "T_liq"] += self.m_dot_liq / (m_liq * self.cp_liq) * dh_liq__T_liq

        # Derivatives originating from partial w.r.t. u_liq
        du_liq__T_liq = self.H2.lh2_u(T_liq, deriv=True)
        J["T_dot_liq", "T_liq"] += -self.m_dot_liq / (m_liq * self.cp_liq) * du_liq__T_liq

        # Derivatives originating from partial w.r.t. m_liq
        J["T_dot_liq", "m_liq"] += -self.T_dot_liq / m_liq

        # Derivatives originating from partial w.r.t. cp_liq
        dcp_liq__T_liq = self.H2.lh2_cp(T_liq, deriv=True)
        J["T_dot_liq", "T_liq"] += -self.T_dot_liq / self.cp_liq * dcp_liq__T_liq

        # ==============================================================================
        # Compute derivatives of bulk boiling and cloud condensation
        # ==============================================================================
        # -------------- Derivative of m_dot_bbcc w.r.t. inputs --------------
        # Derivatives of the additional mass flow from bulk boiling
        # and cloud condensation w.r.t. its components
        dm_dot_bbcc__m_dot_bb = self.bulk_boil_multiplier
        dm_dot_bbcc__bb_mult = self.m_dot_bulk_boil
        dm_dot_bbcc__m_dot_cc = -self.cloud_cond_multiplier
        dm_dot_bbcc__cc_mult = -self.m_dot_cloud_cond

        # Derivative of the sigmoids w.r.t. inputs
        dbb_mult__P_diff = -1 / (1 + np.exp(self.P_diff)) ** 2 * np.exp(self.P_diff)
        dbb_mult__P_liq = dbb_mult__P_diff * self.options["sigmoid_fac"] * (-self.P_gas / self.P_liq**2)
        dbb_mult__P_gas = dbb_mult__P_diff * self.options["sigmoid_fac"] / self.P_liq
        dbb_mult__P_liq[self.idx_P_overflow] = 0.0  # account for the exponentiation overflow protection
        dbb_mult__P_gas[self.idx_P_overflow] = 0.0  # account for the exponentiation overflow protection
        dbb_mult__m_gas = dbb_mult__P_gas * dP_gas__m_gas
        dbb_mult__T_gas = dbb_mult__P_gas * dP_gas__T_gas
        dbb_mult__V_gas = dbb_mult__P_gas * dP_gas__V_gas
        dbb_mult__T_liq = dbb_mult__P_liq * self.H2.lh2_P(T_liq, deriv=True)

        dcc_mult__T_diff = -1 / (1 + np.exp(self.T_diff)) ** 2 * np.exp(self.T_diff)
        dcc_mult__T_int = dcc_mult__T_diff * self.options["sigmoid_fac"] * (-T_gas / self.T_int**2)
        dcc_mult__T_gas = dcc_mult__T_diff * self.options["sigmoid_fac"] / self.T_int + dcc_mult__T_int * dT_int__T_gas
        dcc_mult__T_int[self.idx_T_overflow] = 0.0  # account for the exponentiation overflow protection
        dcc_mult__T_gas[self.idx_T_overflow] = 0.0  # account for the exponentiation overflow protection
        dcc_mult__m_gas = dcc_mult__T_int * dT_int__m_gas
        dcc_mult__V_gas = dcc_mult__T_int * dT_int__V_gas

        # Derivative of m_dot_bulk_boil w.r.t. inputs
        m_dot_bb_coeff = self.m_dot_bb_sign * MOLEC_WEIGHT_H2 * V_gas / (UNIVERSAL_GAS_CONST * T_gas)
        dm_dot_bb__T_dot_liq = m_dot_bb_coeff * self.d_P_liq_d_T
        dm_dot_bb__m_dot_gas = m_dot_bb_coeff * (-UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2) * T_gas / V_gas
        dm_dot_bb__T_dot_gas = m_dot_bb_coeff * (-UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2) * m_gas / V_gas
        dm_dot_bb__V_dot_gas = m_dot_bb_coeff * UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2 * m_gas * T_gas / V_gas**2

        dm_dot_bb__Q_add = (
            dm_dot_bb__T_dot_liq * J["T_dot_liq", "Q_add"]
            + dm_dot_bb__m_dot_gas * J["m_dot_gas", "Q_add"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "Q_add"]
            + dm_dot_bb__V_dot_gas * J["V_dot_gas", "Q_add"]
        )
        dm_dot_bb__A_interface = (
            dm_dot_bb__T_dot_liq * J["T_dot_liq", "A_interface"]
            + dm_dot_bb__m_dot_gas * J["m_dot_gas", "A_interface"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "A_interface"]
            + dm_dot_bb__V_dot_gas * J["V_dot_gas", "A_interface"]
        )
        dm_dot_bb__L_interface = (
            dm_dot_bb__T_dot_liq * J["T_dot_liq", "L_interface"]
            + dm_dot_bb__m_dot_gas * J["m_dot_gas", "L_interface"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "L_interface"]
            + dm_dot_bb__V_dot_gas * J["V_dot_gas", "L_interface"]
        )
        dm_dot_bb__m_gas = (
            dm_dot_bb__T_dot_liq * J["T_dot_liq", "m_gas"]
            + dm_dot_bb__m_dot_gas * J["m_dot_gas", "m_gas"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "m_gas"]
            + dm_dot_bb__V_dot_gas * J["V_dot_gas", "m_gas"]
            + m_dot_bb_coeff
            * (-UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2)
            * (self.T_dot_gas / V_gas - T_gas * self.V_dot_gas / V_gas**2)
        )
        dm_dot_bb__T_gas = (
            dm_dot_bb__T_dot_liq * J["T_dot_liq", "T_gas"]
            + dm_dot_bb__m_dot_gas * J["m_dot_gas", "T_gas"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "T_gas"]
            + dm_dot_bb__V_dot_gas * J["V_dot_gas", "T_gas"]
            - self.m_dot_bulk_boil / T_gas  # first product rule term
            + m_dot_bb_coeff
            * (-UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2)
            * (self.m_dot_gas / V_gas - m_gas * self.V_dot_gas / V_gas**2)  # second product rule term
        )
        dm_dot_bb__V_gas = (
            dm_dot_bb__T_dot_liq * J["T_dot_liq", "V_gas"]
            + dm_dot_bb__m_dot_gas * J["m_dot_gas", "V_gas"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "V_gas"]
            + dm_dot_bb__V_dot_gas * J["V_dot_gas", "V_gas"]
            + self.m_dot_bulk_boil / V_gas  # first product rule term
            + m_dot_bb_coeff
            * (-UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2)
            * (
                -self.m_dot_gas * T_gas / V_gas**2
                - m_gas * self.T_dot_gas / V_gas**2
                + 2 * m_gas * T_gas * self.V_dot_gas / V_gas**3
            )  # second product rule term
        )
        dm_dot_bb__T_liq = (
            dm_dot_bb__T_dot_liq * J["T_dot_liq", "T_liq"]
            + dm_dot_bb__m_dot_gas * J["m_dot_gas", "T_liq"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "T_liq"]
            + dm_dot_bb__V_dot_gas * J["V_dot_gas", "T_liq"]
            + m_dot_bb_coeff * self.T_dot_liq * self.H2.lh2_P(T_liq, deriv=2)
        )
        dm_dot_bb__m_liq = dm_dot_bb__T_dot_liq * J["T_dot_liq", "m_liq"]
        dm_dot_bb__Q_liq = dm_dot_bb__T_dot_liq * J["T_dot_liq", "Q_liq"]
        dm_dot_bb__Q_gas = dm_dot_bb__T_dot_gas * J["T_dot_gas", "Q_gas"]
        dm_dot_bb__m_dot_gas_out = (
            dm_dot_bb__m_dot_gas * J["m_dot_gas", "m_dot_gas_out"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "m_dot_gas_out"]
        )
        dm_dot_bb__m_dot_liq_out = (
            dm_dot_bb__V_dot_gas * J["V_dot_gas", "m_dot_liq_out"]
            + dm_dot_bb__T_dot_liq * J["T_dot_liq", "m_dot_liq_out"]
            + dm_dot_bb__T_dot_gas * J["T_dot_gas", "m_dot_liq_out"]
        )

        # Derivative of m_dot_cloud_cond w.r.t. inputs
        m_dot_cc_coeff = self.m_dot_cc_sign * self.P_gas * V_gas * MOLEC_WEIGHT_H2 / (UNIVERSAL_GAS_CONST * T_gas**2)
        #                  first product rule term
        dm_dot_cc__P_gas = self.m_dot_cloud_cond / self.P_gas + m_dot_cc_coeff * (
            self.H2.sat_gh2_T(self.P_gas, deriv=2)
            * UNIVERSAL_GAS_CONST
            / MOLEC_WEIGHT_H2
            * (
                self.m_dot_gas * T_gas / V_gas
                + m_gas * self.T_dot_gas / V_gas
                - m_gas * T_gas * self.V_dot_gas / V_gas**2
            )
        )  # second product rule term
        dm_dot_cc__m_dot_gas = (
            m_dot_cc_coeff * self.d_T_sat_g_d_P * UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2 * T_gas / V_gas
        )
        dm_dot_cc__T_dot_gas = m_dot_cc_coeff * (
            self.d_T_sat_g_d_P * UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2 * m_gas / V_gas - 1
        )
        dm_dot_cc__V_dot_gas = (
            m_dot_cc_coeff * self.d_T_sat_g_d_P * UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_H2 * (-m_gas * T_gas / V_gas**2)
        )

        dm_dot_cc__Q_add = (
            dm_dot_cc__m_dot_gas * J["m_dot_gas", "Q_add"]
            + dm_dot_cc__V_dot_gas * J["V_dot_gas", "Q_add"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "Q_add"]
        )
        dm_dot_cc__A_interface = (
            dm_dot_cc__m_dot_gas * J["m_dot_gas", "A_interface"]
            + dm_dot_cc__V_dot_gas * J["V_dot_gas", "A_interface"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "A_interface"]
        )
        dm_dot_cc__L_interface = (
            dm_dot_cc__m_dot_gas * J["m_dot_gas", "L_interface"]
            + dm_dot_cc__V_dot_gas * J["V_dot_gas", "L_interface"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "L_interface"]
        )
        dm_dot_cc__T_gas = (
            dm_dot_cc__m_dot_gas * J["m_dot_gas", "T_gas"]
            + dm_dot_cc__V_dot_gas * J["V_dot_gas", "T_gas"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "T_gas"]
            + dm_dot_cc__P_gas * J["P_gas", "T_gas"]
            - 2 * self.m_dot_cloud_cond / T_gas  # first product rule term
            + m_dot_cc_coeff
            * self.d_T_sat_g_d_P
            * UNIVERSAL_GAS_CONST
            / MOLEC_WEIGHT_H2
            * (self.m_dot_gas / V_gas - m_gas * self.V_dot_gas / V_gas**2)  # second product rule term
        )
        dm_dot_cc__T_liq = (
            dm_dot_cc__m_dot_gas * J["m_dot_gas", "T_liq"]
            + dm_dot_cc__V_dot_gas * J["V_dot_gas", "T_liq"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "T_liq"]
        )
        dm_dot_cc__V_gas = (
            dm_dot_cc__m_dot_gas * J["m_dot_gas", "V_gas"]
            + dm_dot_cc__V_dot_gas * J["V_dot_gas", "V_gas"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "V_gas"]
            + dm_dot_cc__P_gas * J["P_gas", "V_gas"]
            + self.m_dot_cloud_cond / V_gas  # first product rule term
            + m_dot_cc_coeff
            * self.d_T_sat_g_d_P
            * UNIVERSAL_GAS_CONST
            / MOLEC_WEIGHT_H2
            * (
                -self.m_dot_gas * T_gas / V_gas**2
                - m_gas * self.T_dot_gas / V_gas**2
                + 2 * m_gas * T_gas * self.V_dot_gas / V_gas**3
            )  # second product rule term
        )
        dm_dot_cc__m_dot_gas_out = (
            dm_dot_cc__m_dot_gas * J["m_dot_gas", "m_dot_gas_out"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "m_dot_gas_out"]
        )
        dm_dot_cc__m_dot_liq_out = (
            dm_dot_cc__V_dot_gas * J["V_dot_gas", "m_dot_liq_out"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "m_dot_liq_out"]
        )
        dm_dot_cc__m_gas = (
            dm_dot_cc__m_dot_gas * J["m_dot_gas", "m_gas"]
            + dm_dot_cc__V_dot_gas * J["V_dot_gas", "m_gas"]
            + dm_dot_cc__T_dot_gas * J["T_dot_gas", "m_gas"]
            + dm_dot_cc__P_gas * J["P_gas", "m_gas"]
            + m_dot_cc_coeff
            * self.d_T_sat_g_d_P
            * UNIVERSAL_GAS_CONST
            / MOLEC_WEIGHT_H2
            * (self.T_dot_gas / V_gas - T_gas * self.V_dot_gas / V_gas**2)
        )
        dm_dot_cc__Q_gas = dm_dot_cc__T_dot_gas * J["T_dot_gas", "Q_gas"]

        # ==============================================================================
        # Add contribution of bulk boiling and cloud condensation to final ODE outputs
        # ==============================================================================
        # -------------- m_dot_liq, m_dot_gas, and V_dot_gas are similar --------------
        for key in ["m_dot_liq", "m_dot_gas", "V_dot_gas"]:
            if key == "m_dot_liq":
                scaler = -1.0
            elif key == "m_dot_gas":
                scaler = 1.0
            elif key == "V_dot_gas":
                scaler = 1 / self.rho_liq
            J[key, "Q_add"] += scaler * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__Q_add + dm_dot_bbcc__m_dot_cc * dm_dot_cc__Q_add
            )
            J[key, "A_interface"] += scaler * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__A_interface + dm_dot_bbcc__m_dot_cc * dm_dot_cc__A_interface
            )
            J[key, "L_interface"] += scaler * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__L_interface + dm_dot_bbcc__m_dot_cc * dm_dot_cc__L_interface
            )
            J[key, "m_gas"] += scaler * (
                dm_dot_bbcc__bb_mult * dbb_mult__m_gas
                + dm_dot_bbcc__cc_mult * dcc_mult__m_gas
                + dm_dot_bbcc__m_dot_bb * dm_dot_bb__m_gas
                + dm_dot_bbcc__m_dot_cc * dm_dot_cc__m_gas
            )
            J[key, "T_gas"] += scaler * (
                dm_dot_bbcc__bb_mult * dbb_mult__T_gas
                + dm_dot_bbcc__cc_mult * dcc_mult__T_gas
                + dm_dot_bbcc__m_dot_bb * dm_dot_bb__T_gas
                + dm_dot_bbcc__m_dot_cc * dm_dot_cc__T_gas
            )
            J[key, "V_gas"] += scaler * (
                dm_dot_bbcc__bb_mult * dbb_mult__V_gas
                + dm_dot_bbcc__cc_mult * dcc_mult__V_gas
                + dm_dot_bbcc__m_dot_bb * dm_dot_bb__V_gas
                + dm_dot_bbcc__m_dot_cc * dm_dot_cc__V_gas
            )
            J[key, "T_liq"] += scaler * (
                dm_dot_bbcc__bb_mult * dbb_mult__T_liq
                + dm_dot_bbcc__m_dot_bb * dm_dot_bb__T_liq
                + dm_dot_bbcc__m_dot_cc * dm_dot_cc__T_liq
            )
            J[key, "m_liq"] += scaler * dm_dot_bbcc__m_dot_bb * dm_dot_bb__m_liq
            J[key, "Q_liq"] += scaler * dm_dot_bbcc__m_dot_bb * dm_dot_bb__Q_liq
            J[key, "Q_gas"] += scaler * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__Q_gas + dm_dot_bbcc__m_dot_cc * dm_dot_cc__Q_gas
            )
            J[key, "m_dot_gas_out"] += scaler * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__m_dot_gas_out + dm_dot_bbcc__m_dot_cc * dm_dot_cc__m_dot_gas_out
            )
            J[key, "m_dot_liq_out"] += scaler * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__m_dot_liq_out + dm_dot_bbcc__m_dot_cc * dm_dot_cc__m_dot_liq_out
            )

        # V_dot_gas addition also is affected by T_liq influence on rho_liq
        J["V_dot_gas", "T_liq"] += -self.m_dot_bbcc / self.rho_liq**2 * drho_liq__T_liq

        # -------------- T_dot_liq and T_dot_gas are similar --------------
        for key in ["T_dot_liq", "T_dot_gas"]:
            if key == "T_dot_liq":
                # Partial of (contribution of added mass flow to T_dot_liq directly,
                # excluding Q_liq_bulk_boil) w.r.t. m_dot_bbcc
                coeff = (self.P_gas / self.rho_liq - (self.h_liq - self.u_liq)) / (m_liq * self.cp_liq)

                # Partial of (temp change from heat added to liquid from bulk boiling) w.r.t. m_dot_bulk_boil
                coeff_dQ__m_dot = -(self.h_gas - self.h_liq) * self.bulk_boil_multiplier / (m_liq * self.cp_liq)

                # Partial of (temp change from heat added to liquid from bulk boiling) w.r.t. bulk_boil_multiplier
                coeff_dQ__mult = -(self.h_gas - self.h_liq) * self.m_dot_bulk_boil / (m_liq * self.cp_liq)
            elif key == "T_dot_gas":
                # Partial of (contribution of added mass flow to T_dot_gas directly,
                # excluding Q_gas_cloud_cond) w.r.t. m_dot_bbcc
                coeff = (-self.P_gas / self.rho_liq + (self.h_gas - self.u_gas)) / (m_gas * self.cv_gas)

                # Partial of (temp change from heat added to gas from cloud condensation) w.r.t. m_dot_cloud_cond
                coeff_dQ__m_dot = (self.h_gas - self.h_liq) * self.cloud_cond_multiplier / (m_gas * self.cv_gas)

                # Partial of (temp change from heat added to gas from cloud condensation) w.r.t. cloud_cond_multiplier
                coeff_dQ__mult = (self.h_gas - self.h_liq) * self.m_dot_cloud_cond / (m_gas * self.cv_gas)

            J[key, "Q_add"] += coeff * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__Q_add + dm_dot_bbcc__m_dot_cc * dm_dot_cc__Q_add
            )
            J[key, "Q_add"] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__Q_add if key == "T_dot_liq" else dm_dot_cc__Q_add
            )
            J[key, "A_interface"] += coeff * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__A_interface + dm_dot_bbcc__m_dot_cc * dm_dot_cc__A_interface
            )
            J[
                key, "A_interface"
            ] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__A_interface if key == "T_dot_liq" else dm_dot_cc__A_interface
            )
            J[key, "L_interface"] += coeff * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__L_interface + dm_dot_bbcc__m_dot_cc * dm_dot_cc__L_interface
            )
            J[
                key, "L_interface"
            ] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__L_interface if key == "T_dot_liq" else dm_dot_cc__L_interface
            )
            J[key, "m_gas"] += coeff * (
                dm_dot_bbcc__bb_mult * dbb_mult__m_gas
                + dm_dot_bbcc__cc_mult * dcc_mult__m_gas
                + dm_dot_bbcc__m_dot_bb * dm_dot_bb__m_gas
                + dm_dot_bbcc__m_dot_cc * dm_dot_cc__m_gas
            )
            J[key, "m_gas"] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__m_gas if key == "T_dot_liq" else dm_dot_cc__m_gas
            ) + coeff_dQ__mult * (dbb_mult__m_gas if key == "T_dot_liq" else dcc_mult__m_gas)
            J[key, "T_gas"] += coeff * (
                dm_dot_bbcc__bb_mult * dbb_mult__T_gas
                + dm_dot_bbcc__cc_mult * dcc_mult__T_gas
                + dm_dot_bbcc__m_dot_bb * dm_dot_bb__T_gas
                + dm_dot_bbcc__m_dot_cc * dm_dot_cc__T_gas
            )
            J[key, "T_gas"] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__T_gas if key == "T_dot_liq" else dm_dot_cc__T_gas
            ) + coeff_dQ__mult * (dbb_mult__T_gas if key == "T_dot_liq" else dcc_mult__T_gas)
            J[key, "V_gas"] += coeff * (
                dm_dot_bbcc__bb_mult * dbb_mult__V_gas
                + dm_dot_bbcc__cc_mult * dcc_mult__V_gas
                + dm_dot_bbcc__m_dot_bb * dm_dot_bb__V_gas
                + dm_dot_bbcc__m_dot_cc * dm_dot_cc__V_gas
            )
            J[key, "V_gas"] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__V_gas if key == "T_dot_liq" else dm_dot_cc__V_gas
            ) + coeff_dQ__mult * (dbb_mult__V_gas if key == "T_dot_liq" else dcc_mult__V_gas)
            J[key, "T_liq"] += coeff * (
                dm_dot_bbcc__bb_mult * dbb_mult__T_liq
                + dm_dot_bbcc__m_dot_bb * dm_dot_bb__T_liq
                + dm_dot_bbcc__m_dot_cc * dm_dot_cc__T_liq
            )
            J[key, "T_liq"] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__T_liq if key == "T_dot_liq" else dm_dot_cc__T_liq
            ) + coeff_dQ__mult * (dbb_mult__T_liq if key == "T_dot_liq" else 0.0)
            J[key, "m_liq"] += coeff * dm_dot_bbcc__m_dot_bb * dm_dot_bb__m_liq
            J[key, "m_liq"] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__m_liq if key == "T_dot_liq" else 0.0
            )
            J[key, "Q_liq"] += coeff * dm_dot_bbcc__m_dot_bb * dm_dot_bb__Q_liq
            J[key, "Q_liq"] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__Q_liq if key == "T_dot_liq" else 0.0
            )
            J[key, "Q_gas"] += coeff * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__Q_gas + dm_dot_bbcc__m_dot_cc * dm_dot_cc__Q_gas
            )
            J[key, "Q_gas"] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__Q_gas if key == "T_dot_liq" else dm_dot_cc__Q_gas
            )
            J[key, "m_dot_gas_out"] += coeff * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__m_dot_gas_out + dm_dot_bbcc__m_dot_cc * dm_dot_cc__m_dot_gas_out
            )
            J[
                key, "m_dot_gas_out"
            ] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__m_dot_gas_out if key == "T_dot_liq" else dm_dot_cc__m_dot_gas_out
            )
            J[key, "m_dot_liq_out"] += coeff * (
                dm_dot_bbcc__m_dot_bb * dm_dot_bb__m_dot_liq_out + dm_dot_bbcc__m_dot_cc * dm_dot_cc__m_dot_liq_out
            )
            J[
                key, "m_dot_liq_out"
            ] += coeff_dQ__m_dot * (  # influence on heat flows from bulk boiling or cloud condensation
                dm_dot_bb__m_dot_liq_out if key == "T_dot_liq" else dm_dot_cc__m_dot_liq_out
            )

        # Include the influence from the other terms in Q_liq_bulk_boil and Q_gas_cloud_cond
        dT_dot_liq_bb__h_gas = -self.m_dot_bulk_boil * self.bulk_boil_multiplier / (m_liq * self.cp_liq)
        J["T_dot_liq", "m_gas"] += dT_dot_liq_bb__h_gas * dh_gas__P_gas * J["P_gas", "m_gas"]
        J["T_dot_liq", "V_gas"] += dT_dot_liq_bb__h_gas * dh_gas__P_gas * J["P_gas", "V_gas"]
        J["T_dot_liq", "T_gas"] += dT_dot_liq_bb__h_gas * (dh_gas__T_gas + dh_gas__P_gas * J["P_gas", "T_gas"])
        J["T_dot_liq", "T_liq"] += (
            self.m_dot_bbcc
            * (-self.P_gas / self.rho_liq**2 * drho_liq__T_liq - (dh_liq__T_liq - du_liq__T_liq))
            / (m_liq * self.cp_liq)
            - self.m_dot_bbcc
            * (self.P_gas / self.rho_liq - (self.h_liq - self.u_liq))
            / (m_liq * self.cp_liq) ** 2
            * m_liq
            * dcp_liq__T_liq
            + dh_liq__T_liq * self.m_dot_bulk_boil * self.bulk_boil_multiplier / (m_liq * self.cp_liq)
            - self.Q_liq_bulk_boil / (m_liq * self.cp_liq) ** 2 * m_liq * dcp_liq__T_liq
        )
        J["T_dot_liq", "m_liq"] += -(
            self.Q_liq_bulk_boil
            + self.P_gas * self.m_dot_bbcc / self.rho_liq
            - self.m_dot_bbcc * (self.h_liq - self.u_liq)
        ) / (m_liq**2 * self.cp_liq)

        # Influence from P_gas in the PdV term of the liquid temperature change caused by the
        # added mass flow from bulk boiling and cloud condensation
        for wrt in ["m_gas", "T_gas", "V_gas"]:
            J["T_dot_liq", wrt] += J["P_gas", wrt] * self.m_dot_bbcc / self.rho_liq / (m_liq * self.cp_liq)

        # The black formatter totally butchers this section, so turn it off temporarily
        # fmt: off
        J["T_dot_gas", "m_gas"] += (
            self.m_dot_bbcc * J["P_gas", "m_gas"] * (-1 / self.rho_liq + dh_gas__P_gas - du_gas__P_gas) / (m_gas * self.cv_gas)
            - self.m_dot_bbcc * (-self.P_gas / self.rho_liq + (self.h_gas - self.u_gas)) / (m_gas * self.cv_gas)**2
            * (self.cv_gas + m_gas * dcv_gas__P_gas * J["P_gas", "m_gas"])
            + dh_gas__P_gas * J["P_gas", "m_gas"] * self.m_dot_cloud_cond * self.cloud_cond_multiplier / (m_gas * self.cv_gas)
            - self.Q_gas_cloud_cond / (m_gas * self.cv_gas)**2 * (self.cv_gas + m_gas * dcv_gas__P_gas * J["P_gas", "m_gas"])
        )
        J["T_dot_gas", "V_gas"] += (
            self.m_dot_bbcc * J["P_gas", "V_gas"] * (-1 / self.rho_liq + dh_gas__P_gas - du_gas__P_gas) / (m_gas * self.cv_gas)
            - self.m_dot_bbcc * (-self.P_gas / self.rho_liq + (self.h_gas - self.u_gas)) / (m_gas * self.cv_gas)**2
            * m_gas * dcv_gas__P_gas * J["P_gas", "V_gas"]
            + dh_gas__P_gas * J["P_gas", "V_gas"] * self.m_dot_cloud_cond * self.cloud_cond_multiplier / (m_gas * self.cv_gas)
            - self.Q_gas_cloud_cond / (m_gas * self.cv_gas)**2 * m_gas * dcv_gas__P_gas * J["P_gas", "V_gas"]
        )
        J["T_dot_gas", "T_gas"] += (
            self.m_dot_bbcc * (J["P_gas", "T_gas"] * (-1 / self.rho_liq + dh_gas__P_gas - du_gas__P_gas) + dh_gas__T_gas - du_gas__T_gas) / (m_gas * self.cv_gas)
            - self.m_dot_bbcc * (-self.P_gas / self.rho_liq + (self.h_gas - self.u_gas)) / (m_gas * self.cv_gas)**2
            * m_gas * (dcv_gas__P_gas * J["P_gas", "T_gas"] + dcv_gas__T_gas)
            + (dh_gas__P_gas * J["P_gas", "T_gas"] + dh_gas__T_gas) * self.m_dot_cloud_cond * self.cloud_cond_multiplier / (m_gas * self.cv_gas)
            - self.Q_gas_cloud_cond / (m_gas * self.cv_gas)**2 * m_gas * (dcv_gas__P_gas * J["P_gas", "T_gas"] + dcv_gas__T_gas)
        )
        J["T_dot_gas", "T_liq"] += (
            self.P_gas * self.m_dot_bbcc / self.rho_liq**2 * drho_liq__T_liq
            - dh_liq__T_liq * self.m_dot_cloud_cond * self.cloud_cond_multiplier
        ) / (m_gas * self.cv_gas)
        # fmt: on


class BoilOffFillLevelCalc(om.ExplicitComponent):
    """
    Computes the fill level in the tank given the
    volume of the gas.

    Inputs
    ------
    radius : float
        Inner radius of the cylinder. This value does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLINDRICAL part of the tank (scalar, m)
    V_gas : float
        Volume of the ullage (vector, m^3)

    Outputs
    -------
    fill_level : float
        Volume fraction of tank (in range 0-1) filled with liquid propellant (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    end_cap_depth_ratio : float
        End cap depth divided by cylinder radius. 1 gives hemisphere, 0.5 gives 2:1 semi ellipsoid.
        Must be in the range 0-1 (inclusive). By default 1.0.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare(
            "end_cap_depth_ratio", lower=0.0, upper=1.0, default=1.0, desc="End cap depth / cylinder radius"
        )

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("radius", val=2.0, units="m")
        self.add_input("length", val=0.5, units="m")
        self.add_input("V_gas", units="m**3", shape=(nn,))
        self.add_output("fill_level", val=0.5, shape=(nn,), lower=0.001, upper=0.999)

        arng = np.arange(nn)
        self.declare_partials("fill_level", ["V_gas"], rows=arng, cols=arng)
        self.declare_partials("fill_level", ["radius", "length"], rows=arng, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        r = inputs["radius"]
        L = inputs["length"]

        V_tank = 4 / 3 * np.pi * r**3 * self.options["end_cap_depth_ratio"] + np.pi * r**2 * L
        outputs["fill_level"] = 1 - inputs["V_gas"] / V_tank

    def compute_partials(self, inputs, J):
        r = inputs["radius"]
        L = inputs["length"]

        V_tank = 4 / 3 * np.pi * r**3 * self.options["end_cap_depth_ratio"] + np.pi * r**2 * L
        J["fill_level", "V_gas"] = -1 / V_tank
        J["fill_level", "radius"] = (
            inputs["V_gas"]
            / V_tank**2
            * (4 * np.pi * r**2 * self.options["end_cap_depth_ratio"] + 2 * np.pi * r * L)
        )
        J["fill_level", "length"] = inputs["V_gas"] / V_tank**2 * (np.pi * r**2)


class InitialTankStateModification(om.ExplicitComponent):
    """
    Inputs
    ------
    radius : float
        Inner radius of the cylinder. This value does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    delta_m_gas : float
        Change in mass of the gaseous hydrogen in the tank ullage since the beginning of the mission (vector, kg)
    delta_m_liq : float
        Change in mass of liquid hydrogen in the tank since the beginning of the mission (vector, kg)
    delta_T_gas : float
        Change in temperature of the gaseous hydrogen in the ullage since the beginning of the mission (vector, K)
    delta_T_liq : float
        Change in temperature of the bulk liquid hydrogen since the beginning of the mission (vector, K)
    delta_V_gas : float
        Change in volume of the ullage since the beginning of the mission (vector, m^3)

    Outputs
    -------
    m_gas : float
        Mass of the gaseous hydrogen in the tank ullage (vector, kg)
    m_liq : float
        Mass of liquid hydrogen in the tank (vector, kg)
    T_gas : float
        Temperature of the gaseous hydrogen in the ullage (vector, K)
    T_liq : float
        Temperature of the bulk liquid hydrogen (vector, K)
    V_gas : float
        Volume of the ullage (vector, m^3)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    fill_level_init : float
        Initial fill level (in range 0-1) of the tank, default 0.95
        to leave space for boil-off gas; 5% adopted from Millis et al. 2009 (scalar, dimensionless)
    ullage_T_init : float
        Initial temperature of gas in ullage, default 21 K (scalar, K)
    ullage_P_init : float
        Initial pressure of gas in ullage, default 150,000 Pa; ullage pressure must be higher than ambient
        to prevent air leaking in and creating a combustible mixture (scalar, Pa)
    liquid_T_init : float
        Initial temperature of bulk liquid hydrogen, default 20 K (scalar, K)
    end_cap_depth_ratio : float
        End cap depth divided by cylinder radius. 1 gives hemisphere, 0.5 gives 2:1 semi ellipsoid.
        Must be in the range 0-1 (inclusive). By default 1.0.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("fill_level_init", default=0.95, desc="Initial fill level")
        self.options.declare("ullage_T_init", default=21.0, desc="Initial ullage temp (K)")
        self.options.declare("ullage_P_init", default=1.5e5, desc="Initial ullage pressure (Pa)")
        self.options.declare("liquid_T_init", default=20.0, desc="Initial bulk liquid temp (K)")
        self.options.declare(
            "end_cap_depth_ratio", lower=0.0, upper=1.0, default=1.0, desc="End cap depth / cylinder radius"
        )

    def setup(self):
        nn = self.options["num_nodes"]

        r_default = 1.0
        L_default = 0.5
        self.add_input("radius", val=r_default, units="m")
        self.add_input("length", val=L_default, units="m")

        self.add_input("delta_m_gas", shape=(nn,), units="kg", val=0.0)
        self.add_input("delta_m_liq", shape=(nn,), units="kg", val=0.0)
        self.add_input("delta_T_gas", shape=(nn,), units="K", val=0.0)
        self.add_input("delta_T_liq", shape=(nn,), units="K", val=0.0)
        self.add_input("delta_V_gas", shape=(nn,), units="m**3", val=0.0)

        # Get reasonable default values for states
        defaults = self._compute_initial_states(r_default, L_default)
        self.add_output("m_gas", shape=(nn,), units="kg", lower=1e-6, val=defaults["m_gas_init"], upper=1e4)
        self.add_output("m_liq", shape=(nn,), units="kg", lower=1e-2, val=defaults["m_liq_init"], upper=1e6)
        self.add_output("T_gas", shape=(nn,), units="K", lower=18, val=defaults["T_gas_init"], upper=150)
        self.add_output("T_liq", shape=(nn,), units="K", lower=14, val=defaults["T_liq_init"], upper=33)
        self.add_output("V_gas", shape=(nn,), units="m**3", lower=1e-5, val=defaults["V_gas_init"], upper=1e4)

        arng = np.arange(nn)
        self.declare_partials("V_gas", "delta_V_gas", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials("m_liq", "delta_m_liq", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials("m_gas", "delta_m_gas", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials("T_gas", "delta_T_gas", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials("T_liq", "delta_T_liq", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials(["V_gas", "m_gas", "m_liq"], ["radius", "length"], rows=arng, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        init_states = self._compute_initial_states(inputs["radius"], inputs["length"])

        outputs["V_gas"] = inputs["delta_V_gas"] + init_states["V_gas_init"]
        outputs["m_gas"] = inputs["delta_m_gas"] + init_states["m_gas_init"]
        outputs["m_liq"] = inputs["delta_m_liq"] + init_states["m_liq_init"]
        outputs["T_gas"] = inputs["delta_T_gas"] + init_states["T_gas_init"]
        outputs["T_liq"] = inputs["delta_T_liq"] + init_states["T_liq_init"]

    def compute_partials(self, inputs, partials):
        r = inputs["radius"]
        L = inputs["length"]
        fill_init = self.options["fill_level_init"]
        T_gas_init = self.options["ullage_T_init"]
        T_liq_init = self.options["liquid_T_init"]
        P_init = self.options["ullage_P_init"]

        # Partial derivatives of tank geometry w.r.t. radius and length
        Vtank_r = 4 * np.pi * r**2 * self.options["end_cap_depth_ratio"] + 2 * np.pi * r * L
        Vtank_L = np.pi * r**2

        partials["V_gas", "radius"] = Vtank_r * (1 - fill_init)
        partials["V_gas", "length"] = Vtank_L * (1 - fill_init)

        coeff = H2_prop.gh2_rho(P_init, T_gas_init).item()
        partials["m_gas", "radius"] = coeff * partials["V_gas", "radius"]
        partials["m_gas", "length"] = coeff * partials["V_gas", "length"]

        partials["m_liq", "radius"] = (Vtank_r - partials["V_gas", "radius"]) * H2_prop.lh2_rho(T_liq_init)
        partials["m_liq", "length"] = (Vtank_L - partials["V_gas", "length"]) * H2_prop.lh2_rho(T_liq_init)

    def _compute_initial_states(self, radius, length):
        """
        Returns a dictionary with inital state values based on the specified options and tank geometry.
        """
        fill_init = self.options["fill_level_init"]
        T_gas_init = self.options["ullage_T_init"]
        T_liq_init = self.options["liquid_T_init"]
        P_init = self.options["ullage_P_init"]

        V_tank = 4 / 3 * np.pi * radius**3 * self.options["end_cap_depth_ratio"] + np.pi * radius**2 * length

        res = {}
        res["T_liq_init"] = T_liq_init
        res["T_gas_init"] = T_gas_init
        res["V_gas_init"] = V_tank * (1 - fill_init)
        res["m_gas_init"] = H2_prop.gh2_rho(P_init, T_gas_init).item() * res["V_gas_init"]
        res["m_liq_init"] = (V_tank - res["V_gas_init"]) * H2_prop.lh2_rho(T_liq_init)

        return res
