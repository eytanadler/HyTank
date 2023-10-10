"""
@File    :   extraction.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : A simple extraction example run script. This script runs the model and then
               generates two files. The first is an OpenMDAO N2 diagram to view the model
               and results and is called extraction_n2.html. The second is a plot of the
               results and is called extraction.pdf.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt

# ==============================================================================
# Extension modules
# ==============================================================================
from hytank import LH2Tank


# -------------- Set up the OpenMDAO Problem --------------
# Options for the LH2 tank model
nn = 51  # number time steps in the simulation, must be an odd number
fill_level_init = 0.95  # initial fill level
ullage_T_init = 26.1  # K, initial ullage temperature
liquid_T_init = 26  # K, initial liquid temperature
ullage_P_init = 4e5  # Pa, initial tank pressure

# Initialize the OpenMDAO Problem and add the model
p = om.Problem()
p.model = LH2Tank(
    num_nodes=nn,
    fill_level_init=fill_level_init,
    ullage_T_init=ullage_T_init,
    liquid_T_init=liquid_T_init,
    ullage_P_init=ullage_P_init,
)

# Some of the options deeper in the model can be set with the model_options.
# These will override the values of any options within components in the model.
# In this example, we decrease the heater's response time from the default.
p.model_options["*"] = {"heater_rate_const": 2e-3}

# Use a Newton solver to converge the heat leak and boil-off coupling loop
p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, atol=1e-8, rtol=1e-8, iprint=2)
p.model.linear_solver = om.DirectSolver()

# Finally, set up the problem
p.setup()

# -------------- Specify inputs --------------
# Duration of the test
duration = 2.0  # hrs
t = np.linspace(0, duration, nn)  # hrs, time during the test (temporal nodes are evenly spaced)
p.set_val("thermals.boil_off.integ.duration", duration, units="h")

# Tank geometry
p.set_val("radius", 0.5, units="m")
p.set_val("length", 1.0, units="m")

# Heater power (off for dormancy tests)
P_h = np.zeros(nn)
P_h[nn // 2 :] = np.linspace(10e3, 0e3, nn // 2 + 1)  # W, turn on the heater half way into the test
p.set_val("P_heater", P_h, units="W")

# Mass flows out of the tank (zero for dormancy tests)
m_dot_liq = np.zeros(nn)
m_dot_liq[: nn // 2] = np.linspace(1, 0, nn // 2)  # kg/min, pull liquid from the tank in the first half
m_dot_gas = np.zeros(nn)
m_dot_gas[nn // 2 :] = 0.5  # kg/min, pull gas from the tank in the second half
p.set_val("m_dot_gas_out", m_dot_gas, units="kg/min")
p.set_val("m_dot_liq_out", m_dot_liq, units="kg/min")

# Ambient temperature
p.set_val("T_env", 300, units="K")

# Tank structural and insulation design parameters
p.set_val("N_layers", 20)  # number of MLI layers
p.set_val("environment_design_pressure", 1.5, units="bar")
p.set_val("max_expected_operating_pressure", 10, units="bar")
p.set_val("vacuum_gap", 5, units="cm")

# -------------- Run the solver --------------
p.run_model()

# -------------- Print some weight results --------------
W_tank = p.get_val("tank_weight", units="kg").item()  # kg, tank weight (without hydrogen)
m_fuel_init = p.get_val("m_liq", units="kg")[0] + p.get_val("m_gas", units="kg")[0]
print(f"\nTank weight: {W_tank} kg")
print(f"Gravimetric efficiency (95% initial fill): {m_fuel_init / (m_fuel_init + W_tank) * 100:.1f}%")
print(f"Final fill level: {p.get_val('fill_level')[-1] * 100}%")

# -------------- Create an OpenMDAO N2 diagram to interactively view the model and results --------------
# See https://openmdao.org/newdocs/versions/latest/features/model_visualization/n2_details/n2_details.html
# for more details about the diagram.
# om.n2(p, outfile="extraction_n2.html")

# -------------- Plot the results --------------
fig, axs = plt.subplots(2, 3, figsize=(10, 5))

# Mass flow
axs[0, 0].plot(t, p.get_val("m_dot_gas_out", units="kg/min"), label="Gas")
axs[0, 0].plot(t, p.get_val("m_dot_liq_out", units="kg/min"), label="Liquid")
axs[0, 0].legend(fontsize=8)
axs[0, 0].set_ylabel("Mass flow out (kg/min)")

# Pressure
axs[0, 1].plot(t, p.get_val("P", units="bar"))
axs[0, 1].set_ylabel("Pressure (bar)")

# Temperature
axs[0, 2].plot(t, p.get_val("T_gas", units="K"), label="Ullage")
axs[0, 2].plot(t, p.get_val("T_liq", units="K"), label="Liquid")
axs[0, 2].legend(fontsize=8)
axs[0, 2].set_ylabel("Temperature (K)")

# Heater
axs[1, 0].plot(t, p.get_val("P_heater", units="kW"), label="Power\nsupplied")
axs[1, 0].plot(t, p.get_val("thermals.boil_off.integ.Q_add", units="kW"), label="Heat added\nto liquid")
axs[1, 0].legend(fontsize=8)
axs[1, 0].set_ylabel("Heater (kW)")

# Fill level
axs[1, 1].plot(t, p.get_val("fill_level") * 100)
axs[1, 1].set_ylabel("Fill level (%)")
axs[1, 1].set_xlabel("Time (hr)")
axs[1, 1].set_ylim((0, 100))

# Mass
axs[1, 2].plot(t, p.get_val("m_gas", units="kg"), label="Ullage")
axs[1, 2].plot(t, p.get_val("m_liq", units="kg"), label="Liquid")
axs[1, 2].legend(fontsize=8)
axs[1, 2].set_ylabel("Mass (kg)")
axs[1, 2].set_xlabel("Time (hr)")

# Fix the alignment of the subplots
fig.tight_layout()

# Save the figure then show it
fig.savefig("extraction.pdf")
# plt.show()
