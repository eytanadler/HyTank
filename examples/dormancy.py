"""
@File    :   dormancy.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : A simple dormancy example run script. This script runs the model and then
               generates two files. The first is an OpenMDAO N2 diagram to view the model
               and results and is called dormancy_n2.html. The second is a plot of the
               results and is called dormancy.pdf.
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
fill_level_init = 0.6  # initial fill level
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

# Use a Newton solver to converge the heat leak and boil-off coupling loop
p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, atol=1e-8, rtol=1e-8, iprint=2)
p.model.linear_solver = om.DirectSolver()

# Finally, set up the problem
p.setup()

# -------------- Specify inputs --------------
# Duration of the test
duration = 12  # hrs
p.set_val("thermals.boil_off.integ.duration", duration, units="h")

# Tank geometry
p.set_val("radius", 0.5, units="m")
p.set_val("length", 1.0, units="m")

# Heater power (off for dormancy tests)
p.set_val("P_heater", np.zeros(nn), units="W")

# Mass flows out of the tank (zero for dormancy tests)
p.set_val("m_dot_gas_out", np.zeros(nn), units="kg/s")
p.set_val("m_dot_liq_out", np.zeros(nn), units="kg/s")

# Ambient temperature
p.set_val("T_env", 300, units="K")

# Tank structural and insulation design parameters
p.set_val("N_layers", 20)  # number of MLI layers
p.set_val("environment_design_pressure", 1.5, units="bar")
p.set_val("max_expected_operating_pressure", 10, units="bar")
p.set_val("vacuum_gap", 5, units="cm")

# -------------- Run the solver --------------
p.run_model()

# -------------- Create an OpenMDAO N2 diagram to interactively view the model and results --------------
# See https://openmdao.org/newdocs/versions/latest/features/model_visualization/n2_details/n2_details.html
# for more details about the diagram.
om.n2(p, outfile="dormancy_n2.html")

# -------------- Plot the results --------------
fig, axs = plt.subplots(2, 2)

# Time during the test (temporal nodes are evenly spaced)
t = np.linspace(0, duration, nn)

# Pressure
axs[0, 0].plot(t, p.get_val("P", units="bar"))
axs[0, 0].set_ylabel("Pressure (bar)")

# Temperature
axs[0, 1].plot(t, p.get_val("T_gas", units="K"), label="Ullage")
axs[0, 1].plot(t, p.get_val("T_liq", units="K"), label="Liquid")
axs[0, 1].legend(fontsize=8)
axs[0, 1].set_ylabel("Temperature (K)")

# Heat leak
axs[1, 0].plot(t, p.get_val("thermals.heat_leak.Q_gas", units="W"), label="Into ullage")
axs[1, 0].plot(t, p.get_val("thermals.heat_leak.Q_liq", units="W"), label="Into liquid")
axs[1, 0].legend(fontsize=8)
axs[1, 0].set_ylabel("Heat leak (W)")
axs[1, 0].set_xlabel("Time (hr)")

# Fill level
axs[1, 1].plot(t, p.get_val("fill_level") * 100)
axs[1, 1].set_ylabel("Fill level (%)")
axs[1, 1].set_xlabel("Time (hr)")

# Fix the alignment of the subplots
fig.tight_layout()

# Save the figure then show it
fig.savefig("dormancy.pdf")
plt.show()
