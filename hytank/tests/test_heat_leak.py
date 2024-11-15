"""
@File    :   test_heat_leak.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : Test the code in heat_leak.py
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

# ==============================================================================
# Extension modules
# ==============================================================================
from hytank.heat_leak import *


class HeatTransferVacuumTankTestCase(unittest.TestCase):
    def test_simple(self):
        """
        Regression test with some reasonable values that also checks the partials.
        """
        p = om.Problem()
        p.model.add_subsystem("model", HeatTransferVacuumTank(heat_multiplier=1.2), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("T_env", 300, units="K")
        p.set_val("N_layers", 10)
        p.set_val("T_liq", 20, units="K")
        p.set_val("T_gas", 25, units="K")
        p.set_val("A_wet", 10, units="m**2")
        p.set_val("A_dry", 5, units="m**2")

        p.run_model()

        assert_near_equal(p.get_val("Q_gas", units="W"), 18.33976445, tolerance=1e-8)
        assert_near_equal(p.get_val("Q_liq", units="W"), 36.7628524, tolerance=1e-8)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-13, rtol=1e-13)

    def test_vector(self):
        """
        Regression test with some reasonable vector values that also checks the partials.
        """
        nn = 3

        p = om.Problem()
        p.model.add_subsystem("model", HeatTransferVacuumTank(num_nodes=nn, heat_multiplier=1.2), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("T_env", np.linspace(200, 300, nn), units="K")
        p.set_val("N_layers", 10)
        p.set_val("T_liq", np.linspace(19, 21, nn), units="K")
        p.set_val("T_gas", np.linspace(25, 30, nn), units="K")
        p.set_val("A_wet", np.linspace(10, 20, nn), units="m**2")
        p.set_val("A_dry", np.linspace(20, 15, nn), units="m**2")

        p.run_model()

        assert_near_equal(
            p.get_val("Q_gas", units="W"), np.array([28.16293302, 40.98797564, 54.87130595]), tolerance=1e-8
        )
        assert_near_equal(
            p.get_val("Q_liq", units="W"), np.array([14.17964567, 35.32862698, 73.49479674]), tolerance=1e-8
        )

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-13, rtol=1e-13)

    def test_areas(self):
        """
        Make areas zero and check that no heat.
        """
        nn = 3
        p = om.Problem()
        p.model.add_subsystem("model", HeatTransferVacuumTank(num_nodes=nn, heat_multiplier=1.4), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("T_env", 300, units="K")
        p.set_val("N_layers", 10)
        p.set_val("T_liq", 21, units="K")
        p.set_val("T_gas", 21, units="K")
        p.set_val("A_wet", np.array([1, 0, 0.5]), units="m**2")
        p.set_val("A_dry", np.array([0, 1, 0.5]), units="m**2")

        p.run_model()

        assert_near_equal(p.get_val("Q_liq", units="W"), np.array([4.28719648, 0.0, 4.28719648 / 2]), tolerance=1e-8)
        assert_near_equal(p.get_val("Q_gas", units="W"), np.array([0.0, 4.28719648, 4.28719648 / 2]), tolerance=1e-8)


class MLIHeatFluxTestCase(unittest.TestCase):
    def test_simple(self):
        """
        Regression test with some reasonable values that also checks the partials.
        """
        p = om.Problem()
        p.model.add_subsystem("model", MLIHeatFlux(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("T_hot", 300, units="K")
        p.set_val("T_cold", 20, units="K")
        p.set_val("N_layers", 20)

        p.run_model()

        assert_near_equal(p.get_val("heat_flux", units="W/m**2"), 1.53178552, tolerance=1e-8)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-13, rtol=1e-13)

    def test_no_temp_diff(self):
        """
        No temperature difference should have zero heat flux.
        """
        p = om.Problem()
        p.model.add_subsystem("model", MLIHeatFlux(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("T_hot", 20, units="K")
        p.set_val("T_cold", 20, units="K")
        p.set_val("N_layers", 20)

        p.run_model()

        assert_near_equal(p.get_val("heat_flux", units="W/m**2"), 0.0, tolerance=1e-8)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-13, rtol=1e-13)

    def test_different_options(self):
        """
        Regression test with some reasonable values that also checks the partials.
        """
        nn = 3
        p = om.Problem()
        p.model.add_subsystem(
            "model",
            MLIHeatFlux(
                num_nodes=nn,
                layer_density=40,
                solid_cond_coeff=10e-8,
                gas_cond_coeff=3e4,
                rad_coeff=2e-10,
                emittance=0.05,
                vacuum_pressure=1e-3,
            ),
            promotes=["*"],
        )

        p.setup(force_alloc_complex=True)

        p.set_val("T_hot", np.linspace(200, 300, nn), units="K")
        p.set_val("T_cold", np.linspace(18, 30, nn), units="K")
        p.set_val("N_layers", 10)

        p.run_model()

        assert_near_equal(
            p.get_val("heat_flux", units="W/m**2"), np.array([36.24413141, 41.37825988, 46.64701375]), tolerance=1e-8
        )

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-13, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
