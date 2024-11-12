"""
@File    :   test_LH2_tank.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : Test the code in LH2_tank.py
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

# ==============================================================================
# Extension modules
# ==============================================================================
from hytank.LH2_tank import *


class LH2TankTestCase(unittest.TestCase):
    def test_simple(self):
        """
        Test that this component runs and the outputs haven't changed.
        """
        nn = 5
        p = om.Problem()
        p.model = LH2Tank(
            ullage_P_init=101325.0,
            fill_level_init=0.95,
            ullage_T_init=25,
            num_nodes=nn,
            weight_fudge_factor=1.1,
            inner_safety_factor=1.5,
            heat_multiplier=1.2,
        )
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver = om.NewtonSolver()
        p.model.nonlinear_solver.options["err_on_non_converge"] = True
        p.model.nonlinear_solver.options["solve_subsystems"] = True
        p.model.nonlinear_solver.options["maxiter"] = 20
        p.model_options["*"] = {
            "heater_boil_frac": 0.75,
            "heat_transfer_C_gas_const": 0.27,
            "heat_transfer_n_gas_const": 0.25,
            "heat_transfer_C_liq_const": 0.27,
            "heat_transfer_n_liq_const": 0.25,
        }
        p.setup(force_alloc_complex=True)

        # Make the test extremely short so all the values are nearly the same in time
        p.set_val("thermals.boil_off.integ.duration", 1e-10, units="s")

        p.run_model()

        assert_near_equal(p.get_val("m_gas", units="kg"), np.full(nn, 0.29929491), tolerance=1e-7)
        assert_near_equal(p.get_val("m_liq", units="kg"), np.full(nn, 389.9321982), tolerance=1e-9)
        assert_near_equal(p.get_val("T_gas", units="K"), np.full(nn, 25), tolerance=1e-9)
        assert_near_equal(p.get_val("T_liq", units="K"), np.full(nn, 20), tolerance=1e-9)
        assert_near_equal(p.get_val("P", units="Pa"), np.full(nn, 101324.73830745), tolerance=1e-9)
        assert_near_equal(p.get_val("fill_level"), np.full(nn, 0.95), tolerance=1e-9)
        assert_near_equal(p.get_val("tank_weight", units="kg"), 252.7094369, tolerance=1e-9)
        assert_near_equal(p.get_val("total_weight", units="kg"), np.full(nn, 642.94093001), tolerance=1e-9)
        assert_near_equal(
            p.get_val("total_weight", units="kg"),
            p.get_val("tank_weight", units="kg") + p.get_val("m_gas", units="kg") + p.get_val("m_liq", units="kg"),
            tolerance=1e-9,
        )

    def test_time_history(self):
        duration = 15.0  # hr
        nn = 11

        p = om.Problem()
        p.model.add_subsystem(
            "tank",
            LH2Tank(
                num_nodes=nn,
                fill_level_init=0.95,
                ullage_P_init=1.5e5,
                ullage_T_init=22,
                liquid_T_init=20,
                weight_fudge_factor=1.1,
                inner_safety_factor=1.5,
                heat_multiplier=1.2,
            ),
            promotes=["*"],
        )

        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver = om.NewtonSolver()
        p.model.nonlinear_solver.options["err_on_non_converge"] = True
        p.model.nonlinear_solver.options["solve_subsystems"] = True
        p.model.nonlinear_solver.options["maxiter"] = 30
        p.model_options["*"] = {
            "heater_boil_frac": 0.75,
            "heat_transfer_C_gas_const": 0.27,
            "heat_transfer_n_gas_const": 0.25,
            "heat_transfer_C_liq_const": 0.27,
            "heat_transfer_n_liq_const": 0.25,
        }

        p.setup()

        p.set_val("thermals.boil_off.integ.duration", duration, units="h")
        p.set_val("radius", 2.75, units="m")
        p.set_val("length", 2.0, units="m")
        p.set_val("P_heater", np.linspace(1e3, 0.0, nn), units="W")
        p.set_val("m_dot_gas_out", -1.0, units="kg/h")
        p.set_val("m_dot_liq_out", 100.0, units="kg/h")
        p.set_val("T_env", 300, units="K")
        p.set_val("N_layers", 10)
        p.set_val("environment_design_pressure", 1, units="atm")
        p.set_val("max_expected_operating_pressure", 3, units="bar")
        p.set_val("vacuum_gap", 0.1, units="m")

        p.run_model()

        assert_near_equal(
            p.get_val("m_gas", units="kg"),
            np.array(
                [
                    12.626614671973162,
                    13.645756202973764,
                    16.412732935906579,
                    19.398491825769295,
                    22.243954951640990,
                    25.027404822100806,
                    27.776183790925224,
                    30.461885804147276,
                    33.059301102728476,
                    35.549448004560823,
                    37.910469097349257,
                ]
            ),
            tolerance=1e-7,
        )
        assert_near_equal(
            p.get_val("m_liq", units="kg"),
            np.array(
                [
                    9114.665133030850484,
                    8965.145991499850425,
                    8813.879014766916953,
                    8662.393255877053889,
                    8511.047792751183806,
                    8359.764342880724143,
                    8208.515563911898425,
                    8057.329861898676427,
                    7906.232446600096409,
                    7755.242299698265015,
                    7604.381278605475927,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_gas", units="K"),
            np.array(
                [
                    22.000000000000000,
                    22.054185887408948,
                    22.211835683915218,
                    22.200215589151572,
                    22.162333938206832,
                    22.165111332449229,
                    22.173423203750112,
                    22.166059894991982,
                    22.148595399325970,
                    22.127251925206551,
                    22.101537245853681,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_liq", units="K"),
            np.array(
                [
                    20.000000000000000,
                    20.080402765622225,
                    20.157359033586083,
                    20.229669774596264,
                    20.296526064955749,
                    20.357837104052312,
                    20.413676159551656,
                    20.463980634932756,
                    20.508630643862144,
                    20.547542142230633,
                    20.580662536204336,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("P", units="Pa"),
            np.array(
                [
                    149998.911153794615529,
                    126733.092428597068647,
                    124263.436056974474923,
                    123008.864079792547273,
                    121239.820643711020239,
                    119815.289032972534187,
                    118595.627578856423497,
                    117309.430684170190943,
                    115906.629578903273796,
                    114386.065546193218324,
                    112711.280710494116647,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("fill_level"),
            np.array(
                [
                    0.950000000000000,
                    0.934406391919294,
                    0.918610536034825,
                    0.902773178604111,
                    0.886932964450786,
                    0.871082949580451,
                    0.855221600889942,
                    0.839353262039495,
                    0.823481976267377,
                    0.807611172332032,
                    0.791744616749511,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("total_weight", units="kg"),
            np.array(
                [
                    15198.085899388050166,
                    15049.585899388050166,
                    14901.085899388050166,
                    14752.585899388050166,
                    14604.085899388050166,
                    14455.585899388050166,
                    14307.085899388050166,
                    14158.585899388050166,
                    14010.085899388051985,
                    13861.585899388053804,
                    13713.085899388053804,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(p.get_val("tank_weight", units="kg"), 6070.79415169, tolerance=1e-9)


class LH2TankThermalsTestCase(unittest.TestCase):
    def test_end_caps(self):
        """
        Test that this component runs and the outputs haven't changed.
        """
        nn = 5
        p = om.Problem()
        p.model = LH2TankThermals(
            ullage_P_init=4e5,
            fill_level_init=0.95,
            ullage_T_init=27,
            num_nodes=nn,
            heater_Q_add_init=5e2,
            end_cap_depth_ratio=0.5,
        )
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver = om.NewtonSolver()
        p.model.nonlinear_solver.options["err_on_non_converge"] = True
        p.model.nonlinear_solver.options["solve_subsystems"] = True
        p.model.nonlinear_solver.options["maxiter"] = 20
        p.setup(force_alloc_complex=True)

        # Make the test extremely short so all the values are nearly the same in time
        p.set_val("boil_off.integ.duration", 1e-10, units="s")

        p.run_model()

        assert_near_equal(p.get_val("m_gas", units="kg"), np.full(nn, 0.8280365990195481), tolerance=1e-7)
        assert_near_equal(p.get_val("m_liq", units="kg"), np.full(nn, 248.13867158470006), tolerance=1e-9)
        assert_near_equal(p.get_val("T_gas", units="K"), np.full(nn, 27), tolerance=1e-9)
        assert_near_equal(p.get_val("T_liq", units="K"), np.full(nn, 20), tolerance=1e-9)
        assert_near_equal(p.get_val("P", units="Pa"), np.full(nn, 399999.71776269376), tolerance=1e-9)
        assert_near_equal(p.get_val("fill_level"), np.full(nn, 0.95), tolerance=1e-9)


if __name__ == "__main__":
    unittest.main()
