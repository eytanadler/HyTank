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
        p.model = LH2Tank(ullage_P_init=101325.0, fill_level_init=0.95, ullage_T_init=25, num_nodes=nn)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver = om.NewtonSolver()
        p.model.nonlinear_solver.options["err_on_non_converge"] = True
        p.model.nonlinear_solver.options["solve_subsystems"] = True
        p.model.nonlinear_solver.options["maxiter"] = 20
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
        assert_near_equal(p.get_val("tank_weight", units="kg"), 252.70942027, tolerance=1e-9)
        assert_near_equal(p.get_val("total_weight", units="kg"), np.full(nn, 642.94091338), tolerance=1e-9)
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
            LH2Tank(num_nodes=nn, fill_level_init=0.95, ullage_P_init=1.5e5, ullage_T_init=22, liquid_T_init=20),
            promotes=["*"],
        )

        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver = om.NewtonSolver()
        p.model.nonlinear_solver.options["err_on_non_converge"] = True
        p.model.nonlinear_solver.options["solve_subsystems"] = True
        p.model.nonlinear_solver.options["maxiter"] = 30

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
                    12.626614671973114,
                    21.891877911165153,
                    32.08242036637532,
                    41.663797513409,
                    50.20290489826887,
                    57.80709082218871,
                    64.57808094124867,
                    70.52227318003337,
                    75.61753654110781,
                    79.85047431832257,
                    83.21685266003419,
                ]
            ),
            tolerance=1e-7,
        )
        assert_near_equal(
            p.get_val("m_liq", units="kg"),
            np.array(
                [
                    9114.665133030858,
                    8956.899869791665,
                    8798.209327336455,
                    8640.127950189422,
                    8483.088842804562,
                    8326.984656880642,
                    8171.713666761581,
                    8017.269474522797,
                    7863.674211161722,
                    7710.941273384508,
                    7559.074895042797,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_gas", units="K"),
            np.array(
                [
                    22.0,
                    24.316539503582735,
                    25.028430049377587,
                    25.132392012907903,
                    25.141809941143737,
                    25.15256491176942,
                    25.125503324568808,
                    25.042993290797167,
                    24.918353991182087,
                    24.767372952128568,
                    24.59742094285993,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_liq", units="K"),
            np.array(
                [
                    20.0,
                    20.03967962785759,
                    20.081629828299302,
                    20.12316866546681,
                    20.163356780443426,
                    20.20208303286758,
                    20.239264118045902,
                    20.27475881016566,
                    20.308446112193817,
                    20.34023262843681,
                    20.370034063664082,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("P", units="Pa"),
            np.array(
                [
                    149998.9111537946,
                    212868.2714251496,
                    252364.4898060021,
                    270927.96002220636,
                    278591.13270184526,
                    280851.05445443915,
                    279308.64754925953,
                    274699.82564601005,
                    267788.7247343958,
                    259180.6030446576,
                    249374.38501624248,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("fill_level"),
            np.array(
                [
                    0.95,
                    0.9335518292782577,
                    0.916996105660649,
                    0.9004931157620089,
                    0.8840883626968652,
                    0.8677710391200572,
                    0.8515309852443136,
                    0.8353680370355739,
                    0.8192850374533575,
                    0.8032839366817282,
                    0.7873657453496401,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("total_weight", units="kg"),
            np.array(
                [
                    15198.085890617343,
                    15049.585890617343,
                    14901.085890617345,
                    14752.585890617343,
                    14604.085890617343,
                    14455.585890617342,
                    14307.085890617343,
                    14158.585890617343,
                    14010.085890617342,
                    13861.585890617343,
                    13713.085890617343,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(p.get_val("tank_weight", units="kg"), 6070.794142914512, tolerance=1e-9)


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
