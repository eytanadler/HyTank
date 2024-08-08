"""
@File    :   test_boil_off.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : Test the code in boil_off.py
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
import hytank.H2_properties_MendezRamos as H2_prop_MendezRamos
from hytank.H2_properties import HydrogenProperties
from hytank.boil_off import *


H2_prop = HydrogenProperties()


class BoilOffTestCase(unittest.TestCase):
    def test_integrated(self):
        """
        A regression test for the fully integrated boil-off model.
        """
        nn = 11
        p = om.Problem()
        p.model.add_subsystem(
            "model",
            BoilOff(num_nodes=nn, fill_level_init=0.9, ullage_T_init=21, ullage_P_init=1.2e5, liquid_T_init=20),
            promotes=["*"],
        )

        p.setup(force_alloc_complex=True)

        p.set_val("integ.duration", 3, units="h")
        p.set_val("radius", 0.7, units="m")
        p.set_val("length", 0.3, units="m")
        p.set_val("m_dot_gas_out", 0.2, units="kg/h")
        p.set_val("m_dot_liq_out", 30.0, units="kg/h")

        A_wet = np.array(
            [5.95723, 5.48700, 5.08361, 4.71719, 4.37258, 4.04051, 3.71445, 3.38888, 3.05836, 2.71649, 2.35443]
        )
        A_dry = np.array(
            [1.51976, 1.98999, 2.39338, 2.75980, 3.10441, 3.43648, 3.76254, 4.08811, 4.41863, 4.76050, 5.12256]
        )
        Q_tot = 50
        p.set_val("Q_liq", Q_tot * A_wet / (A_wet + A_dry), units="W")
        p.set_val("Q_gas", Q_tot * A_dry / (A_wet + A_dry), units="W")

        p.set_val("P_heater", 1, units="kW")

        p.run_model()

        assert_near_equal(
            p.get_val("m_gas", units="kg"),
            np.array(
                [
                    0.293247793577024,
                    0.714516001744989,
                    1.359469349861478,
                    2.111699994691247,
                    2.931699133561161,
                    3.840692837957930,
                    4.877714143446416,
                    6.076741690468928,
                    7.464475297083723,
                    9.064721886596439,
                    10.899298034694326,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("m_liq", units="kg"),
            np.array(
                [
                    121.770788097670689,
                    112.289519889502714,
                    102.584566541386224,
                    92.772335896556470,
                    82.892336757686550,
                    72.923343053289784,
                    62.826321747801309,
                    52.567294200778768,
                    42.119560594163985,
                    31.459314004651254,
                    20.564737856553368,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_gas", units="K"),
            np.array(
                [
                    21.000000000000000,
                    24.928776569433140,
                    26.437966947004398,
                    26.901711031779129,
                    27.186328734990362,
                    27.582276165197001,
                    28.086252025539011,
                    28.635216255982595,
                    29.197172923343686,
                    29.777006447983744,
                    30.410287726172399,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_liq", units="K"),
            np.array(
                [
                    20.000000000000000,
                    20.250392707965091,
                    20.793847903836113,
                    21.507169313999391,
                    22.297781132424635,
                    23.117274925183285,
                    23.950129277603061,
                    24.796843964277858,
                    25.664906990939667,
                    26.571066688775932,
                    27.560786226930457,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("P_gas", units="Pa"),
            np.array(
                [
                    120025.050247269944521,
                    201147.688795425376156,
                    276885.289365169941448,
                    327015.838725500856526,
                    363436.853224313759711,
                    397715.682287567877211,
                    434429.972553370287642,
                    474225.378872489614878,
                    516868.241007197066210,
                    562553.452367093414068,
                    612179.439688944141380,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("fill_level"),
            np.array(
                [
                    0.900000000000000,
                    0.829860540711408,
                    0.757546454164775,
                    0.683660833950111,
                    0.608274642532347,
                    0.531043021942039,
                    0.451495726161860,
                    0.369174837686314,
                    0.283618990779241,
                    0.194284636951486,
                    0.100429938370160,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("integ.Q_add", units="kW"),
            np.array(
                [
                    0.000000000000000,
                    0.623550192494425,
                    0.877159767933210,
                    0.952913277479860,
                    0.971687727015300,
                    0.981034274875462,
                    0.989659837504596,
                    0.995877806363511,
                    0.998909069036199,
                    0.999815941669861,
                    0.999900511525099,
                ]
            ),
            tolerance=1e-9,
        )


class LiquidHeightTestCase(unittest.TestCase):
    def setUp(self):
        self.nn = nn = 7
        self.p = p = om.Problem()
        p.model.add_subsystem("model", LiquidHeight(num_nodes=nn), promotes=["*"])
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, iprint=2, err_on_non_converge=True, maxiter=20
        )
        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=True)

    def test_simple(self):
        r = 0.6
        L = 0.3

        # Define height and work backwards to fill level so we
        # can recompute it and check against the original height
        off = 5.0  # deg
        theta = np.linspace(off, 2 * np.pi - off, self.nn)
        h = r * (1 - np.cos(theta / 2))
        V_fill = r**2 / 2 * (theta - np.sin(theta)) * L + np.pi * h**2 / 3 * (3 * r - h)
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        fill = V_fill / V_tank

        self.p.set_val("fill_level", fill)
        self.p.set_val("radius", r, units="m")
        self.p.set_val("length", L, units="m")

        self.p.run_model()

        assert_near_equal(self.p.get_val("h_liq_frac"), h / (2 * r), tolerance=1e-8)

    def test_flat_end_caps(self):
        self.p = p = om.Problem()
        p.model.add_subsystem("model", LiquidHeight(num_nodes=1, end_cap_depth_ratio=0.0), promotes=["*"])
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, iprint=2, err_on_non_converge=True, maxiter=20
        )
        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=True)

        r = 1.4
        L = 0.4
        h = 3 * r / 2
        th = 2 * np.arccos(1 - h / r)
        V = r**2 / 2 * (th - np.sin(th)) * L

        self.p.set_val("fill_level", V / (np.pi * r**2 * L))
        self.p.set_val("radius", r, units="m")
        self.p.set_val("length", L, units="m")

        self.p.run_model()

        assert_near_equal(self.p.get_val("h_liq_frac"), h / (2 * r), tolerance=1e-8)

    def test_ellipsoidal_end_caps(self):
        """
        Ellipsoidal volumes computed using an online calculator to check
        (https://keisan.casio.com/exec/system/1311572253)
        """
        self.p = p = om.Problem()
        p.model.add_subsystem("model", LiquidHeight(num_nodes=1, end_cap_depth_ratio=0.5), promotes=["*"])
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, iprint=2, err_on_non_converge=True, maxiter=20
        )
        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=True)

        r = 1.5
        L = 1.0
        h = 4 * r / 3
        th = 2 * np.arccos(1 - h / r)
        V = r**2 / 2 * (th - np.sin(th)) * L + 5.23598775598298873

        self.p.set_val("fill_level", V / (np.pi * r**2 * L + 7.06858347057703479))
        self.p.set_val("radius", r, units="m")
        self.p.set_val("length", L, units="m")

        self.p.run_model()

        assert_near_equal(self.p.get_val("h_liq_frac"), h / (2 * r), tolerance=1e-8)

    def test_derivatives(self):
        self.p.set_val("fill_level", np.linspace(0.1, 0.9, self.nn))
        self.p.set_val("radius", 0.6, units="m")
        self.p.set_val("length", 1.2, units="m")

        self.p.run_model()

        partials = self.p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)

    def test_derivatives_ellipsoidal(self):
        self.nn = nn = 7
        self.p = p = om.Problem()
        p.model.add_subsystem("model", LiquidHeight(num_nodes=nn, end_cap_depth_ratio=0.4), promotes=["*"])
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, iprint=2, err_on_non_converge=True, maxiter=20
        )
        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=True)

        self.p.set_val("fill_level", np.linspace(0.1, 0.9, self.nn))
        self.p.set_val("radius", 0.6, units="m")
        self.p.set_val("length", 1.2, units="m")

        self.p.run_model()

        partials = self.p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)


class BoilOffGeometryTestCase(unittest.TestCase):
    def setup_model(self, nn, end_cap_depth_ratio=1.0):
        self.p = p = om.Problem()
        comp = p.model.add_subsystem(
            "model",
            BoilOffGeometry(num_nodes=nn, end_cap_depth_ratio=end_cap_depth_ratio),
            promotes=["*"],
        )
        p.setup(force_alloc_complex=True)
        comp.adjust_h_liq_frac = False

        self.r = 0.7
        self.L = 0.3

        self.p.set_val("radius", self.r, units="m")
        self.p.set_val("length", self.L, units="m")

    def test_empty(self):
        self.setup_model(1)

        self.p.set_val("h_liq_frac", 0)

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), 0.0, tolerance=5e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), A_tank, tolerance=5e-8)

    def test_full(self):
        self.setup_model(1)

        self.p.set_val("h_liq_frac", 1.0)

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), A_tank, tolerance=5e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), 0.0, tolerance=5e-8)

    def test_half(self):
        self.setup_model(1)

        self.p.set_val("h_liq_frac", 0.5)

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(
            self.p.get_val("A_interface", units="m**2").item(),
            np.pi * self.r**2 + 2 * self.r * self.L,
            tolerance=1e-8,
        )
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 2 * self.r, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), A_tank / 2, tolerance=5e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), A_tank / 2, tolerance=5e-8)

    def test_regression(self):
        nn = 5
        self.setup_model(nn)

        self.p.set_val("h_liq_frac", np.linspace(0, 1.0, nn))

        self.p.run_model()

        A_wet = np.array([0.0, 1.97920337, 3.73849526, 5.49778714, 7.47699052])
        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(
            self.p.get_val("A_interface", units="m**2"),
            np.array([0.0, 1.5182659697837129, 1.9593804002589983, 1.5182659697837138, 0.0]),
            tolerance=1e-8,
        )
        assert_near_equal(
            self.p.get_val("L_interface", units="m"),
            np.array([0.0, 1.212435565298214, 1.4, 1.2124355652982144, 0.0]),
            tolerance=1e-8,
        )
        assert_near_equal(self.p.get_val("A_wet", units="m**2"), A_wet, tolerance=5e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2"), A_tank - A_wet, tolerance=5e-8)

    def test_ellipsoidal(self):
        self.setup_model(1, end_cap_depth_ratio=0.7)

        self.p.set_val("h_liq_frac", 1.0)

        self.p.run_model()

        A_tank = 4.97064806830777411 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), A_tank, tolerance=5e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), 0.0, tolerance=5e-8)

    def test_flat_end_caps(self):
        nn = 5
        self.setup_model(nn, end_cap_depth_ratio=0.0)

        h_frac = np.linspace(0, 1, nn)
        h = 2 * h_frac * self.r
        self.p.set_val("h_liq_frac", h_frac)

        self.p.run_model()

        th = 2 * np.arccos(1 - 2 * h_frac)
        c = 2 * np.sqrt(2 * self.r * h - h**2)
        A_tank = 2 * np.pi * self.r * self.L + 2 * np.pi * self.r**2
        A_wet = th * self.r * self.L + self.r**2 * (th - np.sin(th))

        assert_near_equal(self.p.get_val("A_interface", units="m**2"), c * self.L, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m"), c, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2"), A_wet, tolerance=5e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2"), A_tank - A_wet, tolerance=5e-8)

    def test_derivatives(self):
        nn = 7
        self.setup_model(nn)

        off = 1e-6
        self.p.set_val("h_liq_frac", np.linspace(off, 1.0 - off, nn))

        self.p.run_model()

        partials = self.p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)

    def test_derivatives_ellipsoidal(self):
        nn = 7
        self.setup_model(nn, end_cap_depth_ratio=0.5)

        off = 1e-6
        self.p.set_val("h_liq_frac", np.linspace(off, 1.0 - off, nn))

        self.p.run_model()

        partials = self.p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)

    def test_derivatives_flat_end_caps(self):
        nn = 7
        self.setup_model(nn, end_cap_depth_ratio=0.0)

        off = 1e-6
        self.p.set_val("h_liq_frac", np.linspace(off, 1.0 - off, nn))

        self.p.run_model()

        partials = self.p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)


class BoilOffFillLevelCalcTestCase(unittest.TestCase):
    def setup_model(self, nn=1, end_cap_depth_ratio=1.0):
        self.p = p = om.Problem()
        p.model.add_subsystem(
            "model",
            BoilOffFillLevelCalc(num_nodes=nn, end_cap_depth_ratio=end_cap_depth_ratio),
            promotes=["*"],
        )
        p.setup(force_alloc_complex=True)

        self.r = 0.7
        self.L = 0.3

        self.p.set_val("radius", self.r, units="m")
        self.p.set_val("length", self.L, units="m")

    def test_fill_level(self):
        nn = 7
        self.setup_model(nn)

        r = self.r
        L = self.L
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        fill = np.linspace(0.01, 0.99, nn)
        V_gas = (1 - fill) * V_tank
        self.p.set_val("V_gas", V_gas, units="m**3")

        self.p.run_model()

        assert_near_equal(self.p.get_val("fill_level"), fill, tolerance=1e-10)

        partials = self.p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)

    def test_fill_level_flat_end_caps(self):
        nn = 7
        self.setup_model(nn, end_cap_depth_ratio=0.0)

        r = self.r
        L = self.L
        V_tank = np.pi * r**2 * L
        fill = np.linspace(0.01, 0.99, nn)
        V_gas = (1 - fill) * V_tank
        self.p.set_val("V_gas", V_gas, units="m**3")

        self.p.run_model()

        assert_near_equal(self.p.get_val("fill_level"), fill, tolerance=1e-10)

        partials = self.p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)

    def test_fill_level_ellipsoidal(self):
        nn = 7
        self.setup_model(nn, end_cap_depth_ratio=0.5)

        r = self.r
        L = self.L
        V_tank = 2 / 3 * np.pi * r**3 + np.pi * r**2 * L
        fill = np.linspace(0.01, 0.99, nn)
        V_gas = (1 - fill) * V_tank
        self.p.set_val("V_gas", V_gas, units="m**3")

        self.p.run_model()

        assert_near_equal(self.p.get_val("fill_level"), fill, tolerance=1e-10)

        partials = self.p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)


class HeaterODETestCase(unittest.TestCase):
    def test_simple(self):
        """
        Test the model with random scalars.
        """
        p = om.Problem()
        C = 1.7e-2
        p.model.add_subsystem("model", HeaterODE(num_nodes=1, heater_rate_const=C), promotes=["*"])

        p.setup()

        P_h = 1.5e3
        Q_h = 2e3
        p.set_val("P_heater", P_h, units="W")
        p.set_val("Q_add", Q_h, units="W")

        p.run_model()

        assert_near_equal(p.get_val("Q_add_dot", units="W/s"), C * (P_h - Q_h))

    def test_vectorized(self):
        """
        Test the model with random vectors.
        """
        p = om.Problem()
        C = 1.7e-2
        nn = 3
        p.model.add_subsystem("model", HeaterODE(num_nodes=nn, heater_rate_const=C), promotes=["*"])

        p.setup()

        P_h = np.linspace(1.5e3, 0, nn)
        Q_h = np.linspace(2e3, -3e3, nn)
        p.set_val("P_heater", P_h, units="W")
        p.set_val("Q_add", Q_h, units="W")

        p.run_model()

        assert_near_equal(p.get_val("Q_add_dot", units="W/s"), C * (P_h - Q_h))

    def test_partials(self):
        """
        Test the model with random vectors.
        """
        p = om.Problem()
        C = 1.7e-2
        nn = 3
        p.model.add_subsystem("model", HeaterODE(num_nodes=nn, heater_rate_const=C), promotes=["*"])

        p.setup(force_alloc_complex=True)

        P_h = np.linspace(1.5e3, 0, nn)
        Q_h = np.linspace(2e3, -3e3, nn)
        p.set_val("P_heater", P_h, units="W")
        p.set_val("Q_add", Q_h, units="W")

        p.run_model()

        partials = p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)


class LH2BoilOffODETestCase(unittest.TestCase):
    def test_regression(self):
        """
        Test with some random inputs that results in non-zero cloud condensation
        and bulk boiling terms.
        """
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(heat_transfer_C_liq_const=0.05, heat_transfer_n_liq_const=0.23), promotes=["*"])

        p.setup()

        p.set_val("m_gas", 1.9411308219846288, units="kg")
        p.set_val("m_liq", 942.0670752834986, units="kg")
        p.set_val("T_gas", 21, units="K")
        p.set_val("T_liq", 20.7, units="K")
        p.set_val("V_gas", 1.42, units="m**3")
        p.set_val("m_dot_gas_out", 0.0, units="kg/s")
        p.set_val("m_dot_liq_out", 0.0, units="kg/s")
        p.set_val("A_interface", 4.601815537828035, units="m**2")
        p.set_val("L_interface", 0.6051453090136371, units="m")

        # Add heat evenly around the tank
        Q_tot = 51.4
        A_wet = 23.502425642397316
        A_dry = 5.722240017621729
        p.set_val("Q_liq", Q_tot * A_wet / (A_wet + A_dry), units="W")
        p.set_val("Q_gas", Q_tot * A_dry / (A_wet + A_dry), units="W")

        p.run_model()

        assert_near_equal(p.get_val("m_dot_gas", units="kg/s"), 6.587730120657324e-05, tolerance=1e-12)
        assert_near_equal(p.get_val("m_dot_gas", units="kg/s"), -p.get_val("m_dot_liq", units="kg/s"), tolerance=1e-12)
        assert_near_equal(p.get_val("T_dot_gas", units="K/s"), 0.000616693840187, tolerance=1e-12)
        assert_near_equal(p.get_val("T_dot_liq", units="K/s"), 2.028935422089373e-06, tolerance=1e-12)
        assert_near_equal(p.get_val("V_dot_gas", units="m**3/s"), 9.348683773029187e-07, tolerance=1e-12)
        assert_near_equal(p.get_val("P_gas", units="Pa"), 107531.396349127055146, tolerance=1e-12)

    def test_derivatives(self):
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(heat_transfer_C_liq_const=0.05, heat_transfer_n_liq_const=0.23), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.model.model.H2 = H2_prop_MendezRamos

        p.set_val("m_gas", 1.9411308219846288, units="kg")
        p.set_val("m_liq", 942.0670752834986, units="kg")
        p.set_val("T_gas", 21.239503179127798, units="K")
        p.set_val("T_liq", 20.708930544834377, units="K")
        p.set_val("V_gas", 1.4856099323616818, units="m**3")
        p.set_val("m_dot_gas_out", 0.0, units="kg/s")
        p.set_val("m_dot_liq_out", 0.0, units="kg/s")
        p.set_val("A_interface", 4.601815537828035, units="m**2")
        p.set_val("L_interface", 0.6051453090136371, units="m")

        # Add heat evenly around the tank
        Q_tot = 51.4
        A_wet = 23.502425642397316
        A_dry = 5.722240017621729
        p.set_val("Q_liq", Q_tot * A_wet / (A_wet + A_dry), units="W")
        p.set_val("Q_gas", Q_tot * A_dry / (A_wet + A_dry), units="W")
        p.set_val("Q_add", 10, units="W")

        p.run_model()

        partials = p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)

    def test_vectorized_derivatives_with_heat_and_mass_flows(self):
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(heat_transfer_C_liq_const=0.05, heat_transfer_n_liq_const=0.23, num_nodes=3), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.model.model.H2 = H2_prop_MendezRamos

        p.set_val("m_gas", 1.9411308219846288, units="kg")
        p.set_val("m_liq", 942.0670752834986, units="kg")
        p.set_val("T_gas", 21.239503179127798, units="K")
        p.set_val("T_liq", 20.708930544834377, units="K")
        p.set_val("V_gas", 1.4856099323616818, units="m**3")
        p.set_val("m_dot_gas_out", 0.2, units="kg/s")
        p.set_val("m_dot_liq_out", 0.5, units="kg/s")
        p.set_val("A_interface", 4.601815537828035, units="m**2")
        p.set_val("L_interface", 0.6051453090136371, units="m")
        p.set_val("Q_liq", 80, units="W")
        p.set_val("Q_gas", 15, units="W")
        p.set_val("Q_add", 500, units="W")

        p.run_model()

        partials = p.check_partials(method="cs", compact_print=True, out_stream=None)
        assert_check_partials(partials)


class InitialTankStateModificationTestCase(unittest.TestCase):
    def test_init_values(self):
        p = om.Problem()

        fill = 0.6
        T_liq = 20
        T_gas = 21
        P_gas = 2e5

        p.model.add_subsystem(
            "model",
            InitialTankStateModification(
                num_nodes=1,
                fill_level_init=fill,
                ullage_T_init=T_gas,
                ullage_P_init=P_gas,
                liquid_T_init=T_liq,
            ),
            promotes=["*"],
        )

        p.setup()

        r = 1.3  # m
        L = 0.7  # m

        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        # Set all inputs to zero so we can test the computed initial values
        p.set_val("delta_m_gas", 0.0, units="kg")
        p.set_val("delta_m_liq", 0.0, units="kg")
        p.set_val("delta_T_gas", 0.0, units="K")
        p.set_val("delta_T_liq", 0.0, units="K")
        p.set_val("delta_V_gas", 0.0, units="m**3")

        p.run_model()

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        V_gas = V_tank * (1 - fill)
        m_gas = V_gas * H2_prop.gh2_rho(P_gas, T_gas)
        m_liq = V_tank * fill * H2_prop.lh2_rho(T_liq)

        assert_near_equal(p.get_val("m_gas", units="kg"), m_gas, tolerance=1e-12)
        assert_near_equal(p.get_val("m_liq", units="kg"), m_liq, tolerance=1e-12)
        assert_near_equal(p.get_val("T_gas", units="K"), T_gas, tolerance=1e-12)
        assert_near_equal(p.get_val("T_liq", units="K"), T_liq, tolerance=1e-12)
        assert_near_equal(p.get_val("V_gas", units="m**3"), V_tank * (1 - fill), tolerance=1e-12)

    def test_ellipsoidal(self):
        p = om.Problem()

        fill = 0.6
        T_liq = 20
        T_gas = 21
        P_gas = 2e5

        p.model.add_subsystem(
            "model",
            InitialTankStateModification(
                num_nodes=1,
                fill_level_init=fill,
                ullage_T_init=T_gas,
                ullage_P_init=P_gas,
                liquid_T_init=T_liq,
                end_cap_depth_ratio=0.5,
            ),
            promotes=["*"],
        )

        p.setup(force_alloc_complex=True)

        r = 1.3  # m
        L = 0.7  # m

        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        # Set all inputs to zero so we can test the computed initial values
        p.set_val("delta_m_gas", 0.0, units="kg")
        p.set_val("delta_m_liq", 0.0, units="kg")
        p.set_val("delta_T_gas", 0.0, units="K")
        p.set_val("delta_T_liq", 0.0, units="K")
        p.set_val("delta_V_gas", 0.0, units="m**3")

        p.run_model()

        V_tank = 2 / 3 * np.pi * r**3 + np.pi * r**2 * L
        V_gas = V_tank * (1 - fill)
        m_gas = V_gas * H2_prop.gh2_rho(P_gas, T_gas)
        m_liq = V_tank * fill * H2_prop.lh2_rho(T_liq)

        assert_near_equal(p.get_val("m_gas", units="kg"), m_gas, tolerance=1e-12)
        assert_near_equal(p.get_val("m_liq", units="kg"), m_liq, tolerance=1e-12)
        assert_near_equal(p.get_val("T_gas", units="K"), T_gas, tolerance=1e-12)
        assert_near_equal(p.get_val("T_liq", units="K"), T_liq, tolerance=1e-12)
        assert_near_equal(p.get_val("V_gas", units="m**3"), V_tank * (1 - fill), tolerance=1e-12)

        # Partials must be checked with FD because GH2 prop surrogates are not complex safe
        partials = p.check_partials(method="fd", out_stream=None)
        assert_check_partials(partials, atol=5e-4, rtol=5e-5)

    def test_vectorized(self):
        p = om.Problem()

        nn = 5
        fill = 0.2
        T_liq = 18
        T_gas = 22
        P_gas = 1.6e5

        p.model.add_subsystem(
            "model",
            InitialTankStateModification(
                num_nodes=nn,
                fill_level_init=fill,
                ullage_T_init=T_gas,
                ullage_P_init=P_gas,
                liquid_T_init=T_liq,
            ),
            promotes=["*"],
        )

        p.setup(force_alloc_complex=True)

        r = 0.7  # m
        L = 0.9  # m

        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        # Add some delta to see that it works properly
        val = np.linspace(-10, 10, nn)
        p.set_val("delta_m_gas", val, units="kg")
        p.set_val("delta_m_liq", val, units="kg")
        p.set_val("delta_T_gas", val, units="K")
        p.set_val("delta_T_liq", val, units="K")
        p.set_val("delta_V_gas", val, units="m**3")

        p.run_model()

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        V_gas = V_tank * (1 - fill)
        m_gas = V_gas * H2_prop.gh2_rho(P_gas, T_gas)
        m_liq = V_tank * fill * H2_prop.lh2_rho(T_liq)

        assert_near_equal(p.get_val("m_gas", units="kg"), m_gas + val, tolerance=1e-12)
        assert_near_equal(p.get_val("m_liq", units="kg"), m_liq + val, tolerance=1e-12)
        assert_near_equal(p.get_val("T_gas", units="K"), T_gas + val, tolerance=1e-12)
        assert_near_equal(p.get_val("T_liq", units="K"), T_liq + val, tolerance=1e-12)
        assert_near_equal(p.get_val("V_gas", units="m**3"), V_tank * (1 - fill) + val, tolerance=1e-12)

    def test_partials(self):
        nn = 5
        p = om.Problem()
        p.model.add_subsystem("model", InitialTankStateModification(num_nodes=nn), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("radius", 0.6, units="m")
        p.set_val("length", 1.3, units="m")
        
        # Initial conditions
        p.set_val("fill_level_init", 0.16)
        p.set_val("ullage_T_init", 24, units="K")
        p.set_val("ullage_P_init", 1.7e5, units="Pa")
        p.set_val("liquid_T_init", 20, units="K")

        # Add some delta to see that it works properly
        val = np.linspace(-10, 10, nn)
        p.set_val("delta_m_gas", val, units="kg")
        p.set_val("delta_m_liq", val, units="kg")
        p.set_val("delta_T_gas", val, units="K")
        p.set_val("delta_T_liq", val, units="K")
        p.set_val("delta_V_gas", val, units="m**3")

        p.run_model()

        # Partials must be checked with FD because GH2 prop surrogates are not complex safe
        partials = p.check_partials(method="fd", out_stream=None)
        assert_check_partials(partials, atol=3e-4, rtol=5e-5)


if __name__ == "__main__":
    unittest.main()
