import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
import scripts.models.H2_properties_MendezRamos as H2_prop_MendezRamos
from scripts.models.H2_properties import HydrogenProperties
from scripts.models.boil_off import *


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

        A_wet = np.array([5.95723, 5.48700, 5.08361, 4.71719, 4.37258, 4.04051, 3.71445, 3.38888, 3.05836, 2.71649, 2.35443])
        A_dry = np.array([1.51976, 1.98999, 2.39338, 2.75980, 3.10441, 3.43648, 3.76254, 4.08811, 4.41863, 4.76050, 5.12256])
        Q_tot = 50
        p.set_val("Q_liq", Q_tot * A_wet / (A_wet + A_dry), units="W")
        p.set_val("Q_gas", Q_tot * A_dry / (A_wet + A_dry), units="W")

        p.set_val("P_heater", 1, units="kW")

        p.run_model()

        assert_near_equal(
            p.get_val("m_gas", units="kg"),
            np.array([0.29324779357702396, 0.9755895133768798, 2.2827021586516327, 3.9076803536638063, 5.653242606173135, 7.443009431496469, 9.267032677754145, 11.133543749805373, 13.050353608670388, 15.025465846370599, 17.076664226060885]),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("m_liq", units="kg"),
            np.array([121.77078809767069, 112.02844637787082, 101.66133373259608, 90.9763555375839, 80.17079328507457, 69.32102645975124, 58.43700321349357, 47.51049214144234, 36.53368228257732, 25.498570044877113, 14.3873716651868]),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_gas", units="K"),
            np.array([21.0, 27.22920929665294, 29.48250288827393, 30.051772606519144, 30.224005084840208, 30.404892718292665, 30.62040539830591, 30.827574742229217, 31.010484383850883, 31.18200821373119, 31.375056927935542]),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_liq", units="K"),
            np.array([20.0, 20.144187520412697, 20.421985779163165, 20.790750319602548, 21.224204973496075, 21.717111106362765, 22.280328759832923, 22.937024193511277, 23.727494296066705, 24.732382281254885, 26.16774969882635]),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("P_gas", units="Pa"),
            np.array([120025.05024726999, 289995.72285539535, 475610.7007027782, 593235.268619668, 662839.4198457345, 710377.2007241278, 747352.5474152574, 777417.4557615193, 802338.8010654534, 823933.7339836007, 844515.0694483466]),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("fill_level"),
            np.array([0.9, 0.8279470218438476, 0.750987417594691, 0.6712536196684832, 0.590083423726707, 0.5079259185456383, 0.4247197650332723, 0.3402212682006155, 0.25409836202841996, 0.16582612755657833, 0.07427322580024691]),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("integ.Q_add", units="kW"),
            np.array([0.0, 0.6235501924944253, 0.8771597679332102, 0.9529132774798601, 0.9716877270153, 0.9810342748754619, 0.9896598375045959, 0.9958778063635108, 0.9989090690361989, 0.9998159416698612, 0.9999005115250991]),
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
        p.model.add_subsystem("model", LH2BoilOffODE(), promotes=["*"])

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

        assert_near_equal(p.get_val("m_dot_gas", units="kg/s"), 5.509489041153447e-05, tolerance=1e-12)
        assert_near_equal(p.get_val("m_dot_gas", units="kg/s"), -p.get_val("m_dot_liq", units="kg/s"), tolerance=1e-12)
        assert_near_equal(p.get_val("T_dot_gas", units="K/s"), 0.0005454013001323098, tolerance=1e-12)
        assert_near_equal(p.get_val("T_dot_liq", units="K/s"), 2.5540236536296463e-06, tolerance=1e-12)
        assert_near_equal(p.get_val("V_dot_gas", units="m**3/s"), 7.818545971578757e-07, tolerance=1e-12)
        assert_near_equal(p.get_val("P_gas", units="Pa"), 107531.39634912706, tolerance=1e-12)

    def test_derivatives(self):
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(), promotes=["*"])

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
        p.model.add_subsystem("model", LH2BoilOffODE(num_nodes=3), promotes=["*"])

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

        partials = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)

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

        # Add some delta to see that it works properly
        val = np.linspace(-10, 10, nn)
        p.set_val("delta_m_gas", val, units="kg")
        p.set_val("delta_m_liq", val, units="kg")
        p.set_val("delta_T_gas", val, units="K")
        p.set_val("delta_T_liq", val, units="K")
        p.set_val("delta_V_gas", val, units="m**3")

        p.run_model()

        partials = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
