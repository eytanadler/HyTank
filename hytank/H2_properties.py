"""
@File    :   H2_properties.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : Surrogate models based on NIST data to compute thermophysical properties of hydrogen
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import pickle
from time import time

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import scipy.interpolate as interp

# ==============================================================================
# Extension modules
# ==============================================================================
from hytank.H2_property_data.data_parser import get_sat_property, get_property


class HydrogenProperties:
    """
    Class for computing hydrogen properties using surrogate models based on data from
    NIST (https://webbook.nist.gov/cgi/fluid.cgi?ID=C1333740&Action=Page). The surrogate
    models are cached to the H2_property_data folder in two files called
    real_gas_property_surrogate_models.pkl and saturated_property_surrogate_models.pkl.
    To regenerate the surrogate models, delete these two files and they will be automatically
    recreated when this class is initialized.

    The surrogates use SciPy's CubicSpline for the saturated property estimates and
    the CloughTocher2DInterpolator for the real gas properties. CubicSpline provides
    analytic derivatives, while the CloughTocher2DInterpolator is finite differenced.
    """

    def __init__(self, print_output=False):
        self._print_output = print_output

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "H2_property_data")
        sat_dump_file = os.path.join(data_dir, "saturated_property_surrogate_models.pkl")
        gas_dump_file = os.path.join(data_dir, "real_gas_property_surrogate_models.pkl")

        # -------------- Train saturated liquid and gas surrogates --------------
        # Use the cached model if it exists
        if os.path.exists(sat_dump_file):
            if self._print_output:
                print("Reading saturated property surrogate models from cache...", end="")
            t_start = time()
            with open(sat_dump_file, "rb") as f:
                self.sat_surrogates = pickle.load(f)
            if self._print_output:
                print(f"done in {time() - t_start} sec")
        else:
            if self._print_output:
                print("Training saturated property surrogate models...", end="")
            t_start = time()
            self.sat_surrogates = {
                "lh2_P": {"x": get_sat_property("Temperature (K)"), "y": get_sat_property("Pressure (MPa)") * 1e6},
                "lh2_h": {"x": get_sat_property("Temperature (K)"), "y": get_sat_property("Enthalpy (l, kJ/kg)") * 1e3},
                "lh2_u": {
                    "x": get_sat_property("Temperature (K)"),
                    "y": get_sat_property("Internal Energy (l, kJ/kg)") * 1e3,
                },
                "lh2_cp": {"x": get_sat_property("Temperature (K)"), "y": get_sat_property("Cp (l, J/g*K)") * 1e3},
                "lh2_rho": {"x": get_sat_property("Temperature (K)"), "y": get_sat_property("Density (l, kg/m3)")},
                "lh2_k": {
                    "x": get_sat_property("Temperature (K)"),
                    "y": get_sat_property("Therm. Cond. (l, W/m*K)"),
                },
                "lh2_viscosity": {
                    "x": get_sat_property("Temperature (K)"),
                    "y": get_sat_property("Viscosity (l, Pa*s)"),
                },
                "lh2_beta": {
                    "x": get_sat_property("Temperature (K)"),
                    "y": get_sat_property("Thermal Expansion Coefficient (l, 1/K)"),
                },
                "sat_gh2_rho": {"x": get_sat_property("Temperature (K)"), "y": get_sat_property("Density (v, kg/m3)")},
                "sat_gh2_h": {
                    "x": get_sat_property("Temperature (K)"),
                    "y": get_sat_property("Enthalpy (v, kJ/kg)") * 1e3,
                },
                "sat_gh2_cp": {"x": get_sat_property("Temperature (K)"), "y": get_sat_property("Cp (v, J/g*K)") * 1e3},
                "sat_gh2_k": {
                    "x": get_sat_property("Temperature (K)"),
                    "y": get_sat_property("Therm. Cond. (v, W/m*K)"),
                },
                "sat_gh2_viscosity": {
                    "x": get_sat_property("Temperature (K)"),
                    "y": get_sat_property("Viscosity (v, Pa*s)"),
                },
                "sat_gh2_beta": {
                    "x": get_sat_property("Temperature (K)"),
                    "y": get_sat_property("Thermal Expansion Coefficient (v, 1/K)"),
                },
                "sat_gh2_T": {"x": get_sat_property("Pressure (MPa)") * 1e6, "y": get_sat_property("Temperature (K)")},
            }

            for key, val in self.sat_surrogates.items():
                self.sat_surrogates[key]["surrogate"] = interp.CubicSpline(val["x"], val["y"], extrapolate=True)
                self.sat_surrogates[key]["surrogate_deriv"] = self.sat_surrogates[key]["surrogate"].derivative()
                self.sat_surrogates[key]["surrogate_second_deriv"] = self.sat_surrogates[key][
                    "surrogate_deriv"
                ].derivative()

            if self._print_output:
                print(f"done in {time() - t_start} sec")

            # Dump the surrogate model
            t_start = time()
            with open(sat_dump_file, "wb") as f:
                pickle.dump(self.sat_surrogates, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self._print_output:
                print(f"    ...cached surrogate models in {time() - t_start} sec")

        # -------------- Train real gas property surrogates --------------
        # Use the cached model if it exists
        if os.path.exists(gas_dump_file):
            if self._print_output:
                print("Reading real gas property surrogate models from cache...", end="")
            t_start = time()
            with open(gas_dump_file, "rb") as f:
                self.gas_surrogates = pickle.load(f)
            if self._print_output:
                print(f"done in {time() - t_start} sec")
        else:
            if self._print_output:
                print("Training real gas property surrogate models...", end="")
            t_start = time()
            phase = "vapor"

            # Get data from the NIST data tables
            vals = {}
            vals["P"] = get_property("Pressure (MPa)", phase=phase) * 1e6  # Pa
            vals["T"] = get_property("Temperature (K)", phase=phase)  # K
            vals["rho"] = get_property("Density (kg/m3)", phase=phase)  # kg/m^3
            vals["cv"] = get_property("Cv (J/g*K)", phase=phase) * 1e3  # J/(kg-K)
            vals["cp"] = get_property("Cp (J/g*K)", phase=phase) * 1e3  # J/(kg-K)
            vals["u"] = get_property("Internal Energy (kJ/kg)", phase=phase) * 1e3  # J/kg
            vals["h"] = get_property("Enthalpy (kJ/kg)", phase=phase) * 1e3  # J/kg

            surr_keys = {
                "P": ["rho", "T"],
                "rho": ["P", "T"],
                "cv": ["P", "T"],
                "cp": ["P", "T"],
                "u": ["P", "T"],
                "h": ["P", "T"],
            }

            self.gas_surrogates = {}
            for k, v in surr_keys.items():
                self.gas_surrogates[k] = interp.CloughTocher2DInterpolator(
                    np.vstack((vals[v[0]], vals[v[1]])).T, vals[k]
                )

            if self._print_output:
                print(f"done in {time() - t_start} sec")

            # Dump the surrogate model
            t_start = time()
            with open(gas_dump_file, "wb") as f:
                pickle.dump(self.gas_surrogates, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self._print_output:
                print(f"    ...cached surrogate models in {time() - t_start} sec")

        # Step sizes for finite difference
        self.fd_step_P = 1e-1
        self.fd_step_T = 1e-6
        self.fd_step_rho = 1e-6

    def _eval_surrogate(self, name, x, deriv=False):
        """
        Evaluate the surrogate models. If name corresponds to a real gas property, x must be passed
        as a numpy array with dimension (num eval points, 2). If name corresponds to a saturated
        property, x must be passed as either a scalar or 1D numpy array.
        """
        # Real gas properties
        if name in ["P", "rho", "cv", "cp", "u", "h"]:
            if deriv not in [1, True, False]:
                raise ValueError(f"deriv value of {deriv} not supported, must be True, False, or 1")

            # Prevent the solver from calling the surrogate above the pressures it's been trained on
            if name != "P":  # if the pressure surrogate, don't do this
                P_cutoff = 12.4e5
                x[:, 0][x[:, 0] > P_cutoff] = P_cutoff

            val = self.gas_surrogates[name](x)
            if deriv:
                step_first = self.fd_step_rho if name == "P" else self.fd_step_P
                x[:, 0] += step_first
                val_first_step = self.gas_surrogates[name](x)
                x[:, 0] -= step_first

                x[:, 1] += self.fd_step_T
                val_T_step = self.gas_surrogates[name](x)
                x[:, 1] -= self.fd_step_T

                return ((val_first_step - val) / step_first, (val_T_step - val) / self.fd_step_T)
            return val

        # Saturated properties
        else:
            if deriv not in [1, 2, True, False]:
                raise ValueError(f"deriv value of {deriv} not supported, must be True, False, 1, or 2")

            is_float = not isinstance(x, np.ndarray)
            if deriv == 1:  # True == 1 evaluates to True, so this works for both cases
                val = self.sat_surrogates[name]["surrogate_deriv"](x)
            elif deriv == 2:
                val = self.sat_surrogates[name]["surrogate_second_deriv"](x)
            else:
                val = self.sat_surrogates[name]["surrogate"](x)
            return val.item() if is_float else val

    def gh2_P(self, rho, T, deriv=False):
        """
        Compute pressure of hydrogen gas.

        Parameters
        ----------
        rho : float or numpy array
            Hydrogen density (kg/m^3); rho and T must be the same shape if they're both arrays
        T : float or numpy array
            Hydrogen temperature (K); rho and T must be the same shape if they're both arrays
        deriv : bool, optional
            Compute the derivative of the output with respect to rho and T instead
            of the output itself, by default False. If this is set to True, there
            will be two return values that are a numpy array if either P or T is also
            an array and a float otherwise.

        Returns
        -------
        float or numpy array
            Pressure of gaseous hydrogen (Pa) or the derivative with respect to rho
            if deriv is set to True
        float or numpy array
            If deriv is set to True, the derivative of pressure with respect
            to temperature
        """
        # Check inputs
        if isinstance(rho, np.ndarray) and isinstance(T, np.ndarray) and rho.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

        return self._eval_surrogate("P", np.vstack((rho, T)).T, deriv=deriv)

    def gh2_rho(self, P, T, deriv=False):
        """
        Compute density of hydrogen gas.

        Parameters
        ----------
        P : float or numpy array
            Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
        T : float or numpy array
            Hydrogen temperature (K); P and T must be the same shape if they're both arrays
        deriv : bool, optional
            Compute the derivative of the output with respect to P and T instead
            of the output itself, by default False. If this is set to True, there
            will be two return values that are a numpy array if either P or T is also
            an array and a float otherwise.

        Returns
        -------
        float or numpy array
            Density of gaseous hydrogen (kg/m^3) or the derivative with respect to P if deriv is set to True
        float or numpy array
            If deriv is set to True, the derivative of density with respect
            to temperature
        """
        # Check inputs
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

        return self._eval_surrogate("rho", np.vstack((P, T)).T, deriv=deriv)

    def gh2_cv(self, P, T, deriv=False):
        """
        Compute specific heat of hydrogen gas at constant volume.

        Parameters
        ----------
        P : float or numpy array
            Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
        T : float or numpy array
            Hydrogen temperature (K); P and T must be the same shape if they're both arrays
        deriv : bool, optional
            Compute the derivative of the output with respect to P and T instead
            of the output itself, by default False. If this is set to True, there
            will be two return values that are a numpy array if either P or T is also
            an array and a float otherwise.

        Returns
        -------
        float or numpy array
            Specific heat at constant volume of gaseous hydrogen (J/(kg-K)) or
            the derivative with respect to P if deriv is set to True
        float or numpy array
            If deriv is set to True, the derivative of specific heat with respect
            to temperature
        """
        # Check inputs
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

        return self._eval_surrogate("cv", np.vstack((P, T)).T, deriv=deriv)

    def gh2_cp(self, P, T, deriv=False):
        """
        Compute specific heat of hydrogen gas at constant pressure.

        Parameters
        ----------
        P : float or numpy array
            Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
        T : float or numpy array
            Hydrogen temperature (K); P and T must be the same shape if they're both arrays
        deriv : bool, optional
            Compute the derivative of the output with respect to P and T instead
            of the output itself, by default False. If this is set to True, there
            will be two return values that are a numpy array if either P or T is also
            an array and a float otherwise.

        Returns
        -------
        float or numpy array
            Specific heat at constant pressure of gaseous hydrogen (J/(kg-K)) or
            the derivative with respect to P if deriv is set to True
        float or numpy array
            If deriv is set to True, the derivative of specific heat with respect
            to temperature
        """
        # Check inputs
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

        return self._eval_surrogate("cp", np.vstack((P, T)).T, deriv=deriv)

    def gh2_u(self, P, T, deriv=False):
        """
        Compute internal energy of hydrogen gas.

        Parameters
        ----------
        P : float or numpy array
            Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
        T : float or numpy array
            Hydrogen temperature (K); P and T must be the same shape if they're both arrays
        deriv : bool, optional
            Compute the derivative of the output with respect to P and T instead
            of the output itself, by default False. If this is set to True, there
            will be two return values that are a numpy array if either P or T is also
            an array and a float otherwise.

        Returns
        -------
        float or numpy array
            Internal energy of gaseous hydrogen (J/kg) or
            the derivative with respect to P if deriv is set to True
        float or numpy array
            If deriv is set to True, the derivative of specific heat with respect
            to temperature
        """
        # Check inputs
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

        return self._eval_surrogate("u", np.vstack((P, T)).T, deriv=deriv)

    def gh2_h(self, P, T, deriv=False):
        """
        Compute enthalpy of hydrogen gas.

        Parameters
        ----------
        P : float or numpy array
            Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
        T : float or numpy array
            Hydrogen temperature (K); P and T must be the same shape if they're both arrays
        deriv : bool, optional
            Compute the derivative of the output with respect to P and T instead
            of the output itself, by default False. If this is set to True, there
            will be two return values that are a numpy array if either P or T is also
            an array and a float otherwise.

        Returns
        -------
        float or numpy array
            Enthalpy of gaseous hydrogen (J/kg) or
            the derivative with respect to P if deriv is set to True
        float or numpy array
            If deriv is set to True, the derivative of specific heat with respect
            to temperature
        """
        # Check inputs
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

        return self._eval_surrogate("h", np.vstack((P, T)).T, deriv=deriv)

    def lh2_P(self, T, deriv=False):
        """
        Pressure of saturated liquid hydrogen.

        Note: this goes negative for temperature inputs < ~11 K.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Pressure of saturated liquid hydrogen (Pa) or the derivative with
            respect to T if deriv is set to True
        """
        return self._eval_surrogate("lh2_P", T, deriv=deriv)

    def lh2_h(self, T, deriv=False):
        """
        Enthalpy of saturated liquid hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Enthalpy of saturated liquid hydrogen (J/kg) or the derivative
            with respect to T if deriv is set to True
        """
        return self._eval_surrogate("lh2_h", T, deriv=deriv)

    def lh2_u(self, T, deriv=False):
        """
        Internal energy of saturated liquid hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Internal energy of saturated liquid hydrogen (J/kg) or the
            derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("lh2_u", T, deriv=deriv)

    def lh2_cp(self, T, deriv=False):
        """
        Specific heat at constant pressure of saturated liquid hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Specific heat at constant pressure of saturated liquid hydrogen (J/(kg-K))
            or the derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("lh2_cp", T, deriv=deriv)

    def lh2_rho(self, T, deriv=False):
        """
        Density of saturated liquid hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Density of saturated liquid hydrogen (kg/m^3) or the derivative
            with respect to T if deriv is set to True
        """
        return self._eval_surrogate("lh2_rho", T, deriv=deriv)

    def lh2_k(self, T, deriv=False):
        """
        Thermal conductivity of saturated liquid hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Thermal conductivity of saturated liquid hydrogen (W/(m-K)) or
            the derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("lh2_k", T, deriv=deriv)

    def lh2_viscosity(self, T, deriv=False):
        """
        Dynamic viscosity of saturated liquid hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Dynamic viscosity of saturated liquid hydrogen (Pa-s) or
            the derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("lh2_viscosity", T, deriv=deriv)

    def lh2_beta(self, T, deriv=False):
        """
        Coefficient of thermal expansion of saturated liquid hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Coefficient of thermal expansion of saturated liquid hydrogen (1 / K)
            or the derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("lh2_beta", T, deriv=deriv)

    def sat_gh2_rho(self, T, deriv=False):
        """
        Density of saturated gaseous hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Density of saturated gaseous hydrogen (kg/m^3) or the derivative
            with respect to T if deriv is set to True
        """
        return self._eval_surrogate("sat_gh2_rho", T, deriv=deriv)

    def sat_gh2_h(self, T, deriv=False):
        """
        Enthalpy of saturated gaseous hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Enthalpy of saturated gaseous hydrogen (J/kg) or the derivative
            with respect to T if deriv is set to True
        """
        return self._eval_surrogate("sat_gh2_h", T, deriv=deriv)

    def sat_gh2_cp(self, T, deriv=False):
        """
        Specific heat at constant pressure of saturated gaseous hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Specific heat at constant pressure of saturated gaseous hydrogen
            (J/(kg-K)) or the derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("sat_gh2_cp", T, deriv=deriv)

    def sat_gh2_k(self, T, deriv=False):
        """
        Thermal conductivity of saturated gaseous hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Thermal conductivity of saturated gaseous hydrogen (W/(m-K)) or
            the derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("sat_gh2_k", T, deriv=deriv)

    def sat_gh2_viscosity(self, T, deriv=False):
        """
        Dynamic viscosity of saturated gaseous hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Dynamic viscosity of saturated gaseous hydrogen (Pa-s) or
            the derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("sat_gh2_viscosity", T, deriv=deriv)

    def sat_gh2_beta(self, T, deriv=False):
        """
        Coefficient of thermal expansion of saturated gaseous hydrogen.

        Parameters
        ----------
        T : float or numpy array
            Hydrogen temperature (K)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to T instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Coefficient of thermal expansion of saturated gaseous hydrogen (1 / K)
            or the derivative with respect to T if deriv is set to True
        """
        return self._eval_surrogate("sat_gh2_beta", T, deriv=deriv)

    def sat_gh2_T(self, P, deriv=False):
        """
        Temperature of saturated gaseous hydrogen at the specified pressure.

        Parameters
        ----------
        P : float or numpy array
            Hydrogen pressure (Pa)
        deriv : bool or int, optional
            Compute the first derivative of the output with respect to P instead
            of the output itself, or alternatively set deriv to 2 to return
            the second derivative, by default False (setting deriv to 1 is
            equivalent to setting it to True)

        Returns
        -------
        float or numpy array
            Temperature of saturated gaseous hydrogen (K) or the derivative
            with respect to P if deriv is set to True
        """
        return self._eval_surrogate("sat_gh2_T", P, deriv=deriv)
