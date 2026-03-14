"""This file contains the authors objects for the sizing of liquid propellant pumps"""

from turborocket.fluids.fluids import IncompressibleFluid
import numpy as np
from enum import Enum
import pandas as pd

import warnings


class DiffuserType(Enum):

    circular = "Circular"
    rectangular = "rectangular"


class Barske:
    """This object represents a Barske Impeller Pump

    It is used for sizing and efficiency evaluation of the pump under a range of conditions
    """

    def __init__(
        self,
        dp: float,
        m_dot: float,
        N: float,
    ) -> None:
        """Constructor for the Barske Impeller Object

        Args:
            dp (float): Pressure Head Across the Pump (Pa)
            m_dot (float): Nominal Mass Flow Rate Through the Pump (kg/s)
            N (float): Nominal Shaft Speed of the Pump (rad/s)
        """
        self._dp = dp
        self._m_dot = m_dot
        self._N = N
        # We also define our conversion parameters
        self._ms_to_fts = 3.28084
        self._kgm3_to_lbcuft = 0.062428
        self._m_to_in = 39.3701
        self._hp_to_w = 745.7
        self._g = 9.81  # m/s^2

    def size_pump(
        self,
        fluid: IncompressibleFluid,
        l_1: float,
        l_2: float,
        v_0: float = 3.65,
        v_3f: float = 0.85,
        d_1f: float = 1.1,
        a_3f: float = 3.5,
        delta_div: float = 8,
        diffuser_type: DiffuserType = DiffuserType.circular,
        psi: float = 0.2,
    ) -> pd.DataFrame:
        """This function performs the sizing of the pump, based on the inputs of the user

        Args:
            fluid (IncompressibleFluid): Fluid flowing through the Pump
            l_1 (float): Blade Axial Length at Impeller Inlet (m)
            l_2 (float): Blade Axial Length at Impeller Exit (m)
            v_0 (float, Optional): Impeller Inlet Axial Velocity (m/s) . Defaults to 12 ft/s
            v_3f (float, Optional): Diffuser Velocity Factor (0.8 - 0.9). Defaults to 0.85.
            d_1f (float, Optional): Pump Inlet Preswirl Area Expansion Factor. Defaults to 1.1
            a_3f (float, Optional): Diffuser Expansion Area Factor (3 - 4). Defaults to 3.5.
            delta_div (float, Optional): Diffuser Full Exit Angle. For a circular diffuser: [8 - 10]. For a square diffuser: [4 - 8]. (Degrees). Defaults to 10 degree.
            diffuser_type (DiffuserType, Optional): Diffuser Type Enum Object. Defaults to a Circular Diffuser
            psi (float, Optional): Pressure Head Factor for the Impeller, ranging between 0 and 1. Defaults to 0.2.

        Returns:
            pd.DataFrame: DataFrame of Key Geometric Parameters for the Sizing of the Pump
        """

        # We check if the inlet velocity exceeds recommended guidelines

        if v_0 < 5 / self._ms_to_fts:
            warnings.warn(
                f"Inlet Velocity below Recommendation: {v_0*self._ms_to_fts} < 5 ft/s", UserWarning
            )
        
        if v_0 > 12 / self._ms_to_fts:
            warnings.warn(
                f"Inlet Velocity above Recommendation: {v_0*self._ms_to_fts} > 12 ft/s", UserWarning
            )
        
        if l_1 <= 0:
            raise ValueError(f"Blade Entrance Length must be a positive number")
        elif l_2 <= 0:
            raise ValueError(f"Blade Exit Axial Length must be a positive Number")
        if l_1 < l_2:
            raise ValueError(f"Axial Blade Shape is unacceptable! l_1 > l_2")

        # We then check the divergence angle based on the diffuer
        if diffuser_type == DiffuserType.circular:
            if delta_div < 8 or delta_div > 10:
                warnings.warn(
                    f"Divergence Angle of the Diffuser is outwith recommendation for a Circular Diffuser: 8 < {delta_div} < 10", UserWarning
                )
        elif diffuser_type == DiffuserType.rectangular:
            if delta_div < 4 or delta_div > 8:
                warnings.warn(
                    f"Divergence Angle of the Diffuser outwith recommendation for a Rectangular Diffuser: 4 < {delta_div} < 8", UserWarning
                )

        self._psi = psi

        # We get fluid properties
        rho = fluid.get_density()

        # Defining Inlet Conditions
        self._v_0 = v_0
        self._d_0 = 2 * (self._m_dot / (np.pi * rho * self._v_0)) ** (1 / 2)
        self._d_1 = self._d_0 * d_1f

        self._u_1 = (self._d_1 / 2) * self._N  # m/s

        if self._u_1 * self._ms_to_fts > 150:
            warnings.warn(
                f"Inner Blade Speed Exceeds 150 ft/s, slow down blade by either decreasing inlet velocity or diameter. {self._u_1* self._ms_to_fts} > 150 ft/s", UserWarning
            )

        # Evaluates for the head of the pump and associated required exit velocity
        self._h = self._dp / (rho * self._g)  # m

        self._u_2 = ((2 * self._g * self._h + self._u_1**2) / (1 + psi)) ** (1 / 2)
        self._d_2 = 2 * self._u_2 / (self._N)

        # Checking if the impeller Axial Lengths are within guidelines
        self._l_1 = l_1
        self._l_2 = l_2

        if self._l_1 < 0.25 * self._d_1:
            warnings.warn(
                f"Impeller Inlet Blade Axial Width is below recommended guidelines for this design point: {self._l_1} < (1/4) {self._d_1}", UserWarning
            )

        elif self._l_2 < self._l_1 * (self._d_1 / self._d_2):
            warnings.warn(
                f"Impeller Exit Blade Axial Width is below recommended Guideliness for this design point: {self._l_2} < {self._l_1*(self._d_1/self._d_2)}", UserWarning
            )

        # We evaluate for our absolute and relative exit velocities, assuming no pre-swirl and an exit angle of 90 degrees.
        self._v_1 = self._m_dot / (rho * self._l_1 * np.pi * self._d_1)
        self._w_2 = self._m_dot / (rho * self._l_2 * np.pi * self._d_2)

        # We can evluate for the axial and radial clearance
        self._c_1 = self._d_2 / 100

        if self._c_1 * self._m_to_in > 0.04:
            self._c_1 = 0.04 / self._m_to_in

        self._c_2 = self._d_2 * 0.05

        # We check if the exit blade length
        if self._l_2 < 3 * self._c_1:
            warnings.warn(
                f"Axial Blade Length at Exit is too low when compared to the axial clearance of the Impeller. {l_2} > {3*self._c_1}", UserWarning
            )

        # Now Defining our Diffuser Parameters
        self._v_3 = v_3f * self._u_2
        
        self._a_3 = self._m_dot / (rho * self._v_3)
        
        print(f"v_3: {self._v_3}")
        print(f"A_3: {self._a_3}")
        
        self._a_4 = self._a_3 * a_3f

        # We figure out what the dimensions of the diffuser are:
        self._delta = delta_div

        if diffuser_type == DiffuserType.circular:
            self._d_3 = ((4 / np.pi) * self._a_3) ** 0.5
            self._d_4 = ((4 / np.pi) * self._a_4) ** 0.5
            self._l_3 = (self._d_4 - self._d_3) / np.tan(np.deg2rad(self._delta))

        elif diffuser_type == DiffuserType.rectangular:
            # Assume that the diffuser exit dimensions are like this
            self._d_3 = self._a_3 / self._l_2
            self._d_4 = self._a_4 / self._l_2
            self._l_3 = (self._d_4 - self._d_3) / np.tan(np.deg2rad(self._delta))

        # Finally solving for the exit velocity of the diffuser
        self.v_4 = (self._m_dot / rho) / self._a_4

        # We can create our dataframe

        data = {
            "Eye Diameter - d_0 (mm)": [self._d_0 * 1e3],
            "Inlet Diameter - d_1 (mm)": [self._d_1 * 1e3],
            "Exit Diameter - d_2 (mm)": [self._d_2 * 1e3],
            "Entrance Axial Blade Length - l_1 (mm)": [self._l_1 * 1e3],
            "Exit Axial Blade Legnth - l_2 (mm)": [self._l_2 * 1e3],
            "Axial Clearance - c_1 (mm)": [self._c_1 * 1e3],
            "Radial Clearance - c_2 (mm)": [self._c_2 * 1e3],
            "Diffuser Type": [diffuser_type],
            "Diffuser Throat - d_3 (mm)": [self._d_3 * 1e3],
            "Diffuser Exit - d_4 (mm)": [self._d_4 * 1e3],
            "Diffuser Length - L (mm)": [self._l_3 * 1e3],
            "Inlet Eye Velocity - v_o (m/s)": [self._v_0],
            "Impeller Inlet Velocity - v_1 (m/s)": [self._v_1],
            "Relative Exit Velocity - w_2 (m/s)": [self._w_2],
        }

        df = pd.DataFrame(data=data)

        return df

    def size_expeller(
        self,
        fluid: IncompressibleFluid,
        d_1: float,
        l_1: float,
        l_2: float,
        psi: float = 0.2,
    ) -> float:
        # We check if the inlet velocity exceeds recommended guidelines
        if l_1 <= 0:
            raise ValueError(f"Blade Entrance Length must be a positive number")
        elif l_2 <= 0:
            raise ValueError(f"Blade Exit Axial Length must be a positive Number")

        if l_1 < l_2:
            raise ValueError(f"Axial Blade Shape is unacceptable! l_1 > l_2")

        self._l_1 = l_1
        self._l_2 = l_2
        self._psi = psi

        # We get fluid properties
        rho = fluid.get_density()
        self._d_1 = d_1

        self._u_1 = (self._d_1 / 2) * self._N  # m/s

        # Evaluates for the head of the pump and associated required exit velocity
        self._h = self._dp / (rho * self._g)  # m

        self._u_2 = ((2 * self._g * self._h + self._u_1**2) / (1 + psi)) ** (1 / 2)
        self._d_2 = 2 * self._u_2 / (self._N)

        self._c_1 = self._d_2 / 100

        if self._c_1 * self._m_to_in > 0.04:
            self._c_1 = 0.04 / self._m_to_in

        self._c_2 = self._d_2 * 0.05

        data = {
            "Inlet Diameter - d_1 (mm)": [self._d_1 * 1e3],
            "Exit Diameter - d_2 (mm)": [self._d_2 * 1e3],
            "Entrance Axial Blade Length - l_1 (mm)": [self._l_1 * 1e3],
            "Exit Axial Blade Legnth - l_2 (mm)": [self._l_2 * 1e3],
            "Axial Clearance - c_1 (mm)": [self._c_1 * 1e3],
            "Radial Clearance - c_2 (mm)": [self._c_2 * 1e3],
        }

        df = pd.DataFrame(data=data)

        return df

    def get_pump_performance(
        self,
        fluid: IncompressibleFluid,
        m_dot: float,
        psi: float | None = None,
        N: float | None = None,
    ) -> pd.DataFrame:
        """Function that solves for the pumps performance metrics, namely the expected pressure rise, efficiency and required pump power

        Args:
            fluid (IncompressibleFluid): Inlet Fluid Object of the Pump
            m_dot (float): Mass Flow Rate Through Pump.
            psi (float | None, optional): Pressure Factor of the pump. Defaults to Parameter Defined During Sizing.
            N (float | None, optional): Shaft Speed of the Pump (rad/s). Defaults to design shaft speed.

        Returns:
            pd.Dataframe: Dataframe of Pump Performance Metrics
        """

        if psi is None:
            psi = self._psi

            print(psi)

        if N is None:
            N = self._N

        # Getting our fluid density
        rho = fluid.get_density()

        # Solving for the pressure head and pressure rised
        H = self.get_head(psi=psi, N=N)
        dp = H * rho * self._g

        # We then get the efficiency of the pump
        eta = self.get_instantaneous_efficiency(m_dot=m_dot, fluid=fluid, psi=psi, N=N)

        # We can then solve for our required shaft power
        pw_hyd = dp * m_dot / rho
        pw_shaft = (dp * m_dot / rho) / eta

        # From here, we can assemble our dictionary accordingly
        df = pd.DataFrame(
            data={
                "Head Rise (m)": [H],
                "Head Rise (Bar)": [dp / 1e5],
                "Efficiency (%)": [eta * 1e2],
                "Hydraulic Power (kW)": [pw_hyd / 1e3],
                "Required Shaft Power (kW)": [pw_shaft / 1e3],
            }
        )

        return df

    def get_head(
        self,
        psi: float,
        N: float,
    ) -> float:
        """Function that solves for the head rise in the pump

        Args:
            psi (float): Pressure Factor of the Pump (n.d)
            N (float): Shaft Speed of the Pump (rad/s)

        Returns:
            float: Head Rise across the impeller of the pump (m)
        """
        # We firstly need to evaluate for the blade speeds at the tip and at the eye inlet
        u_2 = (self._d_2 / 2) * N
        u_1 = (self._d_1 / 2) * N

        H = (1 / (2 * self._g)) * ((1 + psi) * u_2**2 - u_1**2)

        return H

    def get_instantaneous_efficiency(
        self,
        m_dot: float,
        fluid: IncompressibleFluid,
        psi: float | None = None,
        N: float | None = None,
    ):
        """Function for evaluating the pump efficiency

        Args:
            m_dot (float): Mass Flow Rate Through the Pump (kg/s)
            fluid (IncompressibleFluid): Inlet fluid object of the pump
            psi (float | None, optional): Pressure Factor of the Impeller. Defaults to Parameter Defined During Sizing.
            N (float | None, optional): Shaft Speed of the Pump (rad/s). Defaults to design shaft speed.
        """
        # Getting our fluid properties
        rho = fluid.get_density()

        # Calculating our theoretical and actual pressure heads
        p_prime = self.get_head(psi=1, N=N)
        p = self.get_head(psi=psi, N=N)

        # Calculating our powers associated with the system
        Q = m_dot / rho
        pw_prime = p_prime * rho * self._g * Q
        pw = p * rho * self._g * Q

        # We now need to convert our key parameters into imperial for the friction calculation
        v = (fluid.get_viscosity() / rho) * self._ms_to_fts**2
        rho = rho * self._kgm3_to_lbcuft
        N = N * 60 / (2 * np.pi)
        d_2 = self._d_2 * self._m_to_in
        d_1 = self._d_1 * self._m_to_in
        l_1 = self._l_1 * self._m_to_in

        alpha_1 = np.arctan((self._d_2 / 2 - self._d_1 / 2) / (self._l_1 - self._l_2))

        pf = (
            0.6e-6
            * (rho)
            * (v**0.2)
            * ((N / 1000) ** 2.8)
            * ((((1 / np.sin(alpha_1)) + 1) * d_2**4.6) + (l_1 * 9.2 * d_1**3.6))
        ) * self._hp_to_w

        eta = pw / (pw_prime + pf)

        return eta

    def get_pump_paddle_power(
        self,
        fluid: IncompressibleFluid,
        d_1: float | None = None,
        d_2: float | None = None,
        l_1: float | None = None,
        l_2: float | None = None,
        N: float | None = None,
    ):
        """Function for evaluating the pump efficiency

        Args:
            fluid (IncompressibleFluid): Inlet fluid object of the pump
            d_1 (float | None, optional): Inner Diameter of the Paddle. Defaults to Parameter Defined During Sizing.
            d_2 (float | None, optional): Outer DIameter of the Paddle. Defaults to Parameter Defined During Sizing.
            l_1 (float | None, optional): Inner Blade Height of the Paddle. Defaults to Parameter Defined During Sizing.
            l_2 (float | None, optional): Outer Blade Height of the Paddle. Defaults to Parameter Defined During Sizing.
            N (float | None, optional): Shaft Speed of the Pump (rad/s). Defaults to design shaft speed.
        """
        # We consider conditional parameters
        if d_1 is None:
            d_1 = self._d_1

        if d_2 is None:
            d_2 = self._d_2

        if l_1 is None:
            l_1 = self._l_1

        if l_2 is None:
            l_2 = self._l_2

        # We now need to convert our key parameters into imperial for the friction calculation
        v = (fluid.get_viscosity() / fluid.get_density()) * self._ms_to_fts**2
        rho = fluid.get_density() * self._kgm3_to_lbcuft

        N = N * 60 / (2 * np.pi)

        d_1 = self._d_1 * self._m_to_in
        d_2 = self._d_2 * self._m_to_in
        l_1 = self._l_1 * self._m_to_in
        l_2 = self._l_2 * self._m_to_in

        if l_1 == l_2:
            alpha_1 = 90
        else:
            alpha_1 = np.arctan((d_2 / 2 - d_1 / 2) / (l_1 - l_2))

        pf = (
            0.6e-6
            * (rho)
            * (v**0.2)
            * ((N / 1000) ** 2.8)
            * ((((1 / np.sin(alpha_1)) + 1) * d_2**4.6) + (l_1 * 9.2 * d_1**3.6))
        ) * self._hp_to_w

        return pf
