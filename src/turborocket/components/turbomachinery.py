"""This file contains the turbomachinery compoents used for transient modelling and similar"""

from turborocket.fluids.fluids import IncompressibleFluid, IdealGas
from turborocket.solvers.solver import adjoint
import numpy as np

class Turbine:
    """Object That Defines the Turbine Transient Performance"""

    def __init__(self, a_rat: float, D_m: float, eta_nom: float, u_co_nom: float):
        """Constructor for the Transient Turbine Object

        Args:
            a_rat (float): Area Ratio of Nozzle
            d_m (float): Mean Diameter of Turbine (m)
            eta_nom (float): Nominal Turbine Efficiency (%)
            u_co_nom (float): Nominal Blade Speed Ratio for Turbine
        """
        self._a_rat = a_rat
        self._rm = D_m / 2
        self._eta_nom = eta_nom
        self._u_co_nom = u_co_nom

        return

    def set_performance(
        self,
        eta_nom: float,
        u_co_nom: float,
    ) -> None:
        """Function that sets the turbines performance

        Args:
            eta_nom (float): Nominal Effiency
            u_co_nom (float): Nominal Blade Speed Ratio for Turbine
        """

        self._eta_nom = eta_nom
        self._u_co_nom = u_co_nom

        return

    def get_isentropic_velocity(
        self,
        combustion_gas: IdealGas,
        p_exit: float,
    ) -> float:
        """This function solves for the insentropic velocity of the gas, based on the expansion ratio of the gas

        Args:
            combustion_gas (IdealGas): Combustion Gas Object Produced by the Gas Generator
            p_exit (float): Exit Static Pressure for the Turbine

        Returns:
            float: Isentropic Expansion Velocity of the gas (m/s)
        """
        # We simply call the gas function to resolve for the isentropic expansion velocity

        v_is = combustion_gas.get_cis(p1=p_exit)

        return v_is

    def get_mean_speed(self, N: float) -> float:
        """Need to get the blade speed of the turbine

        Args:
            N (float): Rotational Rate of the Speed (rad/s)

        Returns:
            float: Blade Speed of the Turbine (m/s)
        """
        U = self._rm * N

        return U

    def get_efficiency(
        self, combustion_gas: IdealGas, N: float, p_exit: float
    ) -> float:
        """This function gets the efficiency of the turbine stage at off-design performance

        Assumptions:
            - Based on the Goldman Paper, a linear relationship has been assumed as correlelated to the blade speed ratio.
            - The efficiency is augmented driven based on the maximum expansion ratio of the nozzles at a given shaft speed,
              where if the expansion ratio is higher than as designed, the efficiency will be naturally depreciated.

        Args:
            N (float): Shaft Speed of the Turbopump (rad/s)
            c_o (float): Isentropic Expansion Velocity (m/s)
            p_exit (float): Exit Static Pressure of the GG Stage (Pa)

        Returns:
            float: Total to static efficiency of the turbine (%)
        """
        # We need to intially solve for the meanline blade speed
        u_m = self._rm * N

        # We get the isentropic velocity of the gas
        c_o = self.get_isentropic_velocity(combustion_gas=combustion_gas, p_exit=p_exit)

        u_co_a = u_m / c_o

        if u_co_a > self._u_co_nom:

            eta_bep = self._eta_nom
        else:

            eta_bep = self._eta_nom * u_co_a / self._u_co_nom

        # Finally we need to evaluate what the maxium expansion velocity of the gas is based on the Mach Number
        M = self.get_supersonic_mach(gamma=combustion_gas.get_gamma())

        # We solve our expansion pressur ratio
        P_min = self.get_exit_pressure(
            P_o=combustion_gas.get_pressure(), M=M, gamma=combustion_gas.get_gamma()
        )

        dh_max = combustion_gas.get_enthalpy_drop(p1=P_min)

        dh_theo = combustion_gas.get_enthalpy_drop(p1=p_exit)

        if dh_max == 0:
            return 0
        # We check if the nozzle is underexpanded
        if P_min > p_exit:
            # We then augment the efficiency accordingly to match expectations by clamping power output

            eta = eta_bep * (dh_max / dh_theo) ** 2

        else:
            # We invert this plot and adjust the efficiency based on the difference in enthalpy expansions

            eta = eta_bep * (dh_theo / dh_max) ** 2

        return eta

    def area_error(self, M: float, gamma: float):
        """Error Function for the Nozzle Expansion Ratio Relationship

        Args:
            M (float): Mach number at the Exit of the Nozzle
            gamma (float): Specific Heat Ratio of the Gas
        """

        rhs = (1 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        )

        error = rhs - self._a_rat

        return error

    def get_subsonic_mach(self, gamma):
        """This Function Solves for the subsonic Mach Number solution

        Args:
            gamma (_type_): Specific Heat Ratio of the Gas

        Returns:
            _type_: Subsonic Mach number
        """
        M = adjoint(
            func=self.area_error,
            x_guess=0.4,
            dx=0.01,
            n=500,
            relax=1,
            target=0,
            params=[gamma],
        )

        return M

    def get_supersonic_mach(self, gamma):
        """This function solves for the Supersonic Mach Number solution

        Args:
            gamma (_type_): Specific Heat Ratio of the Gas

        Returns:
            _type_: Supersonic Mach Number
        """
        M = adjoint(
            func=self.area_error,
            x_guess=2,
            dx=0.01,
            n=500,
            relax=1,
            target=0,
            params=[gamma],
        )

        return M

    def get_static_temp(self, T_o: float, M: float, gamma: float) -> float:
        """This function solves for the static temperature of the gas based on the mach number.

        Args:
            T_o (float): Stagnation temperature of Gas (K)
            M (float): Mach Number of Gas Flow

        Returns:
            float: Static Temperature of Gas
        """

        T = T_o / (1 + ((gamma - 1) / 2) * M**2)

        return T

    def get_exit_pressure(self, P_o: float, M: float, gamma: float) -> float:
        """This function solves for the static temperature of the gas based on the mach number.

        Args:
            P_o (float): Stagnation pressure of the gas (Pa)
            M (float): Mach Number of the Gas Flow
            gamma (float): Specific Heat Ratio of the Gas

        Returns:
            float: Static Pressure of the Gas
        """

        P = P_o / (1 + ((gamma - 1) / 2) * M**2) ** (gamma / (gamma - 1))

        return P

    def get_torque(self, combustion_gas: IdealGas, P_exit: float, N: float) -> float:
        """This function solves for the Torque produced by the Turbine Stage

        Args:
            combustion_gas (IdealGas): Combustion Gas Products Produced by the Gas Generator
            P_exit (float): Exit Pressure of the Turbine (Pa)
            N (float): Shaft Speed of the Turbine (rad/s)

        Returns:
            float: Torque produced by the Turbine
        """

        # We can now solve for the expected efficiency of the system
        eta = self.get_efficiency(combustion_gas=combustion_gas, N=N, p_exit=P_exit)

        # We can now solve for the power being produced by the turbine
        Pw = eta * combustion_gas.get_enthalpy_drop(p1=P_exit)

        # We can finally solve for the torque produced by the turbine by dividing it by the current shaft speed.
        T = Pw / N

        return T


class Pump:
    """Object representing the transient functionality of the pump"""

    def __init__(
        self,
        D_1: float,
        D_2: float,
        D_3: float,
    ):
        """_summary_

        Args:
            D_1 (float): Inner Diameter of Pump Eye (m)
            D_2 (float): Outer Diameter of the Pump Tip (m)
            D_3 (float): Diffuser Oulet Diameter (m)
        """

        self._D_1 = D_1
        self._D_2 = D_2
        self._D_3 = D_3

        self._g = 9.18

        return

    def set_performance(
        self,
        C_c: float,
        psi: float,
        eta_bep: float,
        N_nom: float,
    ) -> None:
        """This Function Sets the Pump Performance

        Args:
            C_c (float): Diffuser Vena-Contra Factor (%)
            psi (float): Pressure Coefficient
            eta_bep (float): Best Efficiency Point (%)
            N_nom (float): Nominal Shaft Speed (rad/s)
        """
        self._C_c = C_c
        self._psi = psi
        self._eta_bep = eta_bep
        self._N_nom = N_nom

        return

    def shut_off_head(self, N: float) -> float:
        """This function estimates the the theoretical shut off head of a pump

        Args:
            N (float): Rotational Rate for the Pump (rad/s)
        """

        u_1 = N * (self._D_1 / 2) ** 2
        u_2 = N * (self._D_2 / 2) ** 2

        H_o = (1 / (2 * self._g)) * ((1 + self._psi) * u_2**2 - u_1**2)

        return H_o

    def get_q_max(self, N: float) -> float:
        """This function gets the maximum flow operating point for the turbine at the selected shaft speed

        Args:
            N (float): Shaft Speed (Rad/s)

        Returns:
            float: Maximum Flow Operating Point (m^3/s)
        """

        u_1 = N * (self._D_1 / 2) ** 2
        u_2 = N * (self._D_2 / 2) ** 2

        v_3 = ((1 + self._psi) * u_2**2 - u_1**2) ** (1 / 2)

        a_3 = np.pi * (self._D_3 / 2) ** 2

        return a_3 * v_3 * self._C_c

    def get_eta_bep(self, N: float) -> float:
        """Simpliefied Model for Identifying what the best operating efficiency of the pump is

        Args:
            N (float): Shaft Speed (Rad/s)

        Returns:
            float: Best Operating Efficiency of Pump (%)
        """

        return self._eta_bep * (N / self._N_nom)

    def get_eta(self, Q: float, N: float, fluid: IncompressibleFluid) -> float:
        """Simplified function that solves for the efficiency of the Pump

        Args:
            Q (float): Flow Rate of Fluid Through the Pump (m^3/s)

        Returns:
            float: Efficiency of the Turbine
        """
        # We need to get the fixed shaft power
        Q_max = self.get_q_max(N=N)
        H_o = self.get_head(Q=0, N=N)

        P_max = fluid.get_density() * self._g * H_o * Q_max

        # We can then figure out what our shaft power is
        P_shaft = P_max / self.get_eta_bep(N=N)

        # We can then evaluate for what the actual power is
        H_a = self.get_head(Q=Q, N=N)

        P_actual = fluid.get_density() * self._g * H_a * Q

        # We can thus solve for the efficiency
        eta = P_actual / P_shaft

        if N == 0:
            eta = 0

        return eta

    def get_head(self, Q: float, N: float) -> float:
        """This function solves for the head produced by the pump

        Args:
            Q (float): Volumetric Flow Rate Through the Pump (m^3 /s)
            N (float): Rotational Rate for the Pump (Rad/s)

        Returns:
            float: Head Produced by Pump (m)
        """

        # We firstly need to solve for the shut_off head of the pump
        H_o = self.shut_off_head(N=N)
        print(f"Shut off Head: {H_o} m")
        # We need to get the maximum flow operating point
        Q_max = self.get_q_max(N=N)

        if Q_max == 0:
            # Pump is not spinning at all, hence no head.
            H = 0
            return 0

        # We will model similar to the standard Pump Head Curves presented
        # in the Barske paper, which is that the pump head remains relatively
        # constant across all flow rates, but once a critical flow rate is reached it falls off.

        # This will be modelled simply as the shut off head of the pump being constant, but once the critical
        # flow rate is achieved, a parboal will be modelled

        if Q <= Q_max:
            H = H_o
        else:
            H = -100 * ((Q / Q_max) ** 2 - 1) ** 2 + H_o

        if H < 0:
            H = 0

        return H

    def get_exit_condition(
        self, inlet: IncompressibleFluid, N: float, m_dot: float
    ) -> IncompressibleFluid:
        """This function solves for the exit conditions of the pump

        Args:
            inlet (IncompressibleFluid): Inlet Fluid Object
            N (float): Rotational Rate of the Pump
            m_dot (float): Mass Flow Rate Through the Pump

        Returns:
            IncompressibleFluid: Exit Fluid Object
        """
        rho = inlet.get_density()
        Q = m_dot / rho
        p_inlet = inlet.get_pressure()

        H = self.get_head(Q=Q, N=N)

        # We get the pump efficiency

        dp = H * self._g * rho

        outlet = IncompressibleFluid(rho=rho, P=p_inlet + dp)

        return outlet

    def get_torque(
        self,
        inlet: IncompressibleFluid,
        N: float,
    ) -> float:
        """This function solves for the torque produced from the pump

        Args:
            inlet (IncompressibleFluid): Inlet Fluid of the pump
            N (float): Rotational Rate of the Pump (rad/s)

        Returns:
            float: Torque Produced from the Pump (N m)
        """
        # We solve for the shaft power which is constant
        Q_max = self.get_q_max(N=N)
        H_o = self.get_head(Q=0, N=N)

        P_max = inlet.get_density() * self._g * H_o * Q_max

        print(f"Shaft Seed: {N*60/(2*np.pi)}")
        print(f"Actual Head: {H_o} m")

        # We can then evaluate for the torque of the system, by dividing our max power by best effiency point and shaft speed
        T = P_max / (self.get_eta_bep(N=N) * N)

        if N == 0:
            T = 0

        return T
