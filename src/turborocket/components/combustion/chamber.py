"""This file contains the combustion related objects of the `turborocket` library"""


from turborocket.fluids.fluids import LiquidGasFluid, IdealGas
from turborocket.fluids.isentropic_relations import isen_c_star, isen_static_p_M
from turborocket.solvers.solver import adjoint
import numpy as np
from turborocket.combustion.comb_solver import CombustionCantera

import CoolProp.CoolProp as CP

from enum import Enum

class CombustionChamber:
    """
    Object Defining Combustion Chamber characteristics and behaviours
    """

    def __init__(
        self,
        fuel: LiquidGasFluid | str,
        oxidiser: LiquidGasFluid | str,
        name: str | None = None
    ) -> None:
        """Constructor for the Combustion Chamber Object

        Args:
            fuel (LiquidGasFluid | str): Fluid Object Defining the Fuel used in the combuation chamber.
            oxidiser (LiquidGasFluid | str): Fluid Object Defining the Oxidiser used in the combustion chamber.
            name (str | None, optional): Name of the combustion chamber. Defaults to None.
        """

        if fuel is type(LiquidGasFluid):
            self._fuel = fuel.name
        else:
            self._fuel = fuel
            
        if oxidiser is type(LiquidGasFluid):
            self._ox = oxidiser.name
        else:
            self._ox = oxidiser

        self._name = name

        return

    def setup_cantera(
        self,
        cantera_alias: dict[str, str] | None = None,
        look_up: bool = False,
        look_up_file: str | None = None,
        combustion_file: str | None = None,
    ) -> None:
        """Function that setups up the cantera combustion object for the combustion chamber

        Args:
            cantera_alias (dict[str, str] | None, optional): Dictionary containing aliases for the oxidiser and fuel used in Cantera.
            look_up (bool, optional): Look Up Flag for whether interpolation based approach is used. Defaults to False.
            look_up_file (str | None, optional): Name of the Lookup file to be loaded for the interpolation. Defaults to None.
            combustion_file (str | None, optional): Mechanism File Used for Combustion Modelling. Defaults to None.
        """
        
        # We check to see if a cantera alias exists or not
        if cantera_alias is not None:
            
            if "oxidiser" in cantera_alias:
                ox_name = cantera_alias["oxidiser"]
                
            if "fuel" in cantera_alias:
                fu_name = cantera_alias["fuel"]
        
        # Otherwise, we take the names of the propellant objects
        else:
            ox_name = self._ox
            fu_name = self._fuel

        self._comb = CombustionCantera(
            fuel=fu_name,
            oxidiser=ox_name,
            species_file=combustion_file,
            look_up=look_up,
            look_up_file=look_up_file,
        )

        return

    def ideal_area_rat(self,
                 comb_gas: IdealGas,
                 p_ambient: float) -> float:
        """This function evaluates for the ideal nozzle area ratio for the chamber

        Args:
            gas (IdealGas): Gas Component for the chamber conditions
            p_ambient (float): Ambient Expansion Pressure (Bar)

        Returns:
            float: Required Area ratio for selected expansion
        """
        
        # We extract the specific heat ratio and chamber pressure of the fluid.
        gamma = comb_gas.gamma
        p_0 = comb_gas.p
        
        A_rat = (
            ((gamma - 1)/2)**(1/2) 
            * (2 / (gamma + 1)) ** ((gamma + 1)/(2 * (gamma - 1)))
            * (p_ambient / p_0) ** (-1 / gamma)
            * (1 - (p_ambient / p_0) ** ((gamma - 1) / gamma)) ** (- (1/2))
        )
        
        return A_rat

    def get_cf(self, 
               gas: IdealGas,
               p_ambient: float = 1e5
               ) -> float:
        """This function gets the Thrust Coefficient for the Chamber

        Args:
            gas (IdealGas): Combustion Chamber Ideal Gas Object
            p_ambient (float, optional): Ambient Pressure (Pa). Defaults to 1 Bar.

        Returns:
            float: Thrust Coefficient
        """
        # We get the specific heat ratio and chamber pressure
        gamma = gas.gamma
        p_cc = gas.p
        
        # We get the exit pressure based on the stagnation pressure
        p_exit = self.get_p_exit(gas = gas)

        c_f = (
            (((2 * gamma**2)/(gamma - 1)) * 
              (2 / (gamma + 1))**((gamma + 1)/(gamma - 1)) * 
              (1 - (p_exit / p_cc)**((gamma - 1)/ gamma)) ) ** (1/2) +
              (self._a_e / self._a_t) * ((p_exit - p_ambient)/p_cc)  
        )

        return c_f
    

    def area_error(self, M: float, gamma: float):
        """Error Function for the Nozzle Expansion Ratio Relationship

        Args:
            M (float): Mach number at the Exit of the Nozzle
            gamma (float): Specific Heat Ratio of the Gas
        """

        rhs = (1 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        )

        error = rhs - (self._a_e / self._a_t)

        return error
    
    def solve_M_exit(self,
                     gas: IdealGas):
        """This function solves for the Supersonic Mach Number solution

        Args:
            gas (IdealGas): Combustion Chamber Ideal Gas Object

        Returns:
            float: Supersonic Mach Number [ N.D. ]
        """
        gamma = gas.gamma
        
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
    
    def get_p_exit(self,
                gas: IdealGas) -> float:
        """This function solves for the exit pressure of the chamber

        Args:
            gas (IdealGas): Combustion Gas Object Inside the Chamber

        Returns:
            float: Exit Pressure for the Combustion Chamber
        """
        
        # First we get the supersonic Mach Number at the exit of the chamber
        M_e = self.solve_M_exit(gas = gas)
        
        # We get the chamber conditions
        p_exit = isen_static_p_M(fluid = gas, p_0 = gas.p, M = M_e)
        
        # We can now report the exit pressure
        return p_exit
    
    def get_thrust(self,
                   gas: IdealGas,
                   p_ambient: float = 1e5):
        """This function solves for the thrust of the chamber

        Args:
            gas (IdealGas): Combustion Chamber Gas Condition
            P_a (float, optional): Ambient Pressure (Pa). Defaults to 1 Bar.
        """
        
        # We get the chamber pressure
        p_cc = gas.p
        
        c_f = self.get_cf(gas = gas, p_ambient = p_ambient)

        F = c_f * p_cc * self._a_t
        
        return F
    
    def get_condition(
        self,
        MR: float,
        p_cc: float,
        eta_c: float,
    ) -> dict:
        """This function returns a dictionary of the key parameters for the system at a given condition

        Args:
            MR (float): Mixture Ratio of Propellants entering the combustion chamber
            p_cc (float): Combustion Chamber Pressure for the system
            eta_c (float): Characteristic Velocity Efficiency.

        Returns:
            dict: Dictionary of Key Parameters for the Combustion [m_dot, T_o, F, gas]
        """
        # We evaluate for the condition in the chamber
        gas = self._comb.get_thermo_prop(Pcc=p_cc, MR=MR)

        # We get the c_star
        c_star = isen_c_star(fluid= gas, t_0=gas.t) * eta_c
        
        # From this, we can evaluate for the mass-flow rate through the combustion chamber
        m_dot = p_cc * self._a_t / c_star

        # We can evalaute for the stagnation temperature of the gas, accounting for stagnation temperature.
        t_o = gas.t * eta_c**2
        
        # We get the thrust of the system
        F = self.get_thrust(gas = gas)

        # We create our Dict and finally return it

        dic = {
            "m_dot": m_dot,
            "T_o": t_o,
            "F": F,
            "gas": gas
        }

        return dic
    
    def size_chamber(self, 
                     m_dot: float, 
                     MR: float,
                     p_cc: float,
                     eta_c: float = 1, 
                     p_exit: float = 1e5) -> dict:
        """This function sizes the Combustion Chamber

        Args:
            m_dot (float): Mass Flow Rate Through Combustion Chamber [ kg / s ]
            MR (float): Mixture Ratio of the combustion chamber [ N.D. ]
            p_cc (float): Combustion Chamber Pressure [ Pa ]
            eta_c (float, optional): C* efficiency of the combustion [ % ]. Defaults to 0.8.
            p_exit (float, optional): Nozzle Exit Pressure [ Pa ]. Defaults to 1 Bar (Atmosphere).

        Returns:
            dict: Dictionary containing the geometric parameters and conditions of the chamber.
        """
        
        # Based on the combustion conditions, we can evaluate for the combustion conditions
        comb_gas = self._comb.get_thermo_prop(Pcc = p_cc, MR = MR)
        
        # We can now evaluate for the characteristic velociy of the gas
        c_star = isen_c_star(fluid = comb_gas, t_0 = comb_gas.t) * eta_c

        # We can now size for the throat of the chamber
        self._a_t = c_star * m_dot / p_cc
        
        # We can now size for the exit area of the chamber, based on the ambient pressure
        a_rat = self.ideal_area_rat(comb_gas = comb_gas, p_ambient = p_exit)
        
        # From this, we can evalaute for the exit area of the nozzle of the chamber
        self._a_e = self._a_t * a_rat

        # We then get the condition of the chamber
        dic = self.get_condition(
            MR = MR,
            p_cc = p_cc,
            eta_c = eta_c
        )
        
        # We additionally add the exit and throat diameter
        dic.update(self.get_geometry())

        return dic

    def get_geometry(self) -> dict:
        """Function that gets the injector geometry for the user

        Returns:
            dict: Dictionary Describing key parameters
        """

        dic = {
            "a_t": self._a_t,
            "a_e": self._a_e
        }

        return dic
    
    def set_geometry(self,
                     a_ox: float | None = None,
                     a_fu: float | None = None,
                     a_cc: float | None = None,
                     a_e: float | None = None,
                     ) -> None:
        """Function that sets the geometric dimensions of the engine

        Args:
            a_ox (float | None, optional): Oxidiser Injector Orifice Area [ m^2 ]. Defaults to None.
            a_fu (float | None, optional): Fuel Injector Orifice Area [ m^2 ]. Defaults to None.
            a_cc (float | None, optional): Main Combustion Chamber Throat Area [ m^2 ]. Defaults to None.
            a_e (float | None, optional): Main COmbustion Chamber Exit Area [ m^2 ]. Defaults None.
        """
        
        if a_ox:
            self._a_ox = a_ox
            
        if a_fu:
            self._a_fu = a_fu
            
        if a_cc:
            self._a_cc = a_cc
            
        if a_e:
            self._a_e = a_e

    # def set_pcc_transient(self, P_cc_transient: float) -> None:
    #     """This function sets the transient set pressure used for transient modelling of the combustion chamber

    #     Args:
    #         P_cc_transient (float): Transient Chamber Pressure (Pa)
    #     """

    #     self._pcc_transient = P_cc_transient

    #     return

    # def get_pcc_transient(self) -> float:
    #     """This is a "getter" function for the transient set pressure of the combustion chamber

    #     Returns:
    #         float: Transient Chamber Pressure (Pa)
    #     """

    #     return self._pcc_transient

    # def set_l_star(self, L_star: float) -> None:
    #     """This function sets L* of the chamber

    #     Args:
    #         L_star (float): L_star of the combustion chamber (m)
    #     """

    #     self._l_star = L_star

    #     self._v_cc = self._a_cc * self._l_star

    #     return


    # def get_density(self, Pcc: float, MR: float, eta_c: float) -> float:
    #     """This function gets the Combustion gas density
    #     - Assumption: Ideal Gas

    #     Args:
    #         Pcc (float): Chamber Pressure (Pa)
    #         MR (float): Mixture Ratio
    #         eta_c (float): C* Efficiency of the Gas

    #     Returns:
    #         float: Density of the Gas (kg/s)
    #     """
    #     # Getting combustion gas properties
    #     gas = self._comb.get_thermo_prop(Pcc=Pcc, MR=MR)

    #     R = gas.get_R()
    #     T = gas.get_temperature() * eta_c**2

    #     rho = Pcc / (R * T)

    #     return rho

    # def transient_engine_nofire(
    #     self,
    #     ox_in: IncompressibleFluid,
    #     fu_in: IncompressibleFluid,
    #     m_dot_ox: float,
    #     m_dot_fu: float,
    #     eta_c: float = 0.85,
    # ) -> dict:
    #     """Function for the case where the engine hasnt lit yet.

    #     Args:
    #         ox_in (IncompressibleFluid): _description_
    #         fu_in (IncompressibleFluid): _description_
    #         m_dot_ox (float): _description_
    #         m_dot_fu (float): _description_
    #         eta_c (float, optional): _description_. Defaults to 0.85.

    #     Returns:
    #         dict: Dictionary
    #     """

    #     dp_dt = 0
    #     self._pcc_transient = 1e5

    #     dic = {
    #         "dp_dt": dp_dt,
    #         "P_cc": self._pcc_transient,
    #         "MR": 0,
    #         "T_o": 0,
    #         "Cp": 1005,
    #         "R": 287,
    #         "gamma": 1.4,
    #         "ox_stiffness": 0,
    #         "fu_stiffness": 0,
    #         "m_dot_t": 0,
    #         "m_dot_o": m_dot_ox,
    #         "m_dot_f": m_dot_fu,
    #     }

    #     comb_gas = IdealGas(
    #         p=dic["P_cc"],
    #         t=dic["T_o"],
    #         cp=dic["Cp"],
    #         gamma=dic["gamma"],
    #         R=dic["R"],
    #     )

    #     dic["gas_obj"] = comb_gas

    #     return dic

    # def transient_startup(
    #     self,
    #     ox_in: IncompressibleFluid,
    #     fu_in: IncompressibleFluid,
    #     m_dot_ox: float,
    #     m_dot_fu: float,
    #     eta_c: float = 0.85,
    # ) -> dict:

    #     MR_current = m_dot_ox / m_dot_fu

    #     # We need to solve for the combustion density at the current time
    #     rho_c = self.get_density(Pcc=self._pcc_transient, MR=MR_current, eta_c=eta_c)

    #     # W need to get the c_star of the current condition
    #     c_star = self.get_c_star(Pcc=self._pcc_transient, MR=MR_current, eta_c=eta_c)

    #     # Finally we can solve for the pressure gradient
    #     dp_dt = (self._pcc_transient / (rho_c * self._v_cc)) * (
    #         m_dot_ox + m_dot_fu - (self._pcc_transient * self._a_cc) / c_star
    #     )

    #     # We need to now evaluate for the system performance paramets
    #     gas = self._comb.get_thermo_prop(Pcc=self._pcc_transient, MR=MR_current)

    #     dic = {
    #         "dp_dt": dp_dt,
    #         "P_cc": self._pcc_transient,
    #         "MR": MR_current,
    #         "T_o": gas.get_temperature() * eta_c**2,
    #         "Cp": gas.get_cp(),
    #         "R": gas.get_R(),
    #         "gamma": gas.get_gamma(),
    #         "ox_stiffness": (ox_in.get_pressure() - self._pcc_transient)
    #         / self._pcc_transient,
    #         "fu_stiffness": (fu_in.get_pressure() - self._pcc_transient)
    #         / self._pcc_transient,
    #         "m_dot_t": m_dot_fu + m_dot_ox,
    #         "m_dot_o": m_dot_ox,
    #         "m_dot_f": m_dot_fu,
    #     }

    #     comb_gas = IdealGas(
    #         p=dic["P_cc"],
    #         t=dic["T_o"],
    #         cp=dic["Cp"],
    #         gamma=dic["gamma"],
    #         R=dic["R"],
    #     )

    #     dic["gas_obj"] = comb_gas

    #     # We store the last MR for locking
    #     self._MR_transient = MR_current

    #     return dic

    # def transient_shutdown(
    #     self,
    #     ox_in: IncompressibleFluid,
    #     fu_in: IncompressibleFluid,
    #     m_dot_ox: float,
    #     m_dot_fu: float,
    #     eta_c: float = 0.85,
    # ) -> dict:
    #     """Function characterinsing the shutdown transient

    #     Args:
    #         ox_in (IncompressibleFluid): Inlet Oxidiser Object
    #         fu_in (IncompressibleFluid): Fuel Injector Object
    #         m_dot_ox (float): Oxidiser Mass Flow
    #         m_dot_fu (float): Fuel Mass Flow
    #         eta_c (float, optional): Combustion Efficiency. Defaults to 0.85.

    #     Returns:
    #         dict: _description_
    #     """
    #     c_star = self.get_c_star(
    #         Pcc=self._pcc_transient, MR=self._MR_transient, eta_c=eta_c
    #     )

    #     rho_c = self.get_density(
    #         Pcc=self._pcc_transient, MR=self._MR_transient, eta_c=eta_c
    #     )

    #     dp_dt = (self._pcc_transient / (rho_c * self._v_cc)) * (
    #         -(self._pcc_transient * self._a_cc) / c_star
    #     )
    #     gas = self._comb.get_thermo_prop(Pcc=self._pcc_transient, MR=self._MR_transient)

    #     dic = {
    #         "dp_dt": dp_dt,
    #         "P_cc": self._pcc_transient,
    #         "MR": self._MR_transient,
    #         "T_o": gas.get_temperature() * eta_c**2,
    #         "Cp": gas.get_cp(),
    #         "R": gas.get_R(),
    #         "gamma": gas.get_gamma(),
    #         "ox_stiffness": (ox_in.get_pressure() - self._pcc_transient)
    #         / self._pcc_transient,
    #         "fu_stiffness": (fu_in.get_pressure() - self._pcc_transient)
    #         / self._pcc_transient,
    #         "m_dot_t": m_dot_fu + m_dot_ox,
    #         "m_dot_o": m_dot_ox,
    #         "m_dot_f": m_dot_fu,
    #     }

    #     comb_gas = IdealGas(
    #         p=dic["P_cc"],
    #         t=dic["T_o"],
    #         cp=dic["Cp"],
    #         gamma=dic["gamma"],
    #         R=dic["R"],
    #     )

    #     dic["gas_obj"] = comb_gas

    #     return dic

    # def transient_time_step(
    #     self,
    #     ox_in: IncompressibleFluid,
    #     fu_in: IncompressibleFluid,
    #     eta_c: float = 0.85,
    # ) -> dict:
    #     """Conducts a Transient Time Step for the Combustion Chamber Performance

    #     Args:
    #         ox_in (IncompressibleFluid): _description_
    #         fu_in (IncompressibleFluid): _description_
    #         dt (float): _description_
    #         eta_c (float): C* efficiency of the combustion. Defaults to 0.85.

    #     Returns:
    #         dict: Dictionary of Performance metrics of the combustion
    #     """
    #     m_dot_ox, m_dot_fu = self.get_injector_flow(
    #         ox_in=ox_in, fu_in=fu_in, Pcc=self._pcc_transient
    #     )

    #     # We check for the condition for engine ignition, otherwise we leave the system as is.

    #     if (m_dot_ox == 0) or (m_dot_fu == 0):
    #         # Engine is not lit, hence we dont consider mass-flows, we just deplete the engine until it goes back to ambient

    #         # We check if the chamber is at ambient conditions
    #         if self._pcc_transient <= 1e5:
    #             dic = self.transient_engine_nofire(
    #                 ox_in=ox_in,
    #                 fu_in=fu_in,
    #                 m_dot_fu=m_dot_fu,
    #                 m_dot_ox=m_dot_ox,
    #                 eta_c=eta_c,
    #             )
    #         else:
    #             # We deplete the chamber using the c* value
    #             dic = self.transient_shutdown(
    #                 ox_in=ox_in,
    #                 fu_in=fu_in,
    #                 m_dot_fu=m_dot_fu,
    #                 m_dot_ox=m_dot_ox,
    #                 eta_c=eta_c,
    #             )

    #     else:
    #         # We need to check if our mass flow rates are enough for ignition - arbitrary criterion of 5%
    #         m_dot_t = m_dot_fu + m_dot_ox

    #         if m_dot_t < self._m_dot * 0.05:
    #             dic = self.transient_engine_nofire(
    #                 ox_in=ox_in,
    #                 fu_in=fu_in,
    #                 m_dot_fu=m_dot_fu,
    #                 m_dot_ox=m_dot_ox,
    #                 eta_c=eta_c,
    #             )

    #         else:
    #             dic = self.transient_startup(
    #                 ox_in=ox_in,
    #                 fu_in=fu_in,
    #                 m_dot_fu=m_dot_fu,
    #                 m_dot_ox=m_dot_ox,
    #                 eta_c=eta_c,
    #             )

    #     return dic