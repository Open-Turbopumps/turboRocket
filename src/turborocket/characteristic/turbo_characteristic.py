"""
This file contains the classes used for the generation of the Turbopump Cycle Characteristics
"""

from turborocket.components.start_up import (
    GasGenerator,
    Pump,
    Turbine,
    LiquidValve,
    MechanicalLosses,
)
from turborocket.fluids.fluids import IncompressibleFluid, IdealGas
from turborocket.solvers.solver import adjoint

import numpy as np


class TurboCharacteristics:
    """
    Object Representing the Turbopump + Gas Generator Cycle

    The basic idea behind this object is we supply a set of individual components describing the turbopumps
    performance, from which off-nominal performance evaluations can be performed based on these component
    characteristics
    """

    def __init__(
        self,
        pump: Pump,
        turbine: Turbine,
        gas_generator: GasGenerator,
        valve: LiquidValve | None = None,
        loss: MechanicalLosses | None = None,
    ) -> None:
        """Constructor for the Turbopump Cycle Object

        Args:
            pump (Pump): Pump Object of the Turbopump
            turbine (Turbine): Turbine Object of the Turbopump
            gas_generator (GasGenerator): Gas Generator Object of the Turbopump
            valve (Liquid Valve | None, optional): Valve Object for throttling the gas generator flow. Defaults to None.
            loss (MechanicalLosses | None, optional): Mechanical Loss Object for Power Transmisition. Defaults to None.

        Optional:
            valve (LiquidValve): Throttling valve between pump outlet and gas generator Inlet
        """

        self._pump = pump
        self._turbine = turbine
        self._gg = gas_generator
        self._valve = valve
        self._loss = loss

        return

    def gg_m_dot_error(
        self,
        m_dot_valve: float,
        pump_exit: IncompressibleFluid,
        ox_inlet: IncompressibleFluid,
    ) -> float:
        """This is an error function used to solve for the mass flow rate passing through the gas generator.

        Args:
            m_dot_valve (float): Flow through the IPA valve (kg/s)
            pump_exit (IncompressibleFluid): Pump Exit Condition Object
            ox_inlet (IncompressibleFluid): Oxidiser Inlet Condition Object (for the GG)

        Returns:
            float: Error in mass flow rates
        """

        # We need to firstly evaluate for the valve exit condition based on the mass flow rate and pump inlet condition
        valve_exit = self._valve.get_exit_condition(m_dot=m_dot_valve, upstr=pump_exit)

        # We can now solve for the gas generator conditions
        dic_gg = self._gg.solve_perturb_ss(fu_in=valve_exit, ox_in=ox_inlet)

        # We can extract the mass flow rate of IPA through the gas generator, and then solve for the error in the system
        error = m_dot_valve - dic_gg["m_dot_f"]

        return error

    def get_gg_inlet(
        self,
        m_dot_t: float,
        pump_exit: IncompressibleFluid,
        ox_inlet: IncompressibleFluid,
    ) -> IncompressibleFluid:
        """This function solves for the Gas Generator Inlet Condition, based on the pump exit condition and valve position

        Args:
            m_dot_t (float): Total Mass Flow Rate Being Pumped
            pump_exit (IncompressibleFluid): Pump Exit Condition
            ox_inlet (IncompressibleFluid): Oxidiser Inlet Condition

        Returns:
            float: Gas Generator Mass Flow Rate (kg/s)
        """
        # Firstly we will check if the cycle has a valve or not:
        if self._valve is not None:
            m_dot_guess = m_dot_t * 0.05
            m_dot_dx = m_dot_t * 0.01

            # For this, we will simply setup and adjoint based optimisation to figure out what the mass flow rate through the gas generator would be to meet the cycle requriements.
            m_dot_gg = adjoint(
                func=self.gg_m_dot_error,
                x_guess=m_dot_guess,
                dx=m_dot_dx,
                n=500,
                relax=1,
                target=0,
                params=[pump_exit, ox_inlet],
            )

            # Now that we know the mass flow rate, we can solve for the gas generator inlet condition

            gg_inlet = self._valve.get_exit_condition(upstr=pump_exit, m_dot=m_dot_gg)

        else:
            # If there is no valve, the exit condition of the pump is the same as the inlet condition of the gas generator
            gg_inlet = pump_exit

        return gg_inlet

    def get_torque_error(
        self,
        N: float,
        m_dot_t: float,
        fu_inlet: IncompressibleFluid,
        ox_inlet: IncompressibleFluid,
        p_exit: float = 1e5,
    ) -> float:
        """This function evaluates for the error in power generation for the gas generator turbopump

        Args:
            N (float): Shaft Speed (rad/s)
            m_dot_t (float): Total Mass Flow Rate going Through the Pump (kg/s)
            fu_inlet (IncompressibleFluid): Inlet Condition of the Fuel Pump
            ox_inlet (IncompressibleFluid): Oxidiser Delivery Condition to GG

        Optional:
            p_exit (float): Exit Pressure of Turbine Stage. Defaults to 1e5 (Pa)

        Returns:
            float: Error in Torques produced by the system
        """
        # First thing, we need to evaluate for is what the quantity of the torque and dp
        # that is produced by the pump

        pump_exit = self._pump.get_exit_condition(inlet=fu_inlet, N=N, m_dot=m_dot_t)
        T_pump = self._pump.get_torque(inlet=fu_inlet, N=N, m_dot=m_dot_t)

        # We must now evaluate for what the condition is at the injector inlet.
        gg_fu_inlet = self.get_gg_inlet(
            m_dot_t=m_dot_t, pump_exit=pump_exit, ox_inlet=ox_inlet
        )

        # We can now solve for the final gas generator conditions
        gg_dic = self._gg.solve_perturb_ss(
            fu_in=gg_fu_inlet, ox_in=ox_inlet, obj_flag=True
        )

        m_dot_gg = gg_dic["m_dot_t"]

        T_turbine = m_dot_gg * self._turbine.get_torque(
            combustion_gas=gg_dic["gas_obj"], P_exit=p_exit, N=N
        )

        if self._loss is not None:
            T_bearing = self._loss.get_torque()

        else:
            T_bearing = 0

        # We can now solve for the error in torques
        error = T_turbine - T_pump - T_bearing

        return error

    def get_condition(
        self,
        N: float,
        fu_inlet: IncompressibleFluid,
        ox_inlet: IncompressibleFluid,
        m_dot_t: float,
        P_exit: float = 1e5,
    ) -> dict[str, float | IdealGas | IncompressibleFluid]:
        """This function generates a dictionary describing the current turbopump state.

        Args:
            fu_inlet (IncompressibleFluid): Fuel Inlet Object to Turbopump Assembly (Pumped)
            ox_inlet (incompressibleFluid): Oxidiser Inlet Object to Gas Generator (Not Pumped)
            m_dot_t (float): Total Mass Flow Rate being Pumped by the Pump

        Optional:
            P_exit (float): Exit Expansion Pressure for the Turbopump assembly (Pa)


        Returns:
            dict[str, float | IdealGas | IncompressibleFluid]: Dictionary of Parameters Describing the Turbopump Cycle
        """

        # First thing  we do is solve for the exit condition of the pump
        pump_exit = self._pump.get_exit_condition(inlet=fu_inlet, N=N, m_dot=m_dot_t)

        gg_fu_inlet = self.get_gg_inlet(
            m_dot_t=m_dot_t, pump_exit=pump_exit, ox_inlet=ox_inlet
        )

        # We can now solve for the final gas generator conditions
        gg_dic = self._gg.solve_perturb_ss(
            fu_in=gg_fu_inlet, ox_in=ox_inlet, obj_flag=True
        )

        T_turbine = gg_dic["m_dot_t"] * self._turbine.get_torque(
            combustion_gas=gg_dic["gas_obj"], P_exit=P_exit, N=N
        )

        eta_turbine = self._turbine.get_efficiency(
            combustion_gas=gg_dic["gas_obj"], N=N, p_exit=P_exit
        )

        gg_dic["N"] = N
        gg_dic["Power"] = T_turbine * N
        gg_dic["pump_exit"] = pump_exit
        gg_dic["gg_fuel_inlet"] = gg_fu_inlet
        gg_dic["eta"] = (
            (pump_exit.get_pressure() - fu_inlet.get_pressure())
            * (m_dot_t / fu_inlet.get_density())
            / (T_turbine * N / eta_turbine)
        )

        return gg_dic

    def solve_condition(
        self,
        fu_inlet: IncompressibleFluid,
        ox_inlet: IncompressibleFluid,
        m_dot_t: float,
        P_exit: float = 1e5,
    ) -> dict[str, float | IdealGas | IncompressibleFluid]:
        """Function that evaluates for the Operating Point of The Turbopump

        Args:
            fu_inlet (IncompressibleFluid): Fuel Inlet Object to Turbopump Assembly (Pumped)
            ox_inlet (incompressibleFluid): Oxidiser Inlet Object to Gas Generator (Not Pumped)
            m_dot_t (float): Total Mass Flow Rate being Pumped by the Pump

        Optional:
            N_guess (float): Intial Shaft Speed Guess (rad/s)
            P_exit (float): Exit Expansion Pressure for the Turbopump assembly (Pa)

        Returns:
            dict[str, float]: Dictionary of Performancers of the Turbopump Assembly
        """

        # We run an adjoint to solve for the shaft speed of the system
        N = adjoint(
            func=self.get_torque_error,
            x_guess=self._pump._N_nom,
            dx=100,
            n=500,
            relax=0.5,
            target=0,
            params=[m_dot_t, fu_inlet, ox_inlet, P_exit],
        )

        # Now that we know the shaft speed, we can solve for all our key parameters
        dic = self.get_condition(
            N=N, fu_inlet=fu_inlet, ox_inlet=ox_inlet, m_dot_t=m_dot_t, P_exit=P_exit
        )

        dic["error"] = self.get_torque_error(
            N=N, m_dot_t=m_dot_t, fu_inlet=fu_inlet, ox_inlet=ox_inlet, p_exit=P_exit
        )

        return dic
