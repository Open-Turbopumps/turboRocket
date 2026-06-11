"""This file contains the fluidic components of the turbopump"""
    
from turborocket.fluids.fluids import IncompressibleFluid

class LiquidValve:
    """Object Defining the Behaviour of Liquid Propellant Valves"""

    def __init__(
        self,
        cda: float,
        tau: float,
        s_pos_init: float = 0,
        epsilon: float = 100,
    ):
        """Constructor for the liquid propellant valve

        Args:
            cda (float): Flow Area of the valve
            tau (float): Opening/Closing Time of the valve
            s_pos_init (float): Initial Position of the valve
            epsilon (float): Normalisation Parmaeter (Pa). Defaults to 100.
            L_eff (float): Effective flow length within valve for Damping (m). Defaults to 0.015 m
        """

        self._cda = cda
        self._tau = tau
        self._s_pos = s_pos_init
        self._pos = s_pos_init
        self._epsilon = epsilon

        return

    def actuate(self, position: float) -> None:
        """Set's the commanded position of the liquid valve

        Args:
            position (float): Active position of the liquid valve
        """

        self._s_pos = position

        return

    def update_pos(self, dt: float) -> None:
        """This function updates the valve position using a first order model

        Args:
            dt (float): Time Step
        """

        ds_dt = (self._s_pos - self._pos) / self._tau

        self._pos += ds_dt * dt

        return

    def get_mdot(
        self, upstr: IncompressibleFluid, downstr: IncompressibleFluid, dt: float
    ) -> float:
        """This function gets the massflow rate through the valve based on the upstream and downstream conditions

        Args:
            upstr (IncompressibleFluid): Upstream Fluid Object
            downstr (IncompressibleFluid): Downstream Fluid Object
            dt (float): Integration Time Step

        Returns:
            float: Mass Flow Rate (kg/s)
        """

        p1 = upstr.get_pressure()

        p2 = downstr.get_pressure()

        if p1 > p2:
            rho = upstr.get_density()

        else:
            rho = downstr.get_density()

        a = self._cda * self._pos

        dpe = p1 - p2

        # We normalise the flow equation to allow for non-infinite fradients at low dps
        dp_a = ((dpe) ** 2 + self._epsilon**2) ** (1 / 2)

        m_dot = a * ((dpe) / dp_a) * (2 * rho * dp_a) ** (1 / 2)

        return m_dot

    def get_exit_condition(
        self, upstr: IncompressibleFluid, m_dot: float
    ) -> IncompressibleFluid:
        """This function solves for the exit condition of the valve, based on an inlet and a mass flow rate.

        Args:
            upstr (IncompressibleFluid): Upstream Fluid Object of Valve
            m_dot (float): Mass Flow Rate Through the valve (kg/s)

        Returns:
            IncompressibleFluid: Exit Fluid Object of the Valve
        """

        # For this, we need to re-arrange the incompressible fluid flow equation to figure out what our dp is based on the mass flow rate of the valve.
        rho = upstr.get_density()
        p1 = upstr.get_pressure()

        a = self._cda * self._pos

        dp = (m_dot / a) ** 2 * (1 / (2 * rho))

        # We can now evaluate for our exit pressure and create our return object accordingly
        p2 = p1 - dp

        exit = IncompressibleFluid(rho=rho, P=p2)

        return exit

    def get_pos(self) -> float:
        """Function that gets the position of the valve

        Returns:
            float: Position of the valve
        """

        return self._pos

    def get_inertial_param(
        self, upstr: IncompressibleFluid, downstr: IncompressibleFluid
    ) -> float:
        """This function solves for the inertial flow parameter of the valve (used for modelling inertial flows in transient conditions)

        Args:
            upstr (IncompressibleFluid): Upstream Flow Object
            downstr (IncompressibleFluid): Downstream Flow Object

        Returns:
            float: Inertial Parameter Pressure Drop (Pa)
        """

        m_dot = self.get_mdot(upstr=upstr, downstr=downstr)

        a = self._cda * self._pos

        rho = upstr.get_density()

        dp = m_dot**2 / (2 * rho * (a) ** 2)

        return dp

class Cavity:
    """Object Defining the characteristics of liquid incompressible cavities"""

    def __init__(self, fluid: IncompressibleFluid, V: float) -> None:
        """Constructor for the cavity object

        Args:
            fluid (IncompressibleFluid): Initial fluid state within cavity
        """

        self._fluid = fluid
        self._v = V

    def update_pressure(self, m_dot: float) -> None:
        """This function updates the pressure within the cavity, using the bulk modulus approach

        Args:
            m_dot (float): Mass-flow entering/exiting cavity (kg/s)
        """
        B = self._fluid.get_bulk_modululs()
        rho = self._fluid.get_density()

        dv = m_dot / rho

        dp = B * dv / self._v

        p2 = self._fluid.get_pressure() + dp

        self._fluid.set_pressure(P=p2)

        return

    def get_fluid(self) -> IncompressibleFluid:
        """Function that gets the fluid class of the cavity

        Returns:
            IncompressibleFluid: Fluid Subclass of the Cavity
        """

        return self._fluid