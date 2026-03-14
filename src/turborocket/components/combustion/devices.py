"""File that contains objects representing "combustion devices"""

from turborocket.components.combustion.injector import InjectorElement, InjectorMethods
from turborocket.components.combustion.chamber import CombustionChamber

from turborocket.solvers.solver import adjoint

from turborocket.fluids.fluids import LiquidGasFluid, IdealGas

from enum import Enum

class PropellantTypes(Enum):
    """Propellant Type Enums"""
    
    oxidiser = "Oxidiser"
    fuel = "Fuel"

class InjectorChamberAssembly():
    """Object representing a combustion chamber and injector assembly"""
    
    def __init__(self,
                 name: str,
                 ox_injector: InjectorElement | None = None,
                 fu_injector: InjectorElement | None = None,
                 chamber: CombustionChamber | None = None
                 ) -> None:
        """Constructor for the Injector Chamber Assembly Component. The user can either provide injector or combustion chamber geometries, or sizes these dynamically.

        Args:
            name (str): Name of the Injector Chamber Assembly
            ox_injector (InjectorElement | None, optional): Oxidiser Injector Element. Defaults to None.
            fu_injector (InjectorElement, optional): Fuel Injector Element. Defaults to None.
            chamber (CombustionChamber | None, optional): Combustion Chamber Element. Defaults to None.
        """
        
        self._name = name
        self._ox_injector = ox_injector
        self._fu_injector = fu_injector
        self._chamber = chamber
        
        return
    
    def setup_cantera(
        self,
        cantera_alias: dict[str, str],
        look_up: bool = False,
        look_up_file: str | None = None,
        combustion_file: str | None = None,
    ) -> None:
        """Function to setup the Cantera Combustion Solver

        Args:
            cantera_alias (dict[str, str] | None, optional): _description_. Defaults to None.
            look_up (bool, optional): _description_. Defaults to False.
            look_up_file (str | None, optional): _description_. Defaults to None.
            combustion_file (str | None, optional): _description_. Defaults to None.
        """
        # We firstly instantiate our chamber object
        chamber = CombustionChamber(fuel = cantera_alias["fuel"], oxidiser = cantera_alias["oxidiser"])
        
        chamber.setup_cantera(cantera_alias=cantera_alias,
                              look_up=look_up,
                              look_up_file=look_up_file,
                              combustion_file=combustion_file)
        
        self._chamber = chamber
        
        return
    
    def size_chamber(self,
                     MR: float,
                     fuel: LiquidGasFluid | str,
                     oxidiser: LiquidGasFluid | str,
                     p_cc: float,
                     m_dot: float,
                     eta_c: float = 0.95,
                     p_exit: float = 1e5) -> LiquidGasFluid:
        """Function that sizes the combustion chamber object

        Args:
            MR (float): Mixture Ratio of the propellant in the combustion chamber [ N.D. ]
            fuel (LiquidGasFluid | str): Fuel Object/String
            oxidiser (LiquidGasFluid | str): Oxidiser Object/String
            p_cc (float): Pressure inside the combustion chamber [ Pa ]
            m_dot (float): Mass Flow Rate through the combustion chamber [ kg / s ]
            p_exit (float, optional): Nozzle Exit Pressure [ Pa ]. Defaults to 1 Bar.

        Returns:
            LiquidGasFluid: Combustion Gas Object 
        """
        # We then size the chamber
        _ = self._chamber.size_chamber(m_dot = m_dot, MR = MR, p_cc = p_cc, eta_c = eta_c, p_exit = p_exit)
        
        
        return
    
    def size_injector(self,
                      inlet: LiquidGasFluid,
                      p_cc: float,
                      m_dot: float,
                      prop_flag: PropellantTypes,
                      inj_type: InjectorMethods
                      ) -> None:
        """Function that sizes the oxidiser injector element

        Args:
            inlet (LiquidGasFluid): Propellant Object at inlet of fluid.
            p_cc (float): Combustion Chamber Pressure [ Pa ]
            m_dot (float): Mass Flow Rate through the oxidiser element [ kg / s ]
            prop_flag (PropellantTypes): Flag of Propellant Types.
            inj_type (InjectorMethods): Different Injector Methods.
        """
        
        # We firstly instantiate our injector element with a cda of 1.
        injector = InjectorElement(cda = 1, inj_method=inj_type)
        
        # We define our downstream section of the injector at the chamber conditions
        gas = IdealGas(p = p_cc, t = 295, name="gas", cp=2000, R=800)
        
        # We then evaluate for the unit cda mass-flow rate of the injector
        m_dot_n = injector.m_dot(inlet = inlet, outlet=gas)
        
        # We can then evaluate for the injector area
        cda = m_dot / m_dot_n
        
        # We then set this area
        injector.cda = cda
        
        # We then check the flag of the propellant
        if prop_flag == PropellantTypes.oxidiser:
            self._ox_injector = injector
            
        elif prop_flag == PropellantTypes.fuel:
            self._fu_injector = injector
            
    def evaluate_condition(self,
                           inlet_fuel: LiquidGasFluid,
                           inlet_oxidiser: LiquidGasFluid,
                           eta_c: float = 0.95
                           ) -> dict:
        """Evaluates for the condition within the combustion chamber assembly.

        Args:
            inlet_fuel (LiquidGasFluid): Inlet Oxidiser State
            inlet_oxidiser (LiquidGasFluid): Inlet Fuel State
            eta_c (float, optional): C* efficiency [ % ]

        Returns:
            dict: Dictionary of combustion chamber and injector properties.
        """

        # We firsly get the pressures of both the fuel and oxidiser
        p_f = inlet_fuel.p
        p_o = inlet_fuel.p
        
        # We get the minimum of these two, and subtract from these a small ammount
        p_cc_guess = min(p_f, p_o) * 2/3
        
        # We then define an auxilary function for calculating the error in the mass-flows
        def error_p_cc(p_cc: float, 
                       inlet_fu_test: LiquidGasFluid, 
                       inlet_ox_test: LiquidGasFluid
                       ) -> float:
            
            # We define an arbitrary combustion chamber object

            down = IdealGas(p = p_cc, t = 295, name="gas", cp=2000, R=800)
        
            # We firstly evaluate for the mass flows on both oxidiser and fuel

            m_dot_o = self._ox_injector.m_dot(inlet=inlet_ox_test, outlet=down)
            m_dot_f = self._fu_injector.m_dot(inlet=inlet_fu_test, outlet=down)
            
            # We get the total mass flow rate of the injector
            m_dot_t_i = m_dot_o + m_dot_f
            
            print(f"ox: {m_dot_o}")
            print(f"fu: {m_dot_f}")
            # We calculate the MR
            MR = m_dot_o / m_dot_f
            
            # We then solve for combustion chamber conditions
            comb_cond = self._chamber.get_condition(MR = MR, p_cc = p_cc, eta_c = eta_c)
            
            # We then get the mass-flow rate of the chamber 
            m_dot_t_c = comb_cond["m_dot"]
            
            error = abs(m_dot_t_c - m_dot_t_i)
            
            return error
        
        # We then run an optimisation using the adjoint function we have prepared.
        p_cc = adjoint(
            func = error_p_cc,
            x_guess = p_cc_guess,
            dx = 0.1e5,
            n = 500,
            relax = 1,
            target=0,
            params=[inlet_fuel, inlet_oxidiser]
        )
        
        # Once we have the chamber pressure, we can evaluate for the final conditions
        down = IdealGas(p = p_cc, t = 295, name="gas", cp=2000, R=800)
    
        # We firstly evaluate for the mass flows on both oxidiser and fuel
        m_dot_o = self._ox_injector.m_dot(inlet=inlet_oxidiser, outlet=down)
        m_dot_f = self._fu_injector.m_dot(inlet=inlet_fuel, outlet=down)
        
        # We calculate the MR
        MR = m_dot_o / m_dot_f
        
        # We then solve for combustion chamber conditions
        dic = self._chamber.get_condition(MR = MR, p_cc = p_cc, eta_c = eta_c)
        
        # We append our oxidiser and fuel mass flow, along with stiffness
        dic["m_dot_ox"] = m_dot_o
        dic["m_dot_fu"] = m_dot_f
        dic["dp_ox/p_cc"] = (inlet_oxidiser.p - p_cc) / p_cc
        dic["dp_fu/p_cc"] = (inlet_fuel.p - p_cc) / p_cc
        
        return dic
        
    def geometry(self) -> dict:
        """Function that returns a dictionary of geometries to the user

        Returns:
            dict: Dictionary of the injector chamber geometries.
        """
        
        dic = {
            "cda_ox": self._ox_injector.cda,
            "cda_fu": self._fu_injector.cda,
            "a_t": self._chamber._a_t,
            "a_e": self._chamber._a_e
        }
        
        return dic
        


class GasGenerator(InjectorChamberAssembly):
    """This Object Defines the Characteristics of the Gas Generator Object"""

    def __init__(self,
                name: str,
                ox_injector: InjectorElement | None = None,
                fu_injector: InjectorElement | None = None,
                chamber: CombustionChamber | None = None
                ) -> None:
        """Constructor for the Gas Generator Object"""
        super().__init__(name, ox_injector, fu_injector, chamber)


class MainEngine(CombustionChamber):
    """This Object Defines the Characteristics of the Main Engine Object"""

    def __init__(self,
                name: str,
                ox_injector: InjectorElement | None = None,
                fu_injector: InjectorElement | None = None,
                chamber: CombustionChamber | None = None
                ) -> None:
        """Constructor for the Main Engine Object"""
        super().__init__(name, ox_injector, fu_injector, chamber)