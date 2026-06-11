from turborocket.profiling.Supersonic.circular import inv_M_star
from turborocket.fluids.fluid_support import coolprop_input, ideal_input, ideal_thermodynamic_parameters, ideal_gas_law_parameters

import numpy as np

from abc import ABC, abstractmethod

from CoolProp import AbstractState
import CoolProp.CoolProp as CP

from typing import Self

# We can create a generic class characterising an ideal gas

class LiquidGasFluid(ABC):
    """Abstract Base Class for Liquid/Gas Objects Objects"""
    
    def __init__(self, p: float, t: float, name: str):
        """Constructor of the general liquid gas fluid object

        Args:
            p (float): Pressure of the liquid/gas [ Pa ]
            t (float): Temperature of the liquid/gas [ K ]
            name (str): Name of the liquid/gas [ N.A ]
        """
        self._p = p
        self._t = t
        self._name = name
        
        # We then set the remaining parameters as being None, until they are defined by the user
        self._rho = None
        self._gamma = None
        self._R = None
        self._cp = None
        self._cv = None
        self._c_sonic = None
        self._mue = None
        self._B = None
        self._h = None
        self._s = None
        self._p_vap = None
        
        return
    
    @abstractmethod
    def update() -> None:
        """Abstract method for updating the liquid/gas
        """
        pass
    
    @abstractmethod
    def copy() -> Self:
        """Function used to instantiate an exact copy of the same object
        """
        pass
    
    @property
    def p(self) -> float:
        """Getter for the fluid pressure

        Returns:
            float: Fluid Pressure [ Pa ]
        """
        return self._p
    
    @property
    def t(self) -> float:
        """Getter for the fluid temperature

        Returns:
            float: Fluid Temperature [ K ]
        """
        return self._t
    
    @property
    def rho(self) -> float:
        """Getter for the fluid density

        Returns:
            float: Fluid Density [ kg / m^3 ]
        """
        return self._rho
    
    @property
    def gamma(self) -> float:
        """Getter for the specific heat ratio of the fluid

        Returns:
            float: Fluid Specific Heat Ratio [ N.D. ]
        """
        return self._gamma
    
    @property
    def R(self) -> float:
        """Getter for the specific gas constant of the fluid

        Returns:
            float: Fluid Specific Gas Constant [ J / kg K]
        """
        return self._R
    
    @property
    def cp(self) -> float:
        """Getter for the fluid specific heat capacity at constant pressure

        Returns:
            float: Fluid specific heat capacity at constant pressure [ J / kg K ]
        """
        return self._cp
    
    @property
    def cv(self) -> float:
        """Getter for the fluid specific heat capacity at constant volume

        Returns:
            float: Fluid specific heat capacity at constant volume [ J / kg K ]
        """
        return self._cv
    
    @property
    def c_sonic(self) -> float:
        """Getter for the sonic speed in the fluid

        Returns:
            float: Sonic Speed through the fluid [ m / s ]
        """
        return self._c_sonic
    
    @property
    def mue(self) -> float:
        """Getter for Fluid dynamic viscosity

        Returns:
            float: Dynamic Viscosity [ Pa s ]
        """
        return self._mue
    
    @property
    def B(self) -> float:
        """Getter for Fluid Compressibility Factor

        Returns:
            float: Compressibility Factor [ Pa / m^3]
        """
        return self._B
    
    @property
    def h(self) -> float:
        """Getter for Fluid Specific Enthalpy

        Returns:
            float: Specific Enthalpy [ J / kg ]
        """
        return self._h
    
    @property
    def s(self) -> float:
        """Getter for Fluid Specific Entropy

        Returns:
            float: Specific Entropy [ J / kg ]
        """
        return self._s
    
    @property
    def p_vap(self) -> float:
        """Getter for Fluid Vapour Pressure

        Returns:
            float: Fluid Vapour Pressure [Pa]
        """
        return self._p_vap
    
    @property
    def name(self) -> str:
        return self._name
    

    
    
class CoolPropFluid(LiquidGasFluid):
    """Object used to wrap CoolProp Fluids into the standard fluid interface"""
    
    def __init__(self, p: float, t: float, name: str):
        super().__init__(p, t, name)
        
        # We now construct our fluid object
        self._fluid = AbstractState("HEOS", name)
        
        # We now update hte state of the fluid
        self.update(p=self._p, t=self._t)
        
    def update(self, 
               p: float | None = None, 
               t: float | None = None, 
               rho: float | None = None,
               h: float | None = None,
               s: float | None = None,
               ) -> None:
        """Updates the state of the CoolProp Fluid

        Args:
            p (float, optional): Pressure of the Fluid [Pa]
            t (float, optional): Temperature of the Fluid [K]
            rho (float, optional): Density of the Fluid [kg/m^3]
            h (float, optional): Specific Enthalpy of the Fluid [ J / kg ]
            s (float, optional): Specific Entropy of the Fluid [ J / kg ]
        """
        if sum(x is not None for x in (p, t, rho, h, s)) != 2:
            raise ValueError(f"Exactly two parameters are needed for Fluid updating!")
        
        params = []
        values = []
        
        # We extract relevant parameters
        if p is not None:
            params.append("P")
            values.append(p)
        
        if t is not None:
            params.append("T")
            values.append(t)
            
        if rho is not None:
            params.append("D")
            values.append(rho)
            
        if h is not None:
            params.append("H")
            values.append(h)
        
        if s is not None:
            params.append("S")
            values.append(s)
        
        # We then get our input parameter and value tuples
        input_params = tuple(params)
        input_value = tuple(values)
        
        # We then use our utility function tog et the proper order and associated key
        (input_ordered, value_order, cp_flag) = coolprop_input(input_pair = input_params,
                                                               value_pair = input_value)
        
        # We now update the state of the fluid
        self._fluid.update(cp_flag, value_order[0], value_order[1])
        
        # We can now update all of our relevant properties.
        self._p = self._fluid.p()
        self._t = self._fluid.T()
        self._rho = self._fluid.rhomass()
        
        self._cp = self._fluid.cpmass()
        self._cv = self._fluid.cvmass()
        self._gamma = self._cp / self._cv
        self._R = 8.314 / self._fluid.molar_mass()
        self._B = self._fluid.compressibility_factor()
        
        self._h = self._fluid.hmass()
        self._s = self._fluid.smass()
        self._p_vap = CP.PropsSI("P", "T", self._t, "Q", 0, self._name)
        
    def copy(self) -> Self:
        """Function that returns the CoolProp Fluid itself
        """
        
        return CoolPropFluid(p = self._p, t = self._t, name = self._name)
        
        

class IdealGas(LiquidGasFluid):
    def __init__(
        self,
        p: float,
        t: float,
        name: str,
        cp: float | None = None,
        cv: float | None = None,
        gamma: float | None = None,
        R: float | None = None,
    ) -> None:
        """Constructor for an Ideal Gas Object. Initial properties are derived via Coolprop, or set manually.

        Args:
            p (float): Gas Pressure (Pa)
            t (float): Gas Temperature (K)
            R (float): Gas Constant (J/kg K)
            gamma (float): Specific Heat Ratio (N/D)
            cp (float): Specific Heat Capacity (J/kg K)
        """
        super().__init__(p, t, name)
        
        # If the no additional properties are given, we used CoolProp
        n_param = sum(x is not None for x in (cp, cv, gamma, R))
        if n_param < 2:
            print(f"Less than two thermodynamic parameters given, Using CoolProp fluid object.")
            fluid = CoolPropFluid(p=p, t=t, name=name)
            
            self._cp = fluid.cp()
            self._cv = fluid.cv()
            self._gamma = fluid.gamma()
            self._R = fluid.R()
        elif n_param == 4:
            
            self._cp = cp
            self._cv = cv
            self._gamma = gamma
            self._R = R
        
        else:
            
            params = []
            values = []
            if cp is not None:
                params.append("cp")
                values.append(cp)
                
            if cv is not None:
                params.append("cv")
                values.append(cv)
                
            if gamma is not None:
                params.append("gamma")
                values.append(gamma)
        
            if len(params) == 1:
                if R is not None:
                    params.append("R")
                    values.append(R)
                    
            if len(params) != 2: 
                raise ValueError(f"Need exactly two thermodynamic properties to define the fluid. {params}")
    
            param_t = tuple(params)
            value_t = tuple(values)
            
            # We can organied the parameters
            input_pair, val_pair, key = ideal_input(input_pair=param_t, value_pair=value_t)
            
            thermo_properies = ideal_thermodynamic_parameters(values=val_pair, ideal_flag=key)
            
            # We now extract our properties
            self._cp = thermo_properies["cp"]
            self._cv = thermo_properies["cv"]
            self._gamma = thermo_properies["gamma"]
            self._R = thermo_properies["R"]
            
            # If R is not none, we over-run the specific gas constant.
            if R is not None:
                self._R = R
                
        # We finally update the state of the gas
        self.update(p=p, t=t)

        return

    def update(self, p: float | None = None, t: float | None = None, rho: float | None = None) -> None:
        """Update function using standard ideal gas relations

        Args:
            p (float | None, optional): Pressure of the ideal gas [Pa]. Defaults to None.
            t (float | None, optional): Temperature of the ideal gas [K]. Defaults to None.
            rho (float | None, optional): Density of the ideal gas [kg / m^3]. Defaults to None.
        """
        
        # We firstly go through our selected parameters an see what the gas states are.
        params = []
        values = []
        
        if sum(x is not None for x in (p, t, rho)) != 2:
            raise ValueError(f"Need exactly two parameters to update Ideal gas object!")
        
        if p is not None:
            params.append("p")
            values.append(p)
            
        if t is not None:
            params.append("t")
            values.append(t)
            
        if rho is not None:
            params.append("rho")
            values.append(rho)
            
        # We create our entrance tuples
        param_t = tuple(params)
        value_t = tuple(values)
        
        input_pair, val_pair, key = ideal_input(input_pair=param_t, value_pair=value_t)
        
        thermo_properies = ideal_gas_law_parameters(values=val_pair, R=self._R, flag=key)
        
        self._p = thermo_properies["p"]
        self._t = thermo_properies["t"]
        self._rho = thermo_properies["rho"]
        self._h = self._cp * self._t
        self._u = self._cv * self._t
        self._c_sonic = (self._gamma * self._R * self._t)**(1/2)
        return
        
    def copy(self) -> Self:
        
        return IdealGas(p = self._p, t=self._t, name=self._name, cp =self._cp, cv = self._cv, gamma = self._gamma, R=self._R)


class IncompressibleFluid(LiquidGasFluid):
    """Generic Function Defining the Properties of an Incompressible Fluid"""

    def __init__(
        self,
        name: str,
        p: float,
        t: float,
        rho: float,
        mue: float | None = None,
        B: float | None = None,
    ) -> None:
        """Constructor for the Incompressible Fluid

        Args:
            name (str): Name of the fluid
            rho (float): Fluid Density (kg/m^3)
            P (float): Fluid Pressure (kg/m^3)
            T (float): Fluid Temperature (kg/m^3)
            mue (float | None, optional): Fluid Viscosity (Pa s). Defaults to None.
            B (float | None, optional): Fluid Bulk Modulus (Pa). Defaults to None.
        """

        super().__init__(p, t, name)
        
        self.update(p = self._p,
                    t = self._t,
                    rho = rho,
                    mue = mue,
                    B = B)

        return
    
    def update(self, 
               p: float | None = None, 
               t: float | None = None, 
               rho: float | None = None, 
               mue: float | None = None, 
               B: float | None = None
               ) -> None:
        # We simply assign parameter if specified
        if p is not None:
            self._p = p
            
        if t is not None:
            self._t = t
        
        if rho is not None:
            self._rho = rho
        
        if mue is not None:
            self._mue = mue
            
        if B is not None:
            self._B = B
            
        return
    
    def copy(self) -> Self:
        
        return IncompressibleFluid(name = self._name,
                                   p = self._p,
                                   t = self._t,
                                   rho = self._rho,
                                   mue = self._mue,
                                   B = self._B)
