"""File that contains injector class objects"""

from turborocket.fluids.fluids import LiquidGasFluid  
from enum import Enum

class InjectorMethods(Enum):
    
    SPI = "SPI"
    HEM = "HEM"
    NHEM = "NHEM"


class InjectorElement():
    """Object defining an Injector of a single propellant"""
    
    def __init__(self,
                 cda: float,
                 inj_method: InjectorMethods | None = None
                 ) -> None:
        """Constructor for the Injector Face Object

        Args:
            propellant (IncompressibleFluid): Name of the propellant being injected.
            cda (float): Effective Orifice Area of the Injector Head [ m^2 ]
            inj_method (InjectorMethods | None, option): Injector Methods for modelling. Defaults to None.
        """
        
        self._cda = cda
        self._inj_method = inj_method
        
        return
    
    @property
    def cda(self) -> float:
        """Effective Orifice Area of the injector head

        Returns:
            float: Effective orifice area of the injector head [ m^2 ]
        """
        
        return self._cda
    
    @cda.setter
    def cda(self, cda: float) -> None:
        """Setter for the cda of the fluid"""
        self._cda = cda
        
        return
    
    @property
    def inj_method(self) -> InjectorMethods:
        """Injector Modelling Method"""
        
        return self._inj_method
    
    @inj_method.setter
    def inj_method(self, method: inj_method) -> None:
        """Setter for the Injector Method of the injector"""
        
        self._inj_method = method
    
    def m_dot_spi(self,
                  inlet: LiquidGasFluid, 
                  outlet: LiquidGasFluid, 
                  cda: float
                  ) -> float:
        """Utility function that gets the mass flow rate through the injector, assuming incompressible conditions.

            -> We assume incompressible conditions, using the inlet density.

        Args:
            inlet (LiquidGasFluid): Fluid Object upstream of injector
            outlet (LiquidGasFluid): Fluid Object downstream of injector
            cda (float): Effective Orifice Area of the injector head [ m^2 ]

        Returns:
            float: Mass flow rate of propellant through the injector [ kg/s ]
        """
        
        # We get the inlet conditions of the component
        rho_inlet = inlet.rho
        p_inlet = inlet.p
        
        # We get the chamber pressure from the combustion gas object
        p_outlet = outlet.p
        
        # We can directly evaluate for the mass flow rate.
        m_dot = cda * (2 * rho_inlet * (p_inlet - p_outlet))**(1/2)
        
        return m_dot
        
    
    def m_dot_hem(self,
                  inlet: LiquidGasFluid, 
                  outlet: LiquidGasFluid, 
                  cda: float
                  ) -> float:
        """Utility function that gets the mass flow rate through the injector, assuming the homogenous equilibrium equation.
        
        -> We assume constant entropy across the injector, and evaluate for the mass flow rate based on the enthalpy change.

        Args:
            inlet (LiquidGasFluid): Fluid Object upstream of the injector
            outlet (LiquidGasFluid): Fluid Object downstream of the injector
            cda (float): Effective Orifice Area of the Injector Head [ m^2 ]

        Returns:
            float: Mass flow rate of propellant through the injector [ kg/s ]
        """
        
        # We firstly extract our upstream enthalpy and entropy 
        h_inlet = inlet.h
        s_inlet = inlet.s
        
        
        # We get the downstream pressure
        p_outlet = outlet.p
        
        # We make a copy of the inlet object, and expand it isentropically to the downstream conditions
        inlet_copy: LiquidGasFluid = inlet.copy()
        
        # We then update this copied object at the outlet pressure assuming isentropic expansion
        inlet_copy.update(s=s_inlet, p=p_outlet)
        
        # We then get the enthalpy and density of the downstream section
        h_outlet = inlet_copy.h
        rho_outlet = inlet_copy.rho

        return cda * rho_outlet * (2 * (h_inlet - h_outlet)) ** (1/2)
    
    def m_dot_nhem(self, 
                   inlet: LiquidGasFluid, 
                   outlet: LiquidGasFluid, 
                   cda: float
                   ) -> float:
        """Utility function that gets the mass flow rate through the injector, using the non-homogenous equlibrium equation

        Args:
            inlet (LiquidGasFluid): Fluid Object upstream of the injector
            outlet (LiquidGasFluid): Fluid Object downstream of the injector
            cda (float): Effective Discharge Coefficient of the injector head [ m^2 ]

        Returns:
            float: Mass flow rate of propellant through the injector [ kg/s ]
        """
        # We get the inlet pressure and vapour pressure at the inlet
        p_inlet = inlet.p
        p_vap = inlet.p_vap
        
        # We get the downstream pressure
        p_outlet = outlet.p
        
        # We then evaluate for the kappa parameter
        if p_outlet >= p_vap:
            k= 1000000000000000000000000
        
        else:
            
            k = ( (p_inlet - p_outlet) / (p_vap - p_outlet) )**(1/2)
        
        
        # We get the incompressible mass flow
        m_dot_spi = self.m_dot_spi(inlet=inlet, outlet=outlet, cda=cda)
        
        # We get the HEM mass flow
        m_dot_hem = self.m_dot_hem(inlet=inlet, outlet=outlet, cda=cda)
        
        # We can then evaluate for our NHEM mass flow rate
        m_dot = (k / (1+k))*m_dot_spi + (1 / (1+k)) * m_dot_hem
        
        return m_dot
    
    def m_dot(self, 
              inlet: LiquidGasFluid, 
              outlet: LiquidGasFluid,
              inj_method: InjectorMethods | None = None
              ) -> float:
        """Equation to solve for the mass-flow rate across the injector

        Args:
            inlet (LiquidGasFluid): Fluid Object upstream of the injector
            outlet (LiquidGasFluid): Fluid Object downstream of the injector
            cda (float): Effective Discharge Coefficient of the injector head [ m^2 ]
            method (InjectorMethods, optional): Method for computation of the injector mass-flow rate. Defaults to the method used at instantiation.

        Returns:
            float: Mass flow rate of propellant through the injector [ kg/s ]
        """
        if inj_method is None:
            inj_method = self._inj_method
            
        inlet_2 = inlet
        
        match inj_method:
            
            case InjectorMethods.SPI:
                m_dot = self.m_dot_spi(inlet = inlet, 
                                       outlet = outlet, 
                                       cda = self._cda)
                
            case InjectorMethods.HEM:
                m_dot = self.m_dot_hem(inlet = inlet, outlet = outlet, cda = self._cda)
                
            case InjectorMethods.NHEM:
                m_dot = self.m_dot_nhem(inlet = inlet, outlet = outlet, cda = self._cda)
        
            case _:
                raise ValueError(f"Injector Method has not been set, please set this first. {inj_method}")
        
        # We finally return the parameter to the user
        return m_dot
