"""File to contain all additional isentropic relations used in gas dynamics"""

from turborocket.fluids.fluids import LiquidGasFluid

from turborocket.profiling.Supersonic.circular import inv_M_star

import numpy as np

def isen_expansion_dh(fluid: LiquidGasFluid, p_0: float, p_1: float, t_0: float) -> float:
    """Function that evaluates for the isentropic expansion

    Args:
        fluid (LiquidGasFluid): Fluid Object where thermodynamic properties are derived.
        p_0 (float): Stagnation Pressure of the Fluid [Pa]
        p_1 (float): Static Pressure of the Fluid [Pa]
        t_0 (float): Stagnation Temperature of the Fluid [K]

    Returns:
        float: Enthalpy Drop as a result of an isentropic expansion
    """
    # Extracting thermodynamic properties
    cp = fluid.cp
    gamma = fluid.gamma
    
    # Enthalpy drop
    dh = cp * t_0 * (1 - (p_1 / p_0) ** ((gamma - 1) / gamma))
    
    return dh

def isen_expansion_cis(dh: float) -> float:
    """Gets the isentropic velocity caused by an enthalpy pressure drop

    Args:
        dh (float): Enthalpy Drop caused by an isentropic expansion [ J / kg ]

    Returns:
        float: Isentropic velocity [ m / s ]
    """
    
    return np.sqrt( 2 * dh )

def isen_static_t_M(fluid: LiquidGasFluid, t_0: float, M: float) -> float:
    """Function that gets the static temperature of the fluid.

    Args:
        fluid (LiquidGasFluid): Fluid Object where thermodynamic properties are derived.
        t_0 (float): Stagnation Temperature [ K ]
        M (float): Mach Number of the Fluid.

    Returns:
        float: Static temperature of the fluid [ K ]
    """
    # We get the specific heat ratio of the fluid
    gamma = fluid.gamma
    
    # Static temperature of the fluid
    t_static = t_0 / (1 + (gamma - 1)*M**2 /2)

    return t_static

def isen_static_t_M_star(fluid: LiquidGasFluid, t_0: float, M_star: float) -> float:
    """Function that gets the static temperature fo the fluid, based on the stagnation temperature and crtitical mach number.

    Args:
        fluid (LiquidGasFluid): _Fluid Object where thermodynamic properties are derived.
        t_0 (float): Stagnation Temperature [ K ]
        M_star (float): Mach Number of the Fluid.

    Returns:
        float: Static Temperature of the fluid [ K ]
    """
    # We get the specific heat ratio of the fluid
    gamma = fluid.gamma
    
    # Static Temperature of the fluuid
    t_static = t_0 * ( 1 - ((gamma - 1) / (gamma + 1)) * M_star**2 ) 
    
    return t_static

def isen_static_p_M(fluid: LiquidGasFluid, p_0: float, M:float ) -> float:
    """Function that gets the static pressure of the fluid, given a stagnation pressure and Mach number of the fluid

    Args:
        fluid (LiquidGasFluid): Fluid Object where thermodynamic properties are derived.
        p_0 (float): Stagnation Pressure [ Pa ]
        M (float): Mach Number of the gas [ N.D. ]

    Returns:
        float: Static Pressure of the gas @ M [Pa]
    """
    # We get the specific heat ratio of the fluid
    gamma = fluid.gamma
    
    # We can evaluate for the gamm expression
    gamma_expression = (1 + (gamma - 1)/2 * M**2) ** (gamma / (gamma - 1))
    
    # We finally evalute for the static pressure of the fluid
    p_static = p_0 / gamma_expression
    
    return p_static
    

def isen_static_p_M_Star(fluid: LiquidGasFluid, p_0: float, M_star: float) -> float:
    """Function that gets the static pressure of the fluid, given a stagnation pressure and critical mach number of the gas.

    Args:
        fluid (LiquidGasFluid): Fluid Object where thermodynamic properties are derived.
        p_0 (float): Stagnation Pressure [ Pa ]
        M_star (float): Critical Mach Number of the gas [ N.D. ]

    Returns:
        float: Static Pressure of the gas @ M* [ Pa ]
    """
    # We get the specific heat ratio of the fluid
    gamma = fluid.gamma
    
    # We get the static pressure of the fluid
    p_static = p_0 * ( 1 - ((gamma - 1) / (gamma + 1)) * M_star**2 ) ** (gamma / (gamma - 1))
    
    return p_static

def isen_c_star(fluid: LiquidGasFluid, t_0: float) -> float:
    """Function that evaluates for the characteristics velocity of the fluid

    Args:
        fluid (LiquidGasFluid): Fluid Object where thermodynamic properties are derived.
        t_0 (float): Stagnation Temperature of the fluid [ Pa ]

    Returns:
        float: Characteristic Velocity of the Fluid [ m/s ]
    """
    # Firstly we get the gas properties.
    R = fluid.R
    gamma = fluid.gamma
    
    c_star = np.sqrt(R * t_0 / (gamma)) * np.power(
            2 / (gamma + 1), -(gamma + 1) / (2 * (gamma - 1))
        )
    
    return c_star