"""This file contains Enums, dictionaries and other support functions for the fluids.py file"""

import CoolProp.CoolProp as CP

def coolprop_input(input_pair: tuple[str, str], value_pair: tuple[float, float]) -> tuple[tuple[str, str], tuple[float,float], str]:
    """Utility function that returns to the user a pair of input parameters based on a look-up from the CoolProp Dicitonary.
    
    In addition, inlet value pair is re-arranged accordingt to what the order of parameters is.

    Args:
        input_pair (tuple[str, str]): Input pair of parameters 
        value_pair (tuple[float, float]): Input pair of values

    Returns:
        tuple[tuple[str, str], tuple[float,float]]: Tuple containing correct list of parameters and values, along with the identifier.
    """
    # We firsly look-up the normal pair
    if input_pair in coolprop_lookup:
        
        # Coolprop key
        cp_key = coolprop_lookup[input_pair]
        
    # We now check to see if the reverse case exists.
    elif (input_pair[1], input_pair[0]) in coolprop_lookup:
        
        # We get the cool prop key
        cp_key = coolprop_lookup[(input_pair[1], input_pair[0])]

        # We re-define the input and value pair
        input_pair = (input_pair[1], input_pair[0])
        value_pair = (value_pair[1], value_pair[0])
        
    else:
        raise ValueError(f"Combination of parameters could not be found in look-up dict!")
    
    # We can return to the user a tuple of these parameters:
    return (input_pair, value_pair, cp_key)

def ideal_input(input_pair: tuple[str, str], value_pair: tuple[float, float]) -> tuple[tuple[str, str], tuple[float,float], str]:
    """Utility function that returns to the user a pair of input parameters based on a look-up from the Ideal Dicitonary.
    
    In addition, inlet value pair is re-arranged accordingt to what the order of parameters is.

    Args:
        input_pair (tuple[str, str]): Input pair of parameters 
        value_pair (tuple[float, float]): Input pair of values

    Returns:
        tuple[tuple[str, str], tuple[float,float]]: Tuple containing correct list of parameters and values, along with the identifier.
    """
    # We firsly look-up the normal pair
    if input_pair in ideal_lookup:
        
        # Coolprop key
        ideal_key = ideal_lookup[input_pair]
        
    # We now check to see if the reverse case exists.
    elif (input_pair[1], input_pair[0]) in ideal_lookup:
        
        # We get the cool prop key
        ideal_key = ideal_lookup[(input_pair[1], input_pair[0])]

        # We re-define the input and value pair
        input_pair = (input_pair[1], input_pair[0])
        value_pair = (value_pair[1], value_pair[0])
        
    else:
        raise ValueError(f"Combination of parameters could not be found in look-up dict! {input_pair}")
    
    # We can return to the user a tuple of these parameters:
    return (input_pair, value_pair, ideal_key)

# We can define our dictionary of look up parameters
# We use the key function, that way order of inputs doesnt matter.

coolprop_lookup = {
    ("P", "T"): CP.PT_INPUTS,
    ("D", "P"): CP.DmassP_INPUTS,
    ("D", "T"): CP.DmassT_INPUTS,
    ("H", "S"): CP.HmassSmass_INPUTS,
    ("H", "P"): CP.HmassP_INPUTS,
    ("H", "T"): CP.HmassT_INPUTS,
    ("D", "H"): CP.DmassHmass_INPUTS,
    ("P", "S"): CP.PSmass_INPUTS,
    ("S", "T"): CP.SmassT_INPUTS,
    ("D", "S"): CP.DmassSmass_INPUTS,
}

ideal_lookup = {
    ("cp", "cv"): 0,
    ("cp", "gamma"): 1,
    ("cv", "gamma"): 2,
    ("R", "gamma"): 3,
    ("R", "cp"): 4,
    ("R", "cv"): 5,
    ("p", "t"): 6,
    ("p", "rho"): 7,
    ("t", "rho"): 8
}

def ideal_thermodynamic_parameters(values: tuple[float, float], ideal_flag: int) -> dict:
    """Function for deriving complete thermodynamic properties.

    Args:
        values (tuple[2]): Tuple of value parameters

    Returns:
        dict: Dictionary of thermodynamic parameters
    """
    
    # We get our key-value pair re-organised.
    
    match ideal_flag:
        
        case 0:
            
            cp = values[0]
            cv = values[1]
            gamma = cp / cv
            R = cp - cv
            
        case 1:
            
            cp = values[0]
            gamma = values[1]
            cv = cp/gamma
            R = cp - cv
            
        case 2:
            
            cv = values[0]
            gamma = values[1]
            cp = gamma * cv
            R = cp - cv
            
        case 3:
            
            R = values[0]
            gamma = values[1]
            cv = R / (gamma - 1)
            cp = R + cv
            
        case 4:
            
            R = values[0]
            cp = values[1]
            cv = cp - R
            gamma = cp / cv
            
        case 5:
            
            R = values[0]
            cv = values[1]
            cp = R + cv
            gamma = cp / cv
            
        case _:
            raise NotImplementedError(f"Function not implemented!")
        
    # We can now create our thermodynamic properties dictionary
    dic = {
        "cp": cp,
        "cv": cv,
        "gamma": gamma,
        "R": R
    }
    
    return dic

def ideal_gas_law_parameters(values: tuple[float, float], R: float, flag: int) -> dict:
    """Function for deriving complete thermodynamic properties.

    Args:
        params (tuple[str, str]): Tuple of parameters names
        values (tuple[float, float]): Tuple of value parameters
        R (float): Specific Gas Constant of the Fluid [J / kg K]
        flag (int): Flag for the ideal gas law parameters

    Returns:
        dict: Dictionary of thermodynamic parameters
    """
    
    match flag:
        
        case 6: # (p,t)
            
            p = values[0]
            t = values[1]
            rho = p / (R * t)
            
        case 7: # (p, rho)
            
            p = values[0]
            rho = values[1]
            t = p / (rho * R)
            
        case 8: # (t, rho)
            
            t = values[0]
            rho = values[1]
            p = rho * R * t
            
        case _:
            raise NotImplementedError(f"Function not implemented!")
        
        
        
    # We can now create our thermodynamic properties dictionary
    dic = {
        "p": p,
        "t": t,
        "rho": rho,
    }
    
    return dic