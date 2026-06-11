"""This File contains all the function used for the modelling and design of fixed leading/trailing edges"""

import numpy as np
from turborocket.solvers.solver import adjoint
from turborocket.profiling.Supersonic.circular import M_star, inv_M_star


def edge_area_rat(t_g_rat: float, beta_e: float, beta_i: float) -> float:
    """Function that solves for the area ratio between the inlet conditions of the supersonic
    turbine and the entry conditions of the turbine.

    Args:
        t_g_rat (float): Leading Edge thickness to throat area ratio
        beta_e (float): Leading Edge Entry Angle (rad)
        beta_i (float): Farfield Entry Angle (rad)

    Returns:
        float: Area Ratio of the Profile Entry
    """
    a_rat = (1 - (t_g_rat)) * ((np.cos(beta_e) / np.cos(beta_i)))

    return a_rat


def get_beta_e(t_g_rat: float, beta_i: float, a_rat: float) -> float:
    """Function that evaluates for the blade metal inlet angle, based on the area ratio compared to the far field and mach number

    Args:
        t_g_rat (float): Blade Leading Edge to Spacing Thickness [ N.D. ]
        beta_i (float): Farfield Inlet Angle [ rad ]
        a_rat (float): Ratio of Profile Entry [ N.D. ]

    Returns:
        float: Blade Metal Inlet Angle [ rad ]
    """
    # We can evaluate it
    beta_e = np.arccos((a_rat * np.cos(beta_i) / (1 - (t_g_rat))))

    return beta_e


def oblique_shock_area_rat(M_star_e: float, M_star_i: float, gamma: float) -> float:
    """This function solves for the oblique shock loss area ratio.

    Args:
        M_star_e (float): Turbine Entry Mach Number
        M_star_i (float): Turbine Farfield Inlet Mach Number
        gamma (float): Specific Heat Ratio

    Returns:
        float: Area ratio from the oblique shock
    """

    a_rat = (M_star_i / M_star_e) * (
        ((gamma + 1) / 2 - ((gamma - 1) / 2) * M_star_i**2)
        / ((gamma + 1) / 2 - ((gamma - 1) / 2) * M_star_e**2)
    ) ** (1 / (gamma - 1))

    return a_rat


def get_b_e(t_g_rat: float, beta_i: float, M_i: float, gamma: float) -> float:
    """This function solves for the entry Mach Number of the turbine profile

    Args:
        t_g_rat (float): Blade Thickness to throat area ratio
        beta_e (float): Leading Edge Entry Angle
        beta_i (float): Turbine Farfield Inlet Angle
        M_i (float): Turbine Farfield Inlet Mach Number
        gamma (float): Specific Heat Ratio (Cp/Cv)

    Returns:
        float: Entry Mach number of the turbine profile
    """

    # Firstly we need to solve for the nominal area ratio
    a_rat = edge_area_rat(t_g_rat=t_g_rat, beta_e=beta_e, beta_i=beta_i)

    # We evaluate for the critical Mach Number at the inlet
    M_star_i = M_star(gamma=gamma, M=M_i)

    M_star_e = adjoint(
        func=oblique_shock_area_rat,
        x_guess=M_star_i,
        dx=0.01,
        n=1000,
        relax=0.01,
        target=a_rat,
        params=[M_star_i, gamma],
        RECORD_HIST=False,
    )

    # We need to get the entry mach number based on the previous critical mach number
    M_e = inv_M_star(gamma=gamma, M_star=M_star_e)

    return M_e


def get_m_e(
    t_g_rat: float, beta_e: float, beta_i: float, M_i: float, gamma: float
) -> float:
    """This function solves for the entry Mach Number of the turbine profile

    Args:
        t_g_rat (float): Blade Thickness to throat area ratio
        beta_e (float): Leading Edge Entry Angle
        beta_i (float): Turbine Farfield Inlet Angle
        M_i (float): Turbine Farfield Inlet Mach Number
        gamma (float): Specific Heat Ratio (Cp/Cv)

    Returns:
        float: Entry Mach number of the turbine profile
    """

    # Firstly we need to solve for the nominal area ratio
    a_rat = edge_area_rat(t_g_rat=t_g_rat, beta_e=beta_e, beta_i=beta_i)

    # We evaluate for the critical Mach Number at the inlet
    M_star_i = M_star(gamma=gamma, M=M_i)

    M_star_e = adjoint(
        func=oblique_shock_area_rat,
        x_guess=M_star_i,
        dx=0.01,
        n=1000,
        relax=0.01,
        target=a_rat,
        params=[M_star_i, gamma],
        RECORD_HIST=False,
    )

    # We need to get the entry mach number based on the previous critical mach number
    M_e = inv_M_star(gamma=gamma, M_star=M_star_e)

    return M_e
