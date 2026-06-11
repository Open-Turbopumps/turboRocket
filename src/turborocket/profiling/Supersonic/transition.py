"""
transition.py

This source file encapsulates all the functions required for sizing the transition arc sections of the turbine blade, utilising the method of characteristics.

The following equations follows the method for supersonic turbine design presented in NASA TN D-4421.



# noqa: E501

"""

from turborocket.solvers.solver import adjoint

import numpy as np


def func_r_star(r_star: float, gamma: float) -> float:
    """Function of R* that results.

    Args:
        r_star (float): Normalised Radius Value [ N.D. ]
        gamma (float): Specific Heat Ratio of the fluid [ N.D. ]

    Returns:
        float: Function evaluation, f(R*)
    """
    # We split the evaluation into two parts in this case.
    f_r_star_a = ((gamma + 1) / (gamma - 1)) ** (1 / 2) * np.arcsin(
        ((gamma - 1) / r_star**2) - gamma
    )

    # And for the second part of the relationship is this
    f_r_star_b = np.arcsin((gamma + 1) * r_star**2 - gamma)

    # We can now combine these two parameters to one.
    f_r_star = f_r_star_a + f_r_star_b

    return f_r_star


def func_r_star_k(
    v_i: float,
    k: int,
    dv: float,
    gamma: float,
) -> float:
    """Function that evaluates for the R*_k varient of the equation, which is entirely dependent on the user defined prandtl Meyer Angle and gamma value.

    Args:
        v (float): Desired Prandtl Meyer Angle [ rad ]
        gamma (float): Specific Heat Ratio [ N.D. ]

    Returns:
        float: f(R*_k) function value [ rad ]
    """
    f_r_star_k = (
        2 * v_i
        - (np.pi / 2) * (((gamma + 1) / (gamma - 1)) ** (1 / 2) - 1)
        - 2 * (k - 1) * dv
    )

    return f_r_star_k


def solve_r_star(target: float, guess: float, gamma: float) -> float:
    """Function that solvers for the Normalised Radius of the characterstic.

    Simultaneous solution of both f(R*) and f(R*_k) is made, to derive a final solution for the parameter.

    Args:
        target (float): Target Function R_star value [ N.D. ]
        guess (float): User's first guess on the inital R* value
        gamma (float): Specific Heat Ratio of the gas [ N.D. ]


    Returns:
        _type_: Converged upon R* value [ N.D. ]
    """

    # We can now implement the user of our adjoint based optimised to solve for the r_star value
    r_star = adjoint(
        func=func_r_star,  # Desired function to be optimised
        x_guess=guess,  # Initial Guess of the R_star value
        dx=0.01,  # First Initial Step for the R_star value
        n=20000,  # Number of integration steps
        relax=0.2,  # Relaxation Parameter
        target=target,  # Target function value
        params=[gamma],  # Additional parameter of the function
    )

    return r_star


def vortex_coords(R_star_k: float, phi_k: float) -> tuple[float, float]:
    """Function that evaluates for the bulk velocity fluid co-ordinates, assuming forced vortex flow.

    Please note that co-odinates are returned in the tuple with the following format: ( x, y )

    Args:
        R_star_k (float): Normalised Radius of the velocity vector at point k [ N.D. ]
        phi_k (float): Velocity Vector at point k [ rad ]

    Returns:
        tuple[float, float]: Co-ordinates of the bulk velocity vector at point k
    """
    x_star = -R_star_k * np.sin(phi_k)
    y_star = R_star_k * np.cos(phi_k)

    return (x_star, y_star)


def mue_k(
    r_star_k: float,
    gamma: float,
) -> float:
    """Function that evaluates the angle between the bulk velocity vector and the mach-angle from point k.

    Args:
        r_star_k (float): Normalised radius of point k in the bulk flow [ N.D. ]
        gamma (float): Specific Heat Ratio of the fluid [ N.D. ]

    Returns:
        float: Angle between velocity and the mach angle [ rad ]
    """

    # We can derive the mach angle directly
    mach_angle = -np.arcsin(
        (((gamma + 1) / 2) * r_star_k**2 - ((gamma - 1) / 2)) ** (1 / 2)
    )

    return mach_angle


def mach_slope(
    angles_k_1: tuple[float, float], angles_k_2: tuple[float, float]
) -> float:
    """Function that evaluates for the approximate slope of the mach wave between the bulk fluid velocity at a point and the wall.

    A mean line is drawn between two adjavent points (k_1 and k_2) form which a gradient is evaluated by taking the tan of the combined mean angle.

    Args:
        angles_k_1 (tuple[float, float]): Tuple of the bulk velocity vector angle and the mach slope angle at k_1 ( phi, mue ) [ rad, rad ]
        angles_k_2 (tuple[float, float]): Tuple of the bulk velocity vector angle and the mach slope angle at k_1 ( phi, mue ) [ rad, rad ]

    Returns:
        float: Mean gradient of the mach wave at point k_1 [ N.D. ]
    """
    # We firstly can extract our relevant parameters

    phi_k_1 = angles_k_1[0]
    mue_k_1 = angles_k_1[1]

    phi_k_2 = angles_k_2[0]
    mue_k_2 = angles_k_2[1]

    # We can directly evaluate for the gradient accordingly.
    m_k = np.tan(((phi_k_1 + phi_k_2) / 2) + ((mue_k_1 + mue_k_2) / 2))

    return m_k


def wall_slope(phi_k_2: float) -> float:
    """Function that evaluates for the gradient of wall, which is assumed parallel to the velocity vector of the bulk fluid.

    Args:
        phi_k_2 (float): Angle of bulk fluid velocity vector at point k_2 [ rad ]

    Returns:
        float: Gradient of the wall at selected point k_1 [ N.D. ]
    """

    # We simply evaluate for the gradient directly
    m_bar_k = np.tan(phi_k_2)

    return m_bar_k


def wall_coords(
    star_k_1: tuple[float, float],
    star_lk_2: tuple[float, float],
    m_bar_k_1: float,
    m_k_1: float,
) -> tuple:
    """Function that evaluates for the wall co-ords based on the gradient of the two lines between the mach angle and the mean velocity vector.

    Co-ordinates are to be specified in tuple format, with the following convention: [ x, y ]

    Args:
        star_k_1 (tuple[float, float]): Co-ordinate locations of the bulk velocity fluid at point k_1 [ N.D, N.D. ]
        star_lk_2 (tuple[float, float]): Co-ordinate locations of the wall position at point k_2 [ N.D. N.D. ]
        m_bar_k_1 (float): Slope of the wall at point k_1 [ N.D. ]
        m_k_1 (float): Slope of the mach wave at point k_1 [ N.D. ]

    Returns:
        tuple: Wall Co-ordinates at point k_1 [ N.D., N.D. ]
    """
    # Firstly, we can de-pack our co-ordinate tuples
    x_k_1 = star_k_1[0]
    y_k_1 = star_k_1[1]

    x_lk_2 = star_lk_2[0]
    y_lk_2 = star_lk_2[1]

    x_lk_1 = ((y_lk_2 - m_bar_k_1 * x_lk_2) - (y_k_1 - m_k_1 * x_k_1)) / (
        m_k_1 - m_bar_k_1
    )

    y_lk_1 = (
        m_k_1 * (y_lk_2 - m_bar_k_1 * x_lk_2) - m_bar_k_1 * (y_k_1 - m_k_1 * x_k_1)
    ) / (m_k_1 - m_bar_k_1)

    return (x_lk_1, y_lk_1)


def transform_coord(star_lk: tuple[float, float], alpha_l_i: float) -> tuple:
    """Function that transforms wall co-ordinates from a given reference frame to another, based on a rotation angle.

    It should be noted that the co-ordinates supplied and provided use the following convention: [x, y]
    Args:
        star_lk (tuple[float, float]): Co-ordinates of wall location at point k [ N.D., N.D. ]
        alpha_l_i (float): Angle to rotate the co-ordiantes about [ rad ]

    Returns:
        tuple: Transformed co-ordinates of the wall location at point k [ N.D., N.D. ]
    """
    # First thing we do is extract our core co-ordinates.
    x_star_l_k = star_lk[0]
    y_star_l_k = star_lk[1]

    x_star_l_k_t = x_star_l_k * np.cos(alpha_l_i) - y_star_l_k * np.sin(alpha_l_i)

    y_star_l_k_t = x_star_l_k * np.sin(alpha_l_i) + y_star_l_k * np.cos(alpha_l_i)

    return (x_star_l_k_t, y_star_l_k_t)


def method_of_characteristics(
    k_max: int,
    v_i: float,
    v_l: float,
    gamma: float,
    alpha_l_i: float,
    reverse_x: bool = False,
) -> tuple[list, list]:
    """Simple Method Of Characteristics Functions for Transition Zones

    Args:
        k_max (int): Maximum Number of points on the Transition Arc
        v_i (float): Initial Prandtl Meyer Angle [ rad ]
        v_l (float): Final Surface Prandtl Angle [ rad ]
        alpha_l_i (float): Angle to Offset the Curve Produced [ rad ]
        reverse_x (bool, optional): Flag for reverse the x-axis of the plot. Defaults to False

    Returns:
        tuple[dict, dict]: Co-ordinate Point List for the MoC Transition Zones.
    """

    # First thing we do is setup our arrays for storing all our key dimensions.

    k_array = np.arange(start=k_max + 1, stop=0, step=-1)

    phi_array = np.zeros(k_max + 1)  # Velocity Angle to v_l reference Frame [ rad ]
    mue_array = np.zeros(
        k_max + 1
    )  # Velocity Angle with respect to the mach line of the velocity vector [ rad ]

    r_star_array = np.zeros(k_max + 1)  # Flow Velocity Radius [ N.D ]

    xk_array = np.zeros(
        k_max + 1
    )  # Normalised x co-ordinate of major vortex expansion characteristic [ N.D. ]
    yk_array = np.zeros(
        k_max + 1
    )  # Normalised y co-ordinate of major vortex expansion characteristic [ N.D. ]

    m_k_array = np.zeros(
        k_max + 1
    )  # Gradient of the Mach Line with respect to the normalised co-ordinate system [ N.D. ]
    m_bar_k_array = np.zeros(
        k_max + 1
    )  # Gradient of the Wall Segment, which is assumed to be parallel to velocity direction [ N.D. ]

    xlk_array = np.zeros(
        k_max + 1
    )  # Normalised x co-ordinate of the wall position which is parallel to the flow bulk direction [ N.D. ]
    ylk_array = np.zeros(
        k_max + 1
    )  # Normalised y co-ordinate of the wall position which is parallel to the flow bulk direction [ N.D. ]

    xlkt_array = np.zeros(
        k_max + 1
    )  # Transposed Normalised x co-ordinate of the wall position with respect to the final prandtl Meyer Angle [ N.D. ]
    ylkt_array = np.zeros(
        k_max + 1
    )  # Transposed Normalised y co-ordiante of the wall position with respect to the final Prandtl Meyer Angle [ N.D. ]

    v_array = np.linspace(v_l, v_i, k_max + 1)

    if reverse_x is True:
        x_a = -1
    else:
        x_a = 1

    # We define the bounds of the guess
    guess_max = 1
    guess_min = ((gamma - 1) / (gamma + 1)) ** (1 / 2)

    # We define our guess as the mid point
    guess = (guess_max + guess_min) / 2

    # We can then iterate through our loop
    for i, val in np.ndenumerate(k_array):

        i = i[0]

        # We can evalute for our phi
        phi_array[i] = v_array[i] - v_l

        # We solve for the target r_star
        r_star_t = func_r_star_k(v_i=v_i, k=val, dv=(v_i - v_l) / k_max, gamma=gamma)

        # We now must solver R* using the adjoint method
        r_star_array[i] = solve_r_star(
            target=r_star_t,
            guess=guess,
            gamma=gamma,
        )

        # We can now compute the co-ordinates of the expansion line
        (xk_array[i], yk_array[i]) = vortex_coords(r_star_array[i], phi_array[i])

        # We need to now compute the mach angle
        mue_array[i] = mue_k(r_star_k=r_star_array[i], gamma=gamma)

        # If we are on the first point, we dont need to evaluate for the gradient and assumed will be the same
        if i == 0:
            # The gradient of the initial co-ordinates.
            m_k_array[i] = 0
            m_bar_k_array[i] = 0

            # The wall is the same as the initial co-ordinates.
            xlk_array[i] = xk_array[i]
            ylk_array[i] = yk_array[i]

            # We transform the co-ordinates accordingly.
            [xlkt_array[i], ylkt_array[i]] = transform_coord(
                (xlk_array[i], ylk_array[i]), alpha_l_i
            )
            xlkt_array[i] *= x_a

            # We then go to the next loop accordingly
            continue

        # We must thus compute the slope of the mach line, based on the average of the current point and previous (k+1)
        m_k_array[i] = mach_slope(
            angles_k_1=(phi_array[i], mue_array[i]),
            angles_k_2=(phi_array[i - 1], mue_array[i - 1]),
        )

        # We can now compute the wall segment slope
        m_bar_k_array[i] = wall_slope(phi_k_2=phi_array[i - 1])

        # Wall Co-ordinates
        (xlk_array[i], ylk_array[i]) = wall_coords(
            star_k_1=(xk_array[i], yk_array[i]),
            star_lk_2=(xlk_array[i - 1], ylk_array[i - 1]),
            m_bar_k_1=m_bar_k_array[i],
            m_k_1=m_k_array[i],
        )

        # Finally, we do a co-ordinate transformation

        [xlkt_array[i], ylkt_array[i]] = transform_coord(
            (xlk_array[i], ylk_array[i]), alpha_l_i
        )

        xlkt_array[i] *= x_a

    return (xlkt_array, ylkt_array)
