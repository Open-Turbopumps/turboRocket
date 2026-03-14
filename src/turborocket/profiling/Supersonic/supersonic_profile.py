# This file encapsulates the main function for computing the supersonic profile

from turborocket.profiling.Supersonic.circular import (
    prandtl_meyer,
    M_star,
    vortex_flow_angles,
    inv_M_star,
    beta_o,
)

from turborocket.profiling.Supersonic.transition import method_of_characteristics

from turborocket.profiling.Supersonic.constraints import inv_mass_flow, r_star
from turborocket.profiling.Supersonic.constraints import (
    k_star_max,
    Q,
    C,
    shock_pressure_rat,
    M_i_star_max,
)
from turborocket.profiling.Supersonic.constraints import M_star_u_max, M_star_l_min

from turborocket.fluids.fluids import IdealGas

from turborocket.profiling.Supersonic.fixed_edge import get_m_e

import numpy as np
import math
import matplotlib.pyplot as plt

import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds

from turborocket.solvers.solver import adjoint


class SupersonicProfile:
    """Object Used to Define Supersonic Profiles"""

    # Constant for converting angles to radians
    ANGLE_CONVERSION = np.pi / 180

    def __init__(
        self,
        beta_i: float,
        M_r: tuple[float, float],
        M_s: tuple[float, float],
        fluid: IdealGas,
        deflection: float,
        t_g_rat: float = 0,
        k_max: int = 100,
        m_dot: float = 1,
        h: float = 1,
    ) -> None:
        """Constructor for Supersonic Profiles.

        It should be noted that for the specification of the relative velocity angles at inlet and outlet:

        - A clockwise direction relative to the axial direction is positive.
        - A counter clockwise direction relative to the axial direction is negative.

        As such, with most conventional blade profiles that are of "impulse design", the inlet angle tends to be negative, with the outlet relative blade angle as postiive.

        Args:
            beta_i (float): Inlet Bulk Velocity Angles with reference to the axial direction [ deg ]
            M_r (tuple[float, float]): Relative Inlet Mach Numbers at the Inlet and Outlet ( M_i, M_o ) [ N.D., N.D. ]
            M_s (tuple[float, float]): Peak Mach Numbers on the Upper and Lower **Passage Surfaces** of the Turbine ( M_u, M_l ) [ N.D., N.D.]
            fluid (IdealGas): Object representing the fluid flowing through the turbine profile.
            deflection (float): Deflection Angle of Entire assembly, clockwise [ rad ]
            t_g_rat (float, optionatl): Blade Pitch Wise LE/TE thickness to blade spacing ratio [ N.D. ]. Defaults to zero.
            k_max (int, optional): Number of Points to discretise our curves for.
            m_dot (float, optional): Mass Flow Rate Through Turbine [ kg/s ]. Defaults to 1.
            h (float, optional): Blade Turbine Height [ m ]. Defaults to 1.
        """

        # First think we do is unpack our arrays
        self._M_i, self._M_o = M_r
        self._M_u, self._M_l = M_s

        # We then do some simple validation to confirm the mach numbers provided are reasonable.
        if (self._M_u > self._M_i) or (self._M_u > self._M_o):
            raise ValueError(
                f"Mach Number on Upper Surface Cannot be Greater than Inlet or Outlet! {self._M_u} > {self._M_i}, {self._M_o}"
            )

        elif (self._M_l < self._M_i) or (self._M_l < self._M_o):
            raise ValueError(
                f"Mach Number on Lower Surface Cannot be Less than Inlet or Outlet! {self._M_l} < {self._M_i}, {self._M_o}"
            )

        # We can now solve for our critical velocity ratios.
        self._M_u_star = self.critical_velocity_ratio(m=self._M_u, gas=fluid)
        self._M_l_star = self.critical_velocity_ratio(m=self._M_l, gas=fluid)
        self._M_i_star = self.critical_velocity_ratio(m=self._M_i, gas=fluid)
        self._M_o_star = self.critical_velocity_ratio(m=self._M_o, gas=fluid)

        # We can now go on to solve for our Prandtl Meyer Angles in absolute frame
        self._v_u = self.prantdl_meyer(m_star=self._M_u, gas=fluid)
        self._v_l = self.prantdl_meyer(m_star=self._M_l, gas=fluid)
        self._v_i = self.prantdl_meyer(m_star=self._M_i, gas=fluid)
        self._v_o = self.prantdl_meyer(m_star=self._M_o, gas=fluid)

        # We can store our inlet angle and evaluate for hte outlet.
        self._deflection = deflection * self.ANGLE_CONVERSION
        self._beta_i = (beta_i + deflection) * self.ANGLE_CONVERSION
        self._beta_o = self.exit_blade_angle(
            beta_i=self._beta_i, M_i=self._M_i, M_o=self._M_o, gas=fluid
        )

        # Finally, we can evaluate for our circular mach numbers
        self._alpha_l_i = self.vortex_angle(
            beta_m=self._beta_i, v_i=self._v_i, v_f=self._v_l, surface=True
        )
        self._alpha_l_o = self.vortex_angle(
            beta_m=self._beta_o, v_i=self._v_l, v_f=self._v_o, surface=True
        )
        self._alpha_u_i = self.vortex_angle(
            beta_m=self._beta_i, v_i=self._v_i, v_f=self._v_u, surface=False
        )
        self._alpha_u_o = self.vortex_angle(
            beta_m=self._beta_o, v_i=self._v_u, v_f=self._v_o, surface=False
        )

        print(f"alpha_l_i: {self._alpha_l_o}")
        print(f"alpha_u_i: {self._alpha_u_o}")

        # We can now evaluate for our vortex flow radii
        self._R_l_star = self.vortex_radius(M_star=self._M_l)
        self._R_u_star = self.vortex_radius(M_star=self._M_u)

        # We can now generate our transition curves for inlet
        self._l_i_coord = self.transition_zone(
            v_i=self._v_i,
            v_f=self._v_l,
            k_max=k_max,
            fluid=fluid,
            alpha=self._alpha_l_i,
            reverse_x=True,
        )

        self._u_i_coord = self.transition_zone(
            v_i=self._v_i,
            v_f=self._v_u,
            k_max=k_max,
            fluid=fluid,
            alpha=-self._alpha_u_i,
            reverse_x=False,
        )

        # We deflect our inlet curves
        arrays = [
            self._l_i_coord,
            self._u_i_coord,
        ]

        c = np.cos(self._deflection)
        s = np.sin(self._deflection)

        for arr in arrays:
            x = arr[:, 0].copy()
            y = arr[:, 1].copy()

            arr[:, 0] = c * x - s * y
            arr[:, 1] = s * x + c * y

        # We can now solve for our straight zones:
        self._straight_i_coord = self.straight_zone(
            beta_m=-(self._beta_i - self._deflection),
            initial_coord=(self._l_i_coord[-1]),
            x_f=self._u_i_coord[-1, 0],
        )
        # We now get our blade spacing at inlet
        self._g_star = -(self._straight_i_coord[-1] - self._u_i_coord[-1])[1]

        print(f"target: {self._g_star}")

        # We can then run a study to optimsie for expansion angle to meet even spacing
        d_alpha = adjoint(
            func=self.blade_spacing_TE,
            x_guess=0.00,
            dx=0.02,
            n=10000,
            relax=0.5,
            target=self._g_star,
            params=[
                (self._alpha_u_o, self._alpha_l_o),
                self._v_o,
                self._beta_o,
                self._deflection,
                (self._v_u, self._v_l),
                k_max,
                fluid,
            ],
        )

        alpha_l = self._alpha_l_o - d_alpha
        alpha_u = self._alpha_u_o + d_alpha

        # We then get our outlet transitions
        self._l_o_coord = self.transition_zone(
            v_i=self._v_o + d_alpha,
            v_f=self._v_l,
            k_max=k_max,
            fluid=fluid,
            alpha=-alpha_l,
            reverse_x=False,
        )

        self._u_o_coord = self.transition_zone(
            v_i=self._v_o - 2 * d_alpha,
            v_f=self._v_u,
            k_max=k_max,
            fluid=fluid,
            alpha=alpha_u,
            reverse_x=True,
        )

        # We now need to solve for our circular section
        self._vortex_l_coord = self.vortex_zone(
            alpha=(self._alpha_l_i, self._alpha_l_o - d_alpha),
            r_star=self._R_l_star,
            k_max=k_max,
        )
        self._vortex_u_coord = self.vortex_zone(
            alpha=(self._alpha_u_i, self._alpha_u_o + d_alpha),
            r_star=self._R_u_star,
            k_max=k_max,
        )

        # We can finally now rotate our entire assembly according to the deflection angle.
        # We define our rotation matrix

        arrays = [
            self._l_o_coord,
            self._u_o_coord,
            self._vortex_l_coord,
            self._vortex_u_coord,
        ]

        c = np.cos(self._deflection)
        s = np.sin(self._deflection)

        for arr in arrays:
            x = arr[:, 0].copy()
            y = arr[:, 1].copy()

            arr[:, 0] = c * x - s * y
            arr[:, 1] = s * x + c * y

        # We then re-solve for our leading and trailing edges, considering the new rotation

        self._straight_o_coord = self.straight_zone(
            beta_m=-self._beta_o + self._deflection + 2 * d_alpha,
            initial_coord=(self._l_o_coord[-1]),
            x_f=self._u_o_coord[-1, 0],
        )

        # We can solve for the leading edge and trailing edge thickness
        if t_g_rat != 0:
            self._t = t_g_rat * self._g_star

            # We can now solve for our leading and traling edge angles
            le_radius = self.edge_radius(
                beta_m=(self._beta_i - self._deflection), t=self._t
            )
            te_radius = self.edge_radius(
                beta_m=(-self._beta_o + self._deflection + 2 * d_alpha), t=self._t
            )

            # We can now evaluate for the LE/TE circles
            self._LE_coord = self.vortex_zone(
                alpha=(
                    (self._beta_i - self._deflection) - np.pi,
                    (self._beta_i - self._deflection),
                ),
                r_star=le_radius,
                k_max=k_max,
            )

            self._TE_coord = self.vortex_zone(
                alpha=(
                    -(-self._beta_o + self._deflection + 2 * d_alpha) + np.pi,
                    -(-self._beta_o + self._deflection + 2 * d_alpha),
                ),
                r_star=te_radius,
                k_max=k_max,
            )

            # We then get our edge centroid location
            self._LE_center = self.edge_centroid(
                edge_coord=tuple(self._straight_i_coord[-1]),
                beta_m=(self._beta_i - self._deflection),
                r=le_radius,
            )
            self._TE_center = self.edge_centroid(
                edge_coord=tuple(self._straight_o_coord[-1]),
                beta_m=-(-self._beta_o + self._deflection + 2 * d_alpha),
                r=te_radius,
            )

            # We then offset our co-ordinates accordingly
            self._LE_coord += np.array([self._LE_center[0], self._LE_center[1]])
            self._TE_coord += np.array([self._TE_center[0], self._TE_center[1]])

            print(f"LE: ")
            print(self._LE_coord[-1])
            print(self._LE_coord[0])

            print(f"TE: ")
            print(self._TE_coord[-1])
            print(self._TE_coord[0])

            # Finally, we adjust our LE/TE Inlet Lines
            self._straight_o_coord = self.straight_zone(
                beta_m=-self._beta_o + self._deflection + 2 * d_alpha,
                initial_coord=(self._l_o_coord[-1]),
                x_f=self._TE_coord[-1, 0],
            )

            self._straight_i_coord = self.straight_zone(
                beta_m=-(self._beta_i - self._deflection),
                initial_coord=(self._l_i_coord[-1]),
                x_f=self._LE_coord[-1, 0],
            )

        self._m_dot = m_dot
        self._h = h
        self._fluid = fluid

        # self.generate_turbine_profile()

    def critical_velocity_ratio(self, m: float, gas: IdealGas) -> float:
        """Function that solves for the critical velocity ratio, based on the flowing gas and Mach Number

        Args:
            m (float): Desired Mach Number to be converted [ N.D. ]
            gas (IdealGas): Gas Flowing at the Desired Mach Number.

        Returns:
            float: Critical Velocity Ratio (M*) at the selected mach number [ N.D. ]
        """
        # We firstly need to extract our gamma parameter
        gamma = gas.gamma

        # We can now directly call for our utility function
        crit_vel = M_star(gamma=gamma, M=m)

        return crit_vel

    def prantdl_meyer(self, m_star: float, gas: IdealGas) -> float:
        """Function that solves for the Prandtl Meyer Angle, based on the flowing gas and the critical velocity ratio provided by the user.

        Args:
            m_star (float): Desired Critical Velocity Ratio to be converted [ N.D. ]
            gas (IdealGas): Gas Flowing at the Desired Critical Velocity Ratio.

        Returns:
            float: Prandtl Meyer Angle at the desired critical velocity ratio [ rad ]
        """
        # We firstly extract our gas propertiees
        gamma = gas.gamma

        # We can now use our utility function to evaluate for the prandtl Meyer Angle
        v = prandtl_meyer(gamma=gamma, crit_vel_rat=m_star)

        return v

    def exit_blade_angle(
        self, beta_i: float, M_i: float, M_o: float, gas: IdealGas
    ) -> float:
        """Function that solves for the exit blade angle, as a function of inlet and outlet Mach Numbers, along with gas properties.

        Args:
            beta_i (float): _description_
            M_i (float): Inlet Bulk Flow Mach Number [ N.D ]
            M_o (float): Outlet Bulk Flow Mach Number [ N.D. ]
            gas (IdealGas): Gas Flowing though the passage.

        Returns:
            float: Exit Blade Metal Angle [ rad ]
        """
        # We simply directly use our utility equation.
        b_o_m = beta_o(beta_i=beta_i, M_i=M_i, M_o=M_o, gamma=gas.gamma)

        return b_o_m

    def vortex_angle(
        self, beta_m: float, v_i: float, v_f: float, surface: bool
    ) -> float:
        """Function that evaluates for the suction sufrace vortex flow angles

        Args:
            beta_m (float): Blade Metal Angle
            v_i (float): Initial Prandtl Meyer Angle [ rad ]
            v_f (float): Final Prandtl Meyer Angle [ rad ]
            surface (bool): Surface Flag for Suction or Pressure Surface. Suction Surface is True

        Returns:
            float: Desired Vortex Angle [ rad ]
        """
        # We can directly evaluate for the angle
        angle = vortex_flow_angles(beta_m=beta_m, v_i=v_i, v_f=v_f, surface=surface)

        return angle

    def vortex_radius(self, M_star: float) -> float:
        """Function that Evaluates for the Fortex Flow Radius

        Args:
            M_star (float): Critical Velocity Ratio to which to evaluate the Vortex flow radius [ N.D. ]

        Returns:
            float: Normalised Vortex Flow Radius R* [ m ]
        """

        r_star = 1 / M_star

        return r_star

    def transition_zone(
        self,
        v_i: float,
        v_f: float,
        k_max: int,
        fluid: IdealGas,
        alpha: float = 0,
        reverse_x: bool = False,
    ) -> np.ndarray[np.float64]:
        """Function that returns a transition arc between an initial and final prandtl Meyer Angle.

        Args:
            v_i (float): Initial Prandtl Meyer Angle for the Transition [ rad ]
            v_f (float): Final Prandtl Meyer Angle for the Transition [ rad ]
            k_max (int): Number of Points to Consider for the Transition [ N.D. ]
            fluid (IdealGas): Gas Flowing Throught Turbine Stage.
            alpha (float, optional): Rotation Angle to Displace Transform Transition Arct [ rad ]. Defaults to 0 rad.
            reverse_x (bool): Flag used for reversing the Final X co-ordinates. Defaults to False.

        Returns:
            np.ndarray[np.float64]: Array of co-ordinate points that can be used for plotting.
        """

        # We use our utility function to generate our MoC transition points
        x_pnt, y_pnt = method_of_characteristics(
            k_max=k_max,
            v_i=v_i,
            v_l=v_f,
            gamma=fluid.gamma,
            alpha_l_i=alpha,
            reverse_x=reverse_x,
        )

        # We then combine them together to make a singular array
        coords = np.column_stack((x_pnt, y_pnt))

        return coords

    def vortex_zone(
        self, alpha: tuple[float, float], r_star: float, k_max: int
    ) -> np.ndarray[np.float64]:
        """Function that evaluates for the co-ordinates for the vortex "circular" arc zones of the turbine.

        Args:
            alpha (tuple[float, float]): Tuple containing the starting and ending arc locations (alpha_i, alpha_f) [ rad, rad ]
            R_star (float): Normalised Radius of the Vortex Zone [ N.D. ]
            k_max (float): Number of Points to discretise Curve.

        Returns:
            np.ndarray[np.float64]: Co-ordinate array of all points in the Vortex zone.
        """
        # First think we do is extract our alpha angles
        alpha_i, alpha_f = alpha

        # We then generate a linspace array based on the number of points to consider
        alpha_array = np.linspace(alpha_i, alpha_f, k_max)

        # We then generate our x and y co-ordinates using standard triginometry
        x_array = r_star * np.sin(alpha_array)
        y_array = r_star * np.cos(alpha_array)

        # We then combined these to make our co-ordinates
        coords = np.column_stack((x_array, y_array))

        return coords

    def straight_zone(
        self, beta_m: float, initial_coord: tuple[float, float], x_f: float
    ) -> np.ndarray[np.float64]:
        """Function that evaluates for the straight inlet/outlet sections of the turbine, based on the inlet/outlet angle and co-ord positions

        Args:
            beta_m (float): Blade Metal Angle
            initial_coord (tuple[float, float]): Initial Point for the Staight Section ( x, y ) [ N.D., N.D. ]
            x_f (float): Final x-co-ordiante of the straight section [ N.D. ]

        Returns:
            np.ndarray[np.float64]: Array containing the co-ordinates of the straight line segments (x, y)
        """
        # We firstly unpack our initial co-ordiante location
        x_o, y_o = initial_coord

        # We then convert the beta angle into a gradient
        m = np.tan(beta_m)

        # We can then solve for the offset (y_o) to fit a line to that point and gradient
        c = y_o - m * x_o

        # We can now solve for our new point
        y_f = x_f * m + c

        # We can package this into an array and deliver it to the user
        return np.array([[x_o, y_o], [x_f, y_f]])

    def blade_spacing_TE(
        self,
        d_alpha: float,
        alpha: tuple[float, float],
        v_o: float,
        beta_o: float,
        deflection: float,
        v_s: tuple[float, float],
        k_max: int,
        fluid: IdealGas,
    ) -> float:
        """Function that evaluates for the trailiing edge blad spacing.

        Args:
            d_alpha (float): Circular Arc Offset [ rad ]
            alpha (tuple[float, float]): Tuple of the initial arc angles of the upper and lower
            v_o (float): Outlet Prantdl Meyer Angle
            deflection (float): Deflection Angle [ rad ]
            v_s (tuple[float, float]): Tuple of the Upper Surface and Lower Surface Prandtl Number of Vortex sections [ rad, rad ]
            k_max (int): Number of points to discretise in transition zones
            fluid (IdealGas): Ideal Gas Fluid Object

        Returns:
            float: Blade Spacing at the Trailing Edge of the component
        """
        # We firslty unpack our arrays
        v_u, v_l = v_s
        alpha_u, alpha_l = alpha

        # We then adjust our blade spacing
        alpha_u += d_alpha
        alpha_l -= d_alpha

        # We can now evaluate for our transition zones
        lower = self.transition_zone(
            v_i=v_o + d_alpha,
            v_f=v_l,
            k_max=k_max,
            fluid=fluid,
            alpha=-alpha_l,
            reverse_x=False,
        )
        upper = self.transition_zone(
            v_i=v_o - 2 * d_alpha,
            v_f=v_u,
            k_max=k_max,
            fluid=fluid,
            alpha=alpha_u,
            reverse_x=True,
        )

        # We need to transform these transition zones accordingly.
        arrays = [upper, lower]
        c = np.cos(self._deflection)
        s = np.sin(self._deflection)

        for arr in arrays:
            x = arr[:, 0].copy()
            y = arr[:, 1].copy()

            arr[:, 0] = c * x - s * y
            arr[:, 1] = s * x + c * y

        # We additionally evaluate for our straight sections
        straight = self.straight_zone(
            beta_m=-beta_o + deflection + 2 * d_alpha,
            initial_coord=(lower[-1]),
            x_f=upper[-1, 0],
        )

        # We then evaluate for blade spacing
        g = -(straight[-1] - upper[-1])

        return g[1]

    def edge_radius(self, beta_m: float, t: float) -> float:
        """Function that Evaluates for the required LE/TE radius, based on a leading edge pitch displacement thickness and blade metal angle.

        Args:
            beta_m (float): Blade Metal Angle at Leading Edge/ Trailing Edge [ rad ]
            t (float): Leading Edge/Trailing Edge thickness [ N.D. ]

        Returns:
            float: Radius of the Leading Edge / Trailing Edge [ N.D. ]
        """
        # We make an absolute displacement.
        beta_m = abs(beta_m)

        # We can directly solve for the leading edge and trailing edge
        r = (t * np.cos(beta_m)) / 2

        return r

    def edge_centroid(
        self, edge_coord: tuple[float, float], beta_m: float, r: float
    ) -> tuple[float, float]:
        """Function that evaluates for the centroid location for leading and trailing edges

        Args:
            edge_coord (tuple[float, float]): Co-ordiantes of the selected edge [ N.D., N.D. ]
            beta_m (float): Blade Metal Angle [ rad ]
            r (float): Leading/Trailing Edge Radius [ N.D. ]

        Returns:
            tuple[float,float]: Tuple of the leading/trailing edge centroid co-ordinates
        """
        # We firstly un-pack our edge co-ordinates
        x, y = edge_coord

        # We then evaluate for the offsets
        dx = r * np.sin(beta_m)
        dy = r * np.cos(beta_m)

        # We then shift our co-ordiantes and return it back to the user
        edge_center = (x + dx, y + dy)

        return edge_center

    def plot_passage(self):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(
            self._vortex_l_coord[:, 0],
            self._vortex_l_coord[:, 1],
            label="Lower Circular Arc",
        )
        ax.plot(
            self._vortex_u_coord[:, 0],
            self._vortex_u_coord[:, 1],
            label="Upper Circular Arc",
        )
        ax.plot(self._l_i_coord[:, 0], self._l_i_coord[:, 1], label="Inlet Lower")
        ax.plot(self._u_i_coord[:, 0], self._u_i_coord[:, 1], label="Inlet Upper")
        ax.plot(self._l_o_coord[:, 0], self._l_o_coord[:, 1], label="Outlet Lower")
        ax.plot(self._u_o_coord[:, 0], self._u_o_coord[:, 1], label="Outlet Upper")
        ax.plot(
            self._straight_i_coord[:, 0],
            self._straight_i_coord[:, 1],
            label="Inlet Line",
        )
        ax.plot(
            self._straight_o_coord[:, 0],
            self._straight_o_coord[:, 1],
            label="Outlet Line",
        )
        # ax.plot(
        #     self._straight_o2_coord[:, 0],
        #     self._straight_o2_coord[:, 1],
        #     label="Outlet Line",
        # )

        # ax.set_xlim([2, 3])

        ax.set_ylabel(r"y* ($\frac{y}{r^*}$)")
        ax.set_xlabel(r"x* ($\frac{x}{r^*}$)")

        ax.set_aspect("equal")
        ax.set_title(f"Normalised Flow Passage Profile")

        ax.legend()
        plt.show()

        return

    def plot_blade(self):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(
            self._vortex_l_coord[:, 0],
            self._vortex_l_coord[:, 1] + self._g_star + self._t,
            label="Lower Circular Arc",
        )
        ax.plot(
            self._vortex_u_coord[:, 0],
            self._vortex_u_coord[:, 1],
            label="Upper Circular Arc",
        )
        ax.plot(
            self._l_i_coord[:, 0],
            self._l_i_coord[:, 1] + self._g_star + self._t,
            label="Inlet Lower",
        )
        ax.plot(self._u_i_coord[:, 0], self._u_i_coord[:, 1], label="Inlet Upper")
        ax.plot(
            self._l_o_coord[:, 0],
            self._l_o_coord[:, 1] + self._g_star + self._t,
            label="Outlet Lower",
        )
        ax.plot(self._u_o_coord[:, 0], self._u_o_coord[:, 1], label="Outlet Upper")
        ax.plot(
            self._straight_i_coord[:, 0],
            self._straight_i_coord[:, 1] + self._g_star + self._t,
            label="Inlet Line",
        )
        ax.plot(
            self._straight_o_coord[:, 0],
            self._straight_o_coord[:, 1] + self._g_star + self._t,
            label="Outlet Line",
        )

        ax.plot(
            self._LE_coord[:, 0],
            self._LE_coord[:, 1] + self._g_star,
            label="LE",
        )
        ax.plot(
            self._TE_coord[:, 0],
            self._TE_coord[:, 1] + self._g_star,
            label="TE",
        )

        ax.set_ylabel(r"y* ($\frac{y}{r^*}$)")
        ax.set_xlabel(r"x* ($\frac{x}{r^*}$)")

        ax.set_aspect("equal")
        ax.set_title(f"Normalised Flow Passage Profile")

        # ax.legend()
        plt.show()

        return

    def plot_lete(self):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(
            self._LE_coord[:, 0],
            self._LE_coord[:, 1] + self._g_star,
            label="LE",
        )
        ax.plot(
            self._TE_coord[:, 0],
            self._TE_coord[:, 1] + self._g_star,
            label="TE",
        )

        ax.plot(
            self._straight_i_coord[:, 0],
            self._straight_i_coord[:, 1] + self._g_star + self._t,
            label="Inlet Line",
        )
        ax.plot(
            self._straight_o_coord[:, 0],
            self._straight_o_coord[:, 1] + self._g_star + self._t,
            label="Outlet Line",
        )

        ax.set_ylabel(r"y* ($\frac{y}{r^*}$)")
        ax.set_xlabel(r"x* ($\frac{x}{r^*}$)")

        ax.set_aspect("equal")
        ax.set_title(f"Normalised Flow Passage Profile")

        ax.legend()
        plt.show()

    def generate_turbine_profile(self) -> None:
        """This Function Performs the Geometry generation for the Turbine Blade Profile"""
        # We firstly solve for our Prandtl Meyer Numebers
        self.prantl_meyer()

        # We then get our circular section parameters
        self.circular_section()

        # We solve for the upper maximum and lower minimum mach numbers to prevent flow speeration
        self.M_u_max()
        self.M_l_min()
        print(f"Bing")

        # We solve for the maximum inlet mach number (at turbine inlet) before the turbine would unstarts
        self.M_i_max()

        # We solve for the key geometries of the turbine
        self.generate_transitions()

        # We discretise the circulate sections
        self.discretise_circular(50)

        # We define the straight line segments on the upper surface pretending their is no
        self.straight_line_segments()

        # We can now get the blade spacing (G*) and chord length (C*) to calculate our solidity
        self.get_g_star()
        self.get_c_star()
        self.get_solidity()
        print(f"Initial Solidity: {self._sigma}")

        # Finally we can generate our blade
        self.generate_blade()

        # We can re_caclulate the solidity
        self._sigma = self._c_star / self._g_star

        print(f"Final Solidity: {self._sigma}")

        return

    def r_star_param(self):

        # We specify the number of steps for our integration

        INTEGRAL_NUMBER = 100  # TODO: Fix this magic number
        GAMMA = self._fluid.gamma

        self._wf_parameter = inv_mass_flow(
            M_star_l=self._M_l_star,
            M_star_u=self._M_u_star,
            gamma=GAMMA,
            n=INTEGRAL_NUMBER,
            mass_flow=self._m_dot,
        )

        # We meed to now calculate the total density assuming ideal gas
        density_i_total = self._fluid.get_density()

        # Calculating thet local speed of sound of the ideal gas.
        a_i_total = self._fluid.speed_of_sound()

        # Based on the weight-flow parameter, we can compute the sonic radius
        self._r_star_a = r_star(
            wf_parameter=self._wf_parameter,
            h=self._h,
            a_total_inlet=a_i_total,
            rho_total_inlet=density_i_total,
        )

        return

    def get_c_star(self) -> None:
        """This function gets the blades chord length for the generated profile"""

        # We take the first point and last point for the transition region
        self._c_star = self._xlkt_ol[-1] - self._xlkt_il[-1]

        return

    def get_solidity(self) -> None:
        """This function gets the blade solidity for the generated profile"""
        self.get_g_star()
        self.get_c_star()

        self._sigma = self._c_star / self._g_star

        return

    def generate_surface_maps(self) -> None:
        """This function generates the surface maps for the s_position of the upper and lower surface of the blade profiles

        - We calculate the straight line distance between the points
        """
        ########## Upper Surface ##########

        # We create our upper surface arrays
        df_upper = self.generate_upper_xy()

        # We get the start points
        first_point = df_upper.iloc[0]

        df_upper["s"] = ""

        # Setting the Initial Position
        df_upper.loc[0, "s"] = 0

        for index, row in df_upper.iterrows():

            if index == 0:
                continue

            x_1_u = row["x"]
            y_1_u = row["y"]

            s_0_u = df_upper.loc[index - 1, "s"]
            x_0_u = df_upper.loc[index - 1, "x"]
            y_0_u = df_upper.loc[index - 1, "y"]

            df_upper.loc[index, "s"] = s_0_u + np.sqrt(
                (x_0_u - x_1_u) ** 2 + (y_0_u - y_1_u) ** 2
            )

        ########## Lower Surface ##########

        # We create our upper surface arrays
        df_lower = self.generate_lower_xy()

        # We get the start points
        first_point = df_lower.iloc[0]

        df_lower["s"] = ""

        # Setting the Initial Position
        df_lower.loc[0, "s"] = 0

        for index, row in df_lower.iterrows():

            if index == 0:
                continue

            x_1_l = row["x"]
            y_1_l = row["y"]

            s_0_l = df_lower.loc[index - 1, "s"]
            x_0_l = df_lower.loc[index - 1, "x"]
            y_0_l = df_lower.loc[index - 1, "y"]

            df_lower.loc[index, "s"] = s_0_l + np.sqrt(
                (x_0_l - x_1_l) ** 2 + (y_0_l - y_1_l) ** 2
            )

        # We can now normalise our "s" array from 0 to 1.
        df_upper["s"] = df_upper["s"] / df_upper["s"].max()
        df_lower["s"] = df_lower["s"] / df_lower["s"].max()

        # We can now prepare our interpolation arrays for the x and y positions of the upper and lower surface using linear interpolation.
        self._s_x_u = interp1d(df_upper["s"], df_upper["x"], kind="linear")
        self._s_y_u = interp1d(df_upper["s"], df_upper["y"], kind="linear")

        # We can now do the same process for the lower surface
        self._s_x_l = interp1d(df_lower["s"], df_lower["x"], kind="linear")
        self._s_y_l = interp1d(df_lower["s"], df_lower["y"], kind="linear")

        return

    def get_upper_surface_position(self, s_u: float) -> tuple:
        """Gets the co-ordinates of the upper surface position

        Args:
            s_u (float): Streamline Length along upper surface [0 - 1]. 0 is LE, 1 is TE

        Returns:
            tuple: x,y position of the surface
        """
        # We can get the upper surface position
        x_u = self._s_x_u(s_u)
        y_u = self._s_y_u(s_u)

        return (x_u, y_u)

    def get_lower_surface_position(self, s_l: float) -> tuple:
        """Gets the co-ordinates of the lower surface position

        Args:
            s_l (float): Streamline Length along lower surface [0 - 1]. 0 is LE, 1 is TE

        Returns:
            tuple: x,y position of the surface
        """
        # We can get the lower surface position
        x_l = self._s_x_l(s_l)
        y_l = self._s_y_l(s_l)

        return (x_l, y_l)

    def get_distance_upper(
        self, s_u: float, s_l: float, shift_flag: bool = False
    ) -> float:
        """Gets the distance between two points on the upper lower surface

        Args:
            s_u (float): Position along the Lower Surface [0 - 1]. 0 is LE, 1 is TE
            s_l (float): Position along the Lower Surface [0 - 1]. 0 is LE, 1 is TE
            shift_flag (bool): Whether we shift the position of the y for matching.

        Returns:
            float: Distance between the upper and lower surface blade profiles
        """
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)

        if shift_flag:
            y_l += self._g_star * self._sf

        dl = ((x_u - x_l) ** 2 + (y_u - y_l) ** 2) ** (1 / 2)

        return dl

    def get_distance_lower(
        self, s_l: float, s_u: float, shift_flag: bool = False
    ) -> float:
        """Gets the distance between two points on the upper lower surface

        Args:
            s_u (float): Position along the Lower Surface [0 - 1]. 0 is LE, 1 is TE
            s_l (float): Position along the Lower Surface [0 - 1]. 0 is LE, 1 is TE
            shift_flag (bool): Whether we shift the position of the y for matching.

        Returns:
            float: Distance between the upper and lower surface blade profiles
        """
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)

        if shift_flag:
            y_l += self._g_star * self._sf

        dl = ((x_u - x_l) ** 2 + (y_u - y_l) ** 2) ** (1 / 2)

        return dl

    def fit_surface_circle(
        self, s_u: float, s_l: float, offset: float = 0
    ) -> dict[str, float]:
        """This function fits a cirle based on the upper and lower surface points

        Args:
            s_u (float): Upper Surface Point
            s_l (float): Lower Surface Point
            offset (float, optional): Offset for lower surface positioning

        Returns:
            dict[str, float]: Dictionary of the camber center co-ordinates and the spacing
        """
        if s_l >= 0.999 or s_u >= 0.999:
            ds = -0.001
        else:
            ds = 0.001

        # We get the upper surface position and gradient
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)
        x_up, y_up = self.get_upper_surface_position(s_u=s_u + ds)

        dy_dx_u = (y_up - y_u) / (x_up - x_u)

        # We get the lower surface position and gradient
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)
        x_lp, y_lp = self.get_lower_surface_position(s_l=s_l + ds)

        # y_l += offset
        # y_lp += offset

        dy_dx_l = (y_lp - y_l) / (x_lp - x_l)

        # We can now convert this into an angle
        phi_u = math.atan2(-1, dy_dx_u)
        phi_l = math.atan2(-1, dy_dx_l)

        # We can then solve for our radius
        r = (x_u - x_l) / (np.cos(phi_u) + np.cos(phi_l))

        # We can then solve for our centroid accordingly
        x_0 = x_l + r * np.cos(phi_l)
        y_0 = y_l + r * np.sin(phi_l)

        dic = {"x": x_0, "y": y_0, "dl": 2 * r}

        return dic

    def error_circle(self, s_u: float, s_l: float) -> float:
        """This function gets the radius error between two points on a curve when fitting a circle

        Args:
            s_u (float): Upper Position
            s_l (float): Lower Position

        Returns:
            float: Error in Radii
        """
        if s_l >= 0.999 or s_u >= 0.999:
            ds = -0.001
        else:
            ds = 0.001

        # We get the upper surface position and gradient
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)
        x_up, y_up = self.get_upper_surface_position(s_u=s_u + ds)

        dy_dx_u = (y_up - y_u) / (x_up - x_u)

        # We get the lower surface position and gradient
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)
        x_lp, y_lp = self.get_lower_surface_position(s_l=s_l + ds)

        # y_l += offset
        # y_lp += offset

        dy_dx_l = (y_lp - y_l) / (x_lp - x_l)

        # We can now convert this into an angle
        phi_u = math.atan2(-1, dy_dx_u)
        phi_l = math.atan2(-1, dy_dx_l)

        # We can then solve for our radii
        r_x = (x_u - x_l) / (np.cos(phi_u) + np.cos(phi_l))
        r_y = (y_u - y_l) / (np.sin(phi_u) + np.sin(phi_l))

        return abs(r_x - r_y)

    def camber_position(self, s_l: float) -> dict[str, float]:
        """Gets the location of the camber Line

        Args:
            s_l (float): Surface Position on lower surface

        Returns:
            dict[str, float]: Dictionary of the x and y position of the camber, along with the thickness at his point
        """
        # We get the tangent location on the upper surface
        x_u, y_u = self.get_upper_surface_position(s_u=s_l)
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)

        # We can get the mid point between these two points accordingly.
        dic = {"x": (x_u + x_l) / 2, "y": (y_u + y_l) / 2}

        dic["x_1"] = x_u
        dic["y_1"] = y_u

        dic["x_2"] = x_l
        dic["y_2"] = y_l

        return dic

    def get_cad_shift(self) -> float:
        """This function gets the offset for the CAD

        Returns:
            float: CAD Offset
        """

        df = self.generate_xy()

        offset = -df["y"].min() - 0.5 * (df["y"].max() - df["y"].min())

        return offset

    def passage_position(self, s_l: float) -> dict[str, float]:
        """Gets the location of the passage center line for machining

        Args:
            s_l (float): Surface Position on lower surface

        Returns:
            dict[str, float]: Dictionary of the x and y position of the camber, along with the thickness at his point
        """

        # We get the upper surface position
        bnds = Bounds(lb=0, ub=1)
        res = minimize(
            self.get_distance_upper,
            s_l * 0.8,
            args=(s_l, True),
            bounds=bnds,
            method="Nelder-Mead",
        )

        s_u = res.x

        # We then need to get the x and y positions of the upper and lower surfaces
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)

        x_l, y_l = self.get_lower_surface_position(s_l=s_l)

        y_l += self._g_star * self._sf

        dic = {}

        dic["x_1"] = x_u
        dic["y_1"] = y_u

        dic["x_2"] = x_l
        dic["y_2"] = y_l

        return dic

    def get_passage_spacing(self) -> float:
        """This function evaluates for the blade Passage spacing for machining

        Returns:
            float: Blade Passage Spacing
        """
        up = self.get_upper_surface_position(s_u=0.5)
        down = self.get_lower_surface_position(s_l=0.5)

        y_u = (up / self._sf - self._g_star)[1]

        y_d = (down / self._sf)[1]

        s = y_d - y_u

        return s * self._sf

    def generate_blade(self):

        # Now Shifting all our array points for the upper blade profiling accordingly

        self._y_i_line_up = self._y_i_line  # + self._g_star
        self._y_o_line_up = self._y_o_line  # self._g_star

        # Shifting transition points
        self._ylkt_iu_up = self._ylkt_iu - self._g_star
        self._ylkt_ou_up = self._ylkt_ou - self._g_star

        # Shifting Circular Points
        self._y_u_array_up = self._y_u_array - self._g_star

        return

    def plot_circles(self):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array, self._y_l_array)
        ax.plot(self._x_u_array, self._y_u_array)
        ax.set_aspect("equal")
        plt.show()

    def plot_transition(self):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._xlkt_il, self._ylkt_il, label="Inlet Lower")
        ax.plot(self._xlkt_iu, self._ylkt_iu, label="Inlet Upper")
        ax.plot(self._xlkt_ol, self._ylkt_ol, label="Outlet Lower")
        ax.plot(self._xlkt_ou, self._ylkt_ou, label="Outlet Upper")
        ax.legend()
        ax.set_aspect("equal")
        plt.show()

    def generate_cfd(self, t_lead: float, t_trail: float) -> dict[str, pd.DataFrame]:
        """Function that generates the domain for a CFD Based simulation of the blade passage

        Args:
            t_lead (float): Upstream axial distance to include in domain (mm)
            t_trail (float): Downstream axial distance to include in domain (mm)

        Returns:
            dict[str, pd.DataFrame]: Dictionary containing the dataframes of the upper surface, lower surface and inlet/oulet points
        """

        ################################################## Upper Surface ##################################################

        x_array_upper = np.array([])
        y_array_upper = np.array([])
        z_array_upper = np.array([])

        # We plot the Leading Edge Array,
        x_array_upper = np.append(x_array_upper, self._x_i_line_sf[::-1])
        y_array_upper = np.append(y_array_upper, self._y_i_line_sf_cfd[::-1])

        # The then go to the inlet upper Transition
        x_array_upper = np.append(x_array_upper, (self._xlkt_iu_sf)[-2:1:-1])
        y_array_upper = np.append(y_array_upper, (self._ylkt_iu_sf_cfd)[-2:1:-1])

        # # We then do the inlet Upper Circular element
        x_array_upper = np.append(x_array_upper, self._x_u_array_sf)
        y_array_upper = np.append(y_array_upper, self._y_u_array_sf_cfd)

        # # We then do the outlet Upper Transition
        x_array_upper = np.append(x_array_upper, self._xlkt_ou_sf[1:-1])
        y_array_upper = np.append(y_array_upper, self._ylkt_ou_sf_cfd[1:-1])

        # # We plote the Trailing Edge Array,
        x_array_upper = np.append(x_array_upper, self._x_o_line_sf)
        y_array_upper = np.append(y_array_upper, self._y_o_line_sf_cfd)

        z_array_upper = np.zeros(x_array_upper.size)

        df_upper = pd.DataFrame(
            data={"x": x_array_upper, "y": y_array_upper, "z": z_array_upper}
        )

        ################################################# Lower Surface #################################################

        x_array_lower = np.array([])
        y_array_lower = np.array([])
        z_array_lower = np.array([])

        x_array_lower = np.append(x_array_lower, self._x_i_line_sf[-1])
        y_array_lower = np.append(y_array_lower, self._y_i_line_sf[-1])

        # We then do the inlet lower transition element
        x_array_lower = np.append(x_array_lower, (self._xlkt_il_sf)[1:-2][::-1])
        y_array_lower = np.append(y_array_lower, (self._ylkt_il_sf)[1:-2][::-1])

        # We then do the lower circular element
        x_array_lower = np.append(x_array_lower, (self._x_l_array_sf)[::-1])
        y_array_lower = np.append(y_array_lower, (self._y_l_array_sf)[::-1])

        # We then do the outlet lower transition
        x_array_lower = np.append(x_array_lower, self._xlkt_ol_sf[-1:1:-1][::-1])
        y_array_lower = np.append(y_array_lower, self._ylkt_ol_sf[-1:1:-1][::-1])

        z_array_lower = np.zeros(x_array_lower.size)

        df_lower = pd.DataFrame(
            data={"x": x_array_lower, "y": y_array_lower, "z": z_array_lower}
        )

        ################################################# Upstream and Downstream Zones #################################################

        h_inner = t_lead / np.tan(self._beta_i)
        # TODO: Rest of the work

    def plot_normalised(self):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array, self._y_l_array)
        ax.plot(self._x_u_array, self._y_u_array_up)
        ax.plot(self._xlkt_il, self._ylkt_il, label="Inlet Lower")
        ax.plot(self._xlkt_iu, self._ylkt_iu_up, label="Inlet Upper")
        ax.plot(self._xlkt_ol, self._ylkt_ol, label="Outlet Lower")
        ax.plot(self._xlkt_ou, self._ylkt_ou_up, label="Outlet Upper")
        ax.plot(self._x_i_line, self._y_i_line_up)
        ax.plot(self._x_o_line, self._y_o_line_up)
        ax.set_ylabel(r"y* ($\frac{y}{r^*}$)")
        ax.set_xlabel(r"x* ($\frac{x}{r^*}$)")
        ax.set_aspect("equal")
        ax.set_title(f"Normalised Blade Profile")
        ax.legend()
        plt.show()

    def plot_scaled(self):

        if not hasattr(self, "_x_l_array_sf"):
            self.scale_coords(sf=self._r_star_a)
            scaling_text = r"$r*$"
        else:
            scaling_text = r"$sf$"
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array_sf * 1e3, self._y_l_array_sf * 1e3)
        ax.plot(self._x_u_array_sf * 1e3, self._y_u_array_sf * 1e3)
        ax.plot(
            self._xlkt_il_sf * 1e3,
            self._ylkt_il_sf * 1e3,
            label="Inlet Lower",
        )
        ax.plot(
            self._xlkt_iu_sf * 1e3,
            self._ylkt_iu_sf * 1e3,
            label="Inlet Upper",
        )
        ax.plot(
            self._xlkt_ol_sf * 1e3,
            self._ylkt_ol_sf * 1e3,
            label="Outlet Lower",
        )
        ax.plot(
            self._xlkt_ou_sf * 1e3,
            self._ylkt_ou_sf * 1e3,
            label="Outlet Upper",
        )
        ax.plot(self._x_i_line_sf * 1e3, self._y_i_line_sf * 1e3, label="Leading Edge")
        ax.plot(self._x_o_line_sf * 1e3, self._y_o_line_sf * 1e3, label="Trailing Edge")

        ax.set_ylabel(r"y (mm)")
        ax.set_xlabel(r"x (mm)")

        ax.set_aspect("equal")
        ax.set_title(f"Blade Profile Scaled by {scaling_text}")
        ax.legend()
        plt.show()

    def generate_upper_xy(self) -> pd.DataFrame:
        """Function that generates an x-y data frame of the co-ordinates of the upper surface

        Returns:
            pd.DataFrame: Data frame of profile co-ordinates of the Upper Surface
        """

        # We simply need to create a master x-array and y-array, create a pandas dataframe, then export as csv
        x_array = np.array([])
        y_array = np.array([])
        z_array = np.array([])

        # We plot the Leading Edge Array,
        x_array = np.append(x_array, self._x_i_line_sf[::-1])
        y_array = np.append(y_array, self._y_i_line_sf[::-1])

        # The then go to the inlet upper Transition
        x_array = np.append(x_array, (self._xlkt_iu_sf)[-2:1:-1])
        y_array = np.append(y_array, (self._ylkt_iu_sf)[-2:1:-1])

        # # We then do the inlet Upper Circular element
        x_array = np.append(x_array, self._x_u_array_sf)
        y_array = np.append(y_array, self._y_u_array_sf)

        # # We then do the outlet Upper Transition
        x_array = np.append(x_array, self._xlkt_ou_sf[1:-1])
        y_array = np.append(y_array, self._ylkt_ou_sf[1:-1])

        # # We plote the Trailing Edge Array,
        x_array = np.append(x_array, self._x_o_line_sf)
        y_array = np.append(y_array, self._y_o_line_sf)

        z_array = np.zeros(x_array.size)

        # We need to center in the y_axis- to do this, we will get the maximum and minimum value for the y, half it and shift accordingly.

        df = pd.DataFrame(data={"x": x_array, "y": y_array, "z": z_array})

        return df

    def generate_lower_xy(self) -> pd.DataFrame:
        """Function that generates an x-y data frame of the co-ordinates of the lower surface

        Returns:
            pd.DataFrame: Data frame of profile co-ordinates of the Lower Surface.
        """
        # We simply need to create a master x-array and y-array, create a pandas dataframe, then export as csv
        x_array = np.array([])
        y_array = np.array([])
        z_array = np.array([])

        # # # We then do the outlet lower transition
        x_array = np.append(x_array, self._xlkt_ol_sf[-1:1:-1])
        y_array = np.append(y_array, self._ylkt_ol_sf[-1:1:-1])

        # # We then do the lower circular element
        x_array = np.append(x_array, (self._x_l_array_sf)[::-1])
        y_array = np.append(y_array, (self._y_l_array_sf)[::-1])

        # # We then do the inlet lower transition element
        x_array = np.append(x_array, (self._xlkt_il_sf)[1:-2])
        y_array = np.append(y_array, (self._ylkt_il_sf)[1:-2])

        x_array = np.append(x_array, self._x_i_line_sf[-1])
        y_array = np.append(y_array, self._y_i_line_sf[-1])

        z_array = np.zeros(x_array.size)

        # We need to center in the y_axis- to do this, we will get the maximum and minimum value for the y, half it and shift accordingly.

        df = pd.DataFrame(
            data={"x": x_array[::-1], "y": y_array[::-1], "z": z_array[::-1]}
        )

        return df

    def generate_xy(self) -> pd.DataFrame:
        """Function that generates an x-y data frame of the co-ordinates of the turbine, that can be either plotted or used accordingly.

        Returns:
            pd.DataFrame: Dataframe of Profile Co-ordinates
        """

        # We simply need to create a master x-array and y-array, create a pandas dataframe, then export as csv
        x_array = np.array([])
        y_array = np.array([])
        z_array = np.array([])

        # We plot the Leading Edge Array,
        x_array = np.append(x_array, self._x_i_line_sf[::-1])
        y_array = np.append(y_array, self._y_i_line_sf[::-1])

        # The then go to the inlet upper Transition
        x_array = np.append(x_array, (self._xlkt_iu_sf)[-2:1:-1])
        y_array = np.append(y_array, (self._ylkt_iu_sf)[-2:1:-1])

        # # We then do the inlet Upper Circular element
        x_array = np.append(x_array, self._x_u_array_sf)
        y_array = np.append(y_array, self._y_u_array_sf)

        # # We then do the outlet Upper Transition
        x_array = np.append(x_array, self._xlkt_ou_sf[1:-1])
        y_array = np.append(y_array, self._ylkt_ou_sf[1:-1])

        # # We plote the Trailing Edge Array,
        x_array = np.append(x_array, self._x_o_line_sf)
        y_array = np.append(y_array, self._y_o_line_sf)

        # # # We then do the outlet lower transition
        x_array = np.append(x_array, self._xlkt_ol_sf[-2:1:-1])
        y_array = np.append(y_array, self._ylkt_ol_sf[-2:1:-1])

        # # We then do the lower circular element
        x_array = np.append(x_array, (self._x_l_array_sf)[::-1])
        y_array = np.append(y_array, (self._y_l_array_sf)[::-1])

        # # We then do the inlet lower transition element
        x_array = np.append(x_array, (self._xlkt_il_sf)[1:-2])
        y_array = np.append(y_array, (self._ylkt_il_sf)[1:-2])

        x_array = np.append(x_array, self._x_i_line_sf[-1])
        y_array = np.append(y_array, self._y_i_line_sf[-1])

        z_array = np.zeros(x_array.size)

        # We need to center in the y_axis- to do this, we will get the maximum and minimum value for the y, half it and shift accordingly.
        # y_array = y_array - y_array.min() - 0.5 * (y_array.max() - y_array.min())

        df = pd.DataFrame(data={"x": x_array * 1e3, "y": y_array * 1e3, "z": z_array})

        return df

    def get_xy_mean_line(self) -> pd.DataFrame:
        """Function that gets the mean line between the upper and lower surfaces of the turbine

        Returns:
            pd.DataFrame: Dataframe containing the mean_line co-ordinates of the upper and lower surface
        """

        ####################################### Upper Surface Profile #######################################

        x_array_u = np.array([])
        y_array_u = np.array([])

        # We plot the Leading Edge Array,
        x_array_u = np.append(x_array_u, self._x_i_line_sf[::-1])
        y_array_u = np.append(y_array_u, self._y_i_line_sf[::-1])

        # The then go to the inlet upper Transition
        x_array_u = np.append(x_array_u, (self._xlkt_iu_sf)[-2:1:-1])
        y_array_u = np.append(y_array_u, (self._ylkt_iu_sf)[-2:1:-1])

        # We then do the inlet Upper Circular element
        x_array_u = np.append(x_array_u, self._x_u_array_sf)
        y_array_u = np.append(y_array_u, self._y_u_array_sf)

        # We then do the outlet Upper Transition
        x_array_u = np.append(x_array_u, self._xlkt_ou_sf[1:-1])
        y_array_u = np.append(y_array_u, self._ylkt_ou_sf[1:-1])

        # We plote the Trailing Edge Array,
        x_array_u = np.append(x_array_u, self._x_o_line_sf)
        y_array_u = np.append(y_array_u, self._y_o_line_sf)

        ####################################### Lower Surface Profile #######################################

        x_array_l = np.array([])
        y_array_l = np.array([])

        # We then do the outlet lower transition
        x_array_l = np.append(x_array_l, self._xlkt_ol_sf[-1:1:-1])
        y_array_l = np.append(y_array_l, self._ylkt_ol_sf[-1:1:-1])

        # We then do the lower circular element
        x_array_l = np.append(x_array_l, (self._x_l_array_sf)[::-1])
        y_array_l = np.append(y_array_l, (self._y_l_array_sf)[::-1])

        # We then do the inlet lower transition element
        x_array_l = np.append(x_array_l, (self._xlkt_il_sf)[1:-2])
        y_array_l = np.append(y_array_l, (self._ylkt_il_sf)[1:-2])

        x_array_l = np.append(x_array_l, self._x_i_line_sf[-1])
        y_array_l = np.append(y_array_l, self._y_i_line_sf[-1])

        # We can now inverse the order of the array now
        x_array_l = x_array_l[::-1]
        y_array_l = y_array_l[::-1]

        ######################################### Normalising the distances #########################################

        y_array = np.append(y_array_l, y_array_u)

        # We need to center in the y_axis- to do this, we will get the maximum and minimum value for the y, half it and shift accordingly.
        y_array_l = y_array_l - y_array.min() - 0.5 * (y_array.max() - y_array.min())
        y_array_u = y_array_u - y_array.min() - 0.5 * (y_array.max() - y_array.min())

        ######################################### Interpolation #########################################

        df = pd.DataFrame(data={"x": x_array * 1e3, "y": y_array * 1e3, "z": z_array})

    def M_i_max(self):
        """
        This function solves for the critical inlet mach number for the profile to ensure the geometry can be started succesfully.

        In supersonic turbines, it is critical that the geometry can be started up under low flow conditions.

        This particularly important at startup conditions as the relative inlet velocities are at their highest levels (due to blade speeds being low).

        We can solve for the maximum acceptable inlet mach number/ prantl meyer angle and see if self-starting is possible for the turbine.

        """

        # First we need to solve for our k_star_max based on our upper and lowe mach numbers we have selected
        INTEGRAL_NUMBER = 100  # TODO: Fix this magic number
        GAMMA = self._fluid.gamma

        self._k_star = k_star_max(
            M_star_l=self._M_l_star,
            M_star_u=self._M_u_star,
            gamma=GAMMA,
            n=INTEGRAL_NUMBER,
        )

        # Now we know our k_star max, we can figure out what our Q and C are for the turbine accordingly

        self._Q_blade = Q(
            M_star_l=self._M_l_star,
            M_star_u=self._M_u_star,
            gamma=GAMMA,
            n=INTEGRAL_NUMBER,
        )

        self._C_blade = C(
            M_star_l=self._M_l_star,
            M_star_u=self._M_u_star,
            gamma=GAMMA,
            n=INTEGRAL_NUMBER,
            k_star=self._k_star,
        )

        # From this, we can figure out what our shock pressure ratio is for the gas.

        self._p_rat = shock_pressure_rat(Q=self._Q_blade, C=self._C_blade)

        # Now that we know our shock pressure ratio, we can now calculate back our M_star_i_max value

        self._M_i_star_max = M_i_star_max(p_rat=self._p_rat, gamma=GAMMA)

        # We can back calculate for what this mach number

        self._M_i_max = inv_M_star(gamma=GAMMA, M_star=self._M_i_star_max)

        self._v_i_max = prandtl_meyer(GAMMA, self._M_i_star_max)

        return self._M_i_max

    def M_u_max(self):
        """
        This function solves for the maximum upper surface mach number inorder to prevent flow seperation
        """
        GAMMA = self._fluid.gamma

        self._M_u_star_max = M_star_u_max(M_star_o=self._M_o_star, gamma=GAMMA)

        self._M_u_max = inv_M_star(gamma=GAMMA, M_star=self._M_u_star_max)

        return

    def M_l_min(self):
        """
        This function solves for the minimum lower surface mach number to avoid flow seperation of the gas
        """
        GAMMA = self._fluid.gamma

        self._M_l_star_min = M_star_l_min(m_star_i=self._M_i_star, gamma=GAMMA)

        self._M_l_min = inv_M_star(gamma=GAMMA, M_star=self._M_l_star_min)

        return

    def scale_coords(self, sf: float) -> None:
        """This function scales the geometry, based on a scaling factor for the geometry (either R_star_a or a chord based scale factor)

        Args:
            sf (float): Scale Factor for the Geometry
        """

        self._sf = sf

        self._x_l_array_sf = self._x_l_array * sf
        self._y_l_array_sf = self._y_l_array * sf

        self._x_u_array_sf = self._x_u_array * sf
        self._y_u_array_sf = self._y_u_array_up * sf
        self._y_u_array_sf_cfd = self._y_u_array * sf

        self._xlkt_il_sf = self._xlkt_il * sf
        self._ylkt_il_sf = self._ylkt_il * sf

        self._xlkt_ol_sf = self._xlkt_ol * sf
        self._ylkt_ol_sf = self._ylkt_ol * sf

        self._xlkt_iu_sf = self._xlkt_iu * sf
        self._ylkt_iu_sf = self._ylkt_iu_up * sf
        self._ylkt_iu_sf_cfd = self._ylkt_iu * sf

        self._xlkt_ou_sf = self._xlkt_ou * sf
        self._ylkt_ou_sf = self._ylkt_ou_up * sf
        self._ylkt_ou_sf_cfd = self._ylkt_ou * sf

        self._x_i_line_sf = self._x_i_line * sf
        self._y_i_line_sf = self._y_i_line_up * sf
        self._y_i_line_sf_cfd = self._y_i_line * sf

        self._x_o_line_sf = self._x_o_line * sf
        self._y_o_line_sf = self._y_o_line_up * sf
        self._y_o_line_sf_cfd = self._y_o_line * sf

        return

    def generate_mesh_upper(self, n: int = 1000) -> pd.DataFrame:
        """This function gets the upper surface mesh contour

        Args:
            n (int, optional): Number of Points on the Upper Surface. Defaults to 1000.

        Returns:
            pd.DataFrame: Contour of the Upper Surface
        """

        # We get the offset
        offset = self.get_cad_shift()

        self.generate_surface_maps()

        x_c = []
        y_c = []

        for x in np.linspace(0, 1, n):
            camber = self.camber_position(x)

            x_c.append(camber["x"])
            y_c.append(camber["y"])

        data = {
            "x": np.array(x_c) * 1e3,
            "y": (np.array(y_c) + self._t / 2) * 1e3 - offset,
            "z": np.zeros(np.array(x_c).size),
        }

        df = pd.DataFrame(data)

        return df

    def generate_mesh_lower(self, n: float = 1000) -> pd.DataFrame:
        """This function gets the lower surface mesh contour

        Args:
            n (float, optional): Number of Points on the Upper Surface. Defaults to 1000.

        Returns:
            pd.DataFrame: Contour of the Upper Surface
        """

        # We get the offset
        offset = self.get_cad_shift()

        self.generate_surface_maps()

        x_c = []
        y_c = []
        z_c = []

        for x in np.linspace(0, 1, n):
            camber = self.camber_position(x)

            x_c.append(camber["x"])
            y_c.append(camber["y"])

        data = {
            "x": (np.array(x_c)) * 1e3,
            "y": (np.array(y_c) - self._t / 2) * 1e3 - offset,
            "z": np.zeros(np.array(x_c).size),
        }

        df = pd.DataFrame(data)

        return df


class SymmetricFiniteEdge(SupersonicProfile):
    """This object represents a symmetric finite edge supersonic profile w/boundary layer correction.

    We assume

    Args:
        SupersonicProfile (_type_): Infinite Edge Supersonic Profile Object
    """

    def __init__(
        self,
        beta_ei: float,
        beta_i: float,
        M_i: float,
        M_u: float,
        M_l: float,
        m_dot: float,
        h: float,
        t_g_rat: float,
        g_expand: float,
        le_angle: float,
        fluid: IdealGas,
    ):
        # First thing to do is get the acutal entry conditions for the turbine based on the farfiedl
        ANGLE_CONVERSION = np.pi / 180

        # We need to firstly solve for what the Mach number at the inlet of the turbine will be
        M_e = get_m_e(
            t_g_rat=t_g_rat,
            beta_e=beta_ei * ANGLE_CONVERSION,
            beta_i=beta_i * ANGLE_CONVERSION,
            M_i=M_i,
            gamma=fluid.gamma,
        )

        # We can then Initialise for our turbine
        super().__init__(
            beta_i=beta_ei,
            beta_o=-beta_ei,
            M_i=M_e,
            M_o=-M_e,
            M_u=M_u,
            M_l=M_l,
            m_dot=m_dot,
            h=h,
            fluid=fluid,
        )

        # We then log the information as it relates to the finite leading edges and trailing edges - along with boundary layer computations.
        self._t_g_rat = t_g_rat
        self._g_expand = g_expand
        self._le_angle = le_angle * ANGLE_CONVERSION

        return

    def generate_turbine_profile(self) -> None:
        """This Function Performs the Geometry generation for the Turbine Blade Profile"""
        # We firstly solve for our Prandtl Meyer Numebers
        self.prantl_meyer()

        # We then get our circular section parameters
        self.circular_section()

        # We solve for the upper maximum and lower minimum mach numbers to prevent flow speeration
        self.M_u_max()
        self.M_l_min()
        print(f"Bing")

        # We solve for the maximum inlet mach number (at turbine inlet) before the turbine would unstarts
        self.M_i_max()

        # We solve for the key geometries of the turbine
        self.generate_transitions()

        # We discretise the circulate sections
        self.discretise_circular(50)

        # We define the straight line segments on the upper surface pretending their is no
        self.straight_line_segments()

        # We can now get the blade spacing (G*) and chord length (C*) to calculate our solidity
        self.get_g_star()
        self.get_c_star()
        self.get_solidity()

        # We can now generate the finite edge thickness
        self.generate_finite_edge()

        # We then further expand the blade spacing based on user input
        self.adjust_blade_spacing(b_factor=self._g_expand)

        # Finally we can generate our blade
        self.generate_blade()

        # We can re_caclulate the solidity
        self._sigma = self._c_star / self._g_star

        return

    def get_performance(self) -> dict[str, float]:
        """This function gets the key performance parameters as it relates to the turbine, namely "startability" and margin till flow seperation

        Returns:
            (dict): Dictionary of Performance Parameters
        """

        # Firstly we evaluate what our maximum possible upper surface mach number is along with minimum lower surface mach number
        self.M_u_max()
        self.M_l_min()

        # We check if we are in compliance
        if self._M_u_max < self._M_u:
            raise ValueError(
                f"Mach number too high on upper surface for flow seperation: {self._M_u} > {self._M_u_max}"
            )

        if self._M_l < self._M_l_min:
            raise ValueError(
                f"Mach number too low on lower surface for flow seperation: {self._M_l} < {self._M_l_min}"
            )

        # We now check if both of these are higher than the inlet conditions

        if self._M_u < self._M_i:
            raise ValueError(
                f"Mach Number too low on upper surface and is decelerating from inlet! {self._M_u} < {self._M_i}"
            )

        if self._M_l > self._M_i:
            raise ValueError(
                f"Mach Number too high on lower surface and is accelerating from inlet! {self._M_l} > {self._M_i}"
            )

        # We then evaluate for the maximum possible Inlet Mach Number
        self.M_i_max()

        # We check if we are in compliance
        if self._M_i_max < self._M_i:
            raise ValueError(
                f"Inlet Mach number exceeded Maximum For Starting: {self._M_i} > {self._M_i_max}"
            )

        # We then assemble our array and dictionary of key parameters
        dic = {
            "M_u_max": self._M_u_max,
            "M_u_margin": (self._M_u_max - self._M_u) / self._M_u,
            "M_l_min": self._M_l_min,
            "M_l_margin": (self._M_l - self._M_l_min) / self._M_l,
            "M_e_max": (self._M_i_max),
            "M_e_margin": (self._M_i_max - self._M_i) / self._M_i,
        }

        return dic

    def size_geometry(
        self, D_m: float, N: int | None = None, b: float | None = None
    ) -> dict[str, float]:
        """This function is used for scaling the geometry based on the solidity we use for the blade design.

        Args:
            D_m (float): Mean Diameter of the Turbine
            N (int | None): Number of Blades at the meanline Diameter (m). Defaults to None
            b (float | None): Blade Chord Length (m). Defaults to None
        """

        if N is not None and b is not None:
            raise ValueError("Problem is Over defined!")

        elif b is not None:
            # We calcualte the blade spacing based on the chord length
            self._t = b / self._sigma

            self._N = round(D_m / self._t)

        elif N is not None:
            # We calculate the chord length based on the
            self._N = N
        else:
            raise ValueError(
                "Not enough information. Require either the blade chord length or number of profiles at the mean diameter"
            )

        # We can now solve for the chord length based on the number of blades we intend to have
        self._t = np.pi * D_m / self._N

        # From this, we can figure out what the blade chord should be

        self._b = self._sigma * self._t

        # We can then generate our scaled components based on the distances fromt he last points.
        b_normal = self._x_o_line[-1] - self._x_i_line[-1]

        self._sf = self._b / b_normal

        # We can scale accordingly for both the x and y axis for all the key dimensions
        self.scale_coords(sf=self._sf)

        # We can finally return a dictionary containing the key Properties of the turbine
        dic = {
            "sigma": self._sigma,
            "b": self._b,
            "t": self._t,
        }

        return dic

    def generate_finite_edge(self, N: int = 100) -> None:
        """This function generates a finite leading edge for the Turbine

        Args:
            N (int, optional): Number of Points to discretise on each line segment. Defaults to 100.

        Raises:
            ValueError: Leading Edge Angle is Protruding
        """

        # We need to figure out what the blade thickness is
        self._t = self._t_g_rat * self._g_star

        # We need to fogire out now what the x displacement is based on the angle
        theta = self._le_angle + self._beta_i

        append_flag = True

        if theta == np.pi / 2:
            self._dx_edge = 0

            self._dy_edge = self._t

        elif theta > np.pi / 2:
            raise ValueError("Leading Edge Angle leading to a protruding LE shape!")

        elif self._le_angle == 0:
            # No leading edge Angle
            append_flag = False

        else:
            self._dx_edge = self._t / (
                np.tan(self._beta_i + self._le_angle) - np.tan(self._beta_i)
            )

            self._dy_edge = self._dx_edge * np.tan(self._beta_i)

        if append_flag:
            # We can generate our intersection points

            x_i_new = self._x_i_line[-1] + self._dx_edge
            y_i_new = self._y_i_line[-1] + self._dy_edge

            self._x_i_line = np.append(
                np.linspace(self._x_i_line[0], x_i_new, N),
                np.linspace(x_i_new, self._x_i_line[-1], N)[1:],
            )
            self._y_i_line = np.append(
                np.linspace(self._y_i_line[0], y_i_new, N),
                np.linspace(y_i_new, self._y_i_line[-1] - self._t, N)[1:],
            )

            x_o_new = self._x_o_line[-1] - self._dx_edge
            y_o_new = self._y_o_line[-1] + self._dy_edge

            self._x_o_line = np.append(
                np.linspace(self._x_o_line[0], x_o_new, N),
                np.linspace(x_o_new, self._x_o_line[-1], N)[1:],
            )
            self._y_o_line = np.append(
                np.linspace(self._y_o_line[0], y_o_new, N),
                np.linspace(y_o_new, self._y_o_line[-1] - self._t, N)[1:],
            )

        # Update Leading Edge Thickness
        self._g_star += self._t

        return

    def adjust_blade_spacing(self, b_factor: float) -> None:
        """This function adjusts the blade spacing, along with all with all the co-ordinates of the system

        Args:
            b_factor (float): Blade Spacing Factor (% change in geometry)
        """

        # Firstly we adjust our g_star value based on the b_factor and get the displacement distance

        dy = self._g_star * b_factor

        self._g_star = self._g_star * (1 + b_factor)

        # We can then decrease all the suction surface co-ordinates accordingly.
        self._y_i_line -= dy
        self._y_o_line -= dy

        # Shifting transition points
        self._ylkt_iu -= dy

        # Shifting Circular Points
        self._y_u_array -= dy

        return
