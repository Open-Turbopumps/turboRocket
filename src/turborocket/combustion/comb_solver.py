"""
This file contains the objects that are used for solving for combustion conditions of the gas:

    - A custom combustion solver has been written using cantera, to allow for more accurate combustion modelling.
    - This solver should have comparable results to RocketCEA, however it will discount condensed fractions (solid)
      In the evaluation of the specific heat capacity and specific heat ratios of the gasses, which is critical to know.
"""

import cantera as ct
from turborocket.fluids.fluids import IdealGas
import numpy as np
from scipy.interpolate import interpn
import pandas as pd


class CombustionCantera:
    """Object representing the combustion solving object"""

    def __init__(
        self,
        fuel: str,
        oxidiser: str,
        species_file: str | None = None,
        look_up: bool = False,
        look_up_file: str | None = None,
    ) -> None:
        """Constructor for the Combustion Object

        Args:
            fuel (str): Molecular Name of the Fuel
            oxidiser (str): Molecular Name of the Oxidiser
            species_file (str): Name for Species YAML file for Cantera
            look_up (bool): Look_up flag for the solution procedure
            look_up_file (str, optional): Look_up file for the combustion properties.
        """

        self._fuel = fuel
        self._oxidiser = oxidiser

        fu_present = False
        ox_present = False

        if species_file is None:
            species_file = "nasa_gas.yaml"

        # We then get the full species list and check if the fuel and oxidiser has been listed
        full_species = ct.Species.list_from_file(species_file)

        self._molecules = []

        for species in full_species:

            if species.name == self._fuel:
                fu_present = True

                comp = species.composition

                self._molecules += comp

            elif species.name == self._oxidiser:
                ox_present = True

                comp = species.composition

                self._molecules += comp

        # We then do error checking if the fuel or oxidiser is present
        if fu_present is False:
            raise ValueError(f"Invalid Fuel Name: {self._fuel}")

        elif ox_present is False:
            raise ValueError(f"Invalid Oxidiser Name: {self._oxidiser}")

        self.setup_gas_mixture(species_list=full_species)

        self._look_up = look_up
        self._look_up_file = look_up_file

        if self._look_up == True:
            self.generate_look_up()

    def setup_gas_mixture(self, species_list: list) -> None:
        """This function gets the combustion object,

        Args:
            species_list (list): List of Species
        """
        # We need to get a list of the molecules from the propellants we created
        specific = []
        for species in species_list:

            comp = species.composition

            # we get the keys
            if not any(item not in self._molecules for item in comp.keys()):
                specific.append(species)

        self._gas = ct.Solution(thermo="ideal-gas", species=specific)
        self._gcr = ct.Solution("graphite.yaml")

        self._mix_phases = [(self._gas, 1.0), (self._gcr, 0.0)]

    def generate_look_up(
        self,
        P_max: float = 60e5,
        P_min: float = 5e5,
        MR_max: float = 6,
        MR_min: float = 0.1,
        N: float = 100,
        frozen: bool = True,
    ) -> None:
        """This function generates a set of lookup interpolation functions for combustion properties

        Args:
            P_max (float, optional): Maximum Combustion Pressure (Pa). Defaults to 60e5.
            P_min (float, optional): Minimum Combustion Pressure (Pa). Defaults to 10e5.
            MR_max (float, optional): Maximum Mixture Ratio (n.d.). Defaults to 8.
            MR_min (float, optional): Minimum Mixture Ratio (n.d.). Defaults to 0.3.
            N (float, optional): Number of Points to Consider. Defaults to 50
            frozen (bool, optional): Frozen Composition Flag. Defaults to True
        """

        if self._look_up_file is not None:
            df = pd.read_csv(self._look_up_file)

            # We then need to extract our properties
            self._CP_array = df["Cp"].to_numpy()
            self._CV_array = df["Cv"].to_numpy()
            self._T_array = df["T"].to_numpy()
            self._R_array = df["R"].to_numpy()

            # We need to now now get the shape of the overarching mesh grid, which we can do by looking at unique values for the pressure and temperature
            P_N = df["P"].unique().size

            MR_N = df["MR"].unique().size

            # We then resize our arrays and generate our x and y arrays
            self._P_array = np.linspace(df["P"].min(), df["P"].max(), P_N)
            self._MR_array = np.linspace(df["MR"].min(), df["MR"].max(), MR_N)

            self._CP_array = self._CP_array.reshape(P_N, MR_N)
            self._CV_array = self._CV_array.reshape(P_N, MR_N)
            self._T_array = self._T_array.reshape(P_N, MR_N)
            self._R_array = self._R_array.reshape(P_N, MR_N)

        else:
            # We create our arrays
            self._P_array = np.linspace(P_min, P_max, N)
            self._MR_array = np.linspace(MR_min, MR_max, N)

            # We mesh grid these accordingly
            P_array, MR_array = np.meshgrid(self._P_array, self._MR_array)

            # We then setup our arrays of interest
            self._CP_array = np.zeros([N, N])
            self._CV_array = np.zeros([N, N])
            self._T_array = np.zeros([N, N])
            self._R_array = np.zeros([N, N])

            # We then iterate through these points
            for index, x in np.ndenumerate(P_array):

                result = self.solve_cantera(
                    Pcc=P_array[index], MR=MR_array[index], frozen=frozen
                )

                # We then append these accordingly
                self._CP_array[index] = result.get_cp()
                self._CV_array[index] = result.get_cp() / result.get_gamma()
                self._T_array[index] = result.get_temperature()
                self._R_array[index] = result.get_R()

                print(f"Index Number: {index}")

            # We finally create our csv file to export accordingly.
            data = {
                "P": P_array.flatten(),
                "MR": MR_array.flatten(),
                "Cv": self._CV_array.flatten(),
                "Cp": self._CP_array.flatten(),
                "T": self._T_array.flatten(),
                "R": self._R_array.flatten(),
            }

            df = pd.DataFrame(data=data)

            df.to_csv("combustion_date.csv")

        return

    def solve_cantera(
        self, Pcc: float, MR: float, T: float = 295, dt: float = 1, frozen: bool = True
    ) -> IdealGas:
        """Gets the Specific Heat Capacity of the Combustion Mixture

        Args:
            Pcc (float): Chamber Pressure of the Gas (Pa)
            MR (float): Mixture Ratio of Propellants

        Optional:
            T (float): Inlet Prop Conditions (K)
            dt (float): Temperature Change Increment used for Specific Heat Derivation
            frozen (bool): Frozen Composition Flag for the defintion of Thermodynamic Properties.

        Returns:
            float: Combustion Gas Object
        """

        # Define our combustion gas conditions and setting up our mixture
        self._gas.Y = {self._fuel: 1, self._oxidiser: MR}
        mix = ct.Mixture(self._mix_phases)

        # We setup our mixture temperature pressure
        mix.T = T
        mix.P = Pcc

        # We then equilibrate the conditions
        mix.equilibrate("HP")

        # We then get the gas conditions
        h_o = self._gas.enthalpy_mass
        d_o = self._gas.density_mass
        u_o = self._gas.int_energy_mass

        T_o = self._gas.T
        R = ct.gas_constant / self._gas.mean_molecular_weight

        if not frozen:

            # We then perturb the gas mixture
            self._gas.TP = (T_o + dt, Pcc)

            # We re-equilibriate the gas
            self._gas.equilibrate("TP")

            # We then get the new enthalpy
            h_1 = self._gas.enthalpy_mass

            # We can then shift the mixture to a point with a fixed density at the perturbed termperature
            self._gas.TD = (T_o + dt, d_o)

            # We equilibriate
            self._gas.equilibrate("TV")

            u_1 = self._gas.int_energy_mass

            # Thermodynamic Properties
            cp = (h_1 - h_o) / dt
            cv = (u_1 - u_o) / dt
            gamma = cp / cv

        else:
            # For a frozen case, we just directly pull the data from the cantera gas object

            cp = self._gas.cp_mass
            cv = self._gas.cv_mass
            gamma = cp / cv

        gas = IdealGas(name="exhaust", p=Pcc, t=T_o, gamma=gamma, R=R, cp=cp)

        return gas

    def get_thermo_prop(
        self, Pcc: float, MR: float, T: float = 295, dt: float = 1, frozen: bool = True
    ) -> IdealGas:
        """Function for getting the thermodynamic Properties of the Gas

        Args:
            Pcc (float): Chamber Pressure of the Gas (Pa)
            MR (float): Mixture Ratio of the Gas
            T (float, optional): Initial Temperature of the Propellants (K). Defaults to 295.
            dt (float, optional): Temperature Increment for Equilibrium solutions (K). Defaults to 1.
            frozen (bool, optional): Frozen species flag for thermodynamics. Defaults to True.

        Returns:
            IdealGas: Ideal Gas Object
        """
        Pcc = float(Pcc)
        MR = float(MR)

        if self._look_up:
            # We check if the pressure and mixture ratio are within the present look-up table bounds
            if Pcc >= self._P_array.min() and Pcc <= self._P_array.max():

                if MR >= self._MR_array.min() and MR <= self._MR_array.max():

                    # We perform a regression of all our points
                    cp = interpn(
                        (self._P_array, self._MR_array), self._CP_array.T, [Pcc, MR]
                    )
                    gamma = cp / interpn(
                        (self._P_array, self._MR_array), self._CV_array.T, [Pcc, MR]
                    )
                    R = interpn(
                        (self._P_array, self._MR_array), self._R_array.T, [Pcc, MR]
                    )
                    T = interpn(
                        (self._P_array, self._MR_array), self._T_array.T, [Pcc, MR]
                    )

                    gas = IdealGas(p=Pcc, t=T, gamma=gamma, R=R, cp=cp)

                else:
                    gas = self.solve_cantera(Pcc=Pcc, MR=MR, T=T, dt=dt, frozen=frozen)
            else:
                gas = self.solve_cantera(Pcc=Pcc, MR=MR, T=T, dt=dt, frozen=frozen)
        else:
            gas = self.solve_cantera(Pcc=Pcc, MR=MR, T=T, dt=dt, frozen=frozen)

        return gas
