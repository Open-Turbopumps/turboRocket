"""This file contains the mechanical components used to simulate the turbopump performance"""

class MechanicalLosses:
    """This object represents the Mechanical Losses of the System"""

    def __init__(self, m_bearing: list[float], n_bearing: list[int]) -> None:
        """Constructor for the Bearing Object

        Args:
            m_bearing (list[float]): Array of Selected Bearing Moments (Nm)
            n_bearing (list[float]): Array of Selected Bearings and Number of each type
        """

        self._m_bearing = m_bearing
        self._n_bearing = n_bearing

        if len(self._m_bearing) != len(self._n_bearing):
            raise ValueError(
                "Number of Bearings listed dow not equal the number of bearings"
            )

        return

    def get_torque(self) -> float:
        # This function gets the total induced torque by the bearing system
        T = 0

        i = 0
        for n in self._n_bearing:
            T += self._m_bearing[i] * n

            i += 1

        return T
