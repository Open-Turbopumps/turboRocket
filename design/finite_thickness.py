from turborocket.profiling.Supersonic.supersonic_profile import SymmetricFiniteEdge
from turborocket.fluids.fluids import IdealGas
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    target_fluid = IdealGas(p=20e5, t=816.495, R=1305.355, gamma=1.3)

    super = SymmetricFiniteEdge(
        beta_ei=68.34,
        beta_i=68.34,  # Relative angles
        M_i=1.846,  # Relative Mach Numbers Nominally
        M_u=2,
        M_l=1.5,
        m_dot=0.0158,
        h=10e-3,
        t_g_rat=0.1,
        g_expand=0.1,
        le_angle=20,
        fluid=target_fluid,
    )

    super.generate_turbine_profile()
    super.get_performance()
    print(super.size_geometry(D_m=0.2, N=50))
    # super.plot_passage()
    # super.plot_normalised()
    # super.plot_scaled()

    df = super.generate_xy()
    
    #test
