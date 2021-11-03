import math
import numpy as np
# import xarray as xr
import obspy

# How many forward simulations will be ran?
SIM_NUM = 1

def generate_GRF(nx = 51, xmin = 0, xmax=1, ax=1, az=1, sigma=5, delta=0.1):
""" Generate GRF with von Karman covariance function
    Code by Jorge C. Castellanos
    
    Arguments:
        nx: grid size
        ax: x correlation 
        ay: y correlation
        sigma: std amplitude of perturbation 
        delta: spacing size
"""
    ny = nx
    ymin = xmin
    ymax = xmax
    # Generating the domain:
    x = np.linspace(-5e3, 5e3, nx)
    y = np.linspace(-5e3, 5e3, ny)

    xx, yy = np.meshgrid(x, y, indexing="ij")

    vp = 3000.0 * np.ones_like(xx)
    rho = 2200.0 * np.ones_like(xx)

    # -----------------------------------
    # Generating the random perturbations
    # -----------------------------------
    # The perturbation parameters are:
    #      Delta - element size
    #      ax - correlation length in the x-direction
    #      az - correlation length in the z-direction
    #      sigma - standard deviation
    
#     delta = 0.1
#     ax = 1
#     az = 1
#     sigma = 5

    M = 2 * ny
    N = 2 * nx

    dP = 2 * np.random.rand(N, M) - 1
    Y2 = np.fft.fft2(dP)

    kx1 = np.mod(1 / 2 + (np.arange(0, M)) / M, 1) - 1 / 2
    kx = kx1 * (2 * math.pi / delta)
    kz1 = np.mod(1 / 2 + (np.arange(0, N)) / N, 1) - 1 / 2
    kz = kz1 * (2 * math.pi / delta)
    [KX, KZ] = np.meshgrid(kx, kz)

    K_sq = KX**2 * ax ** 2 + KZ**2 * az**2
    P_K = (ax * az) / ((1 + K_sq)**(3 / 2))

    Y2 = Y2 * np.sqrt(P_K)
    dP_New = np.fft.ifft2(Y2)

    test = np.real(dP_New[0:N,0:M])
    test = sigma / np.std(test.flatten()) * test

    Mid_M = int(np.floor(M / 4))
    Mid_N = int(np.floor(N / 4))
    M = int(M / 2)
    N = int(N / 2)
    dV = test[Mid_N : Mid_N + N, Mid_M : Mid_M + M]

    vp = vp + (vp * (dV / 100))
    rho = rho + (rho * (dV / 100))

#     ds = xr.Dataset(
#             data_vars={
#                 "vp": (["x", "y"], vp),
#                 "rho": (["x", "y"], rho),
#             },
#             coords={"x": x, "y": y},
#           )

#     # saving the model:
#     ds.to_netcdf(OUTPUT_DIR + f"Models/Model_{_i}.nc")
    return vp