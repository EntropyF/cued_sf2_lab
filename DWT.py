import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
from scipy.optimize import minimize_scalar
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.dwt import dwt, idwt

# Initialise three images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

Xl = lighthouse - 128.0
Xb = bridge - 128.0
Xf = flamingo - 128.0

def nlevdwt(X, n):
    # your code here
    current_image = X.copy()
    m, w = X.shape
    for _ in range(n+1):
        current_image[:m, :m] = dwt(current_image[:m, :m])
        m = m // 2
    return current_image


def nlevidwt(Y, n):
    # your code here
    current_image = Y.copy()
    m, w = Y.shape
    m = m // (2**n)
    for _ in range(n+1):
        current_image[:m, :m] = idwt(current_image[:m, :m])
        m = 2 * m
    return current_image


# ---- EQUAL MSE

def get_step_ratios(image_size=256, layers=3):
    ratios = np.zeros((3, layers + 1))
    impulse_amplitude = 100

    # Iterate to find impulse centres
    edge = image_size  # initialise the right edge value
    d = image_size // 4  # initialise the distance of centre to the right edge
    for layer in range(layers):
        # Initialize Yt and Zt
        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        # Implement impulse
        Yt[edge - d, d] = impulse_amplitude # Top right
        Z = nlevidwt(Yt, layers)
        ratios[0][layer] =  1/ np.std(Z)

        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        Yt[d, edge - d] = impulse_amplitude # Bottom left
        Z = nlevidwt(Yt, layers)
        ratios[1][layer] =  1/ np.std(Z)

        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        Yt[edge - d, edge - d] = impulse_amplitude # Bottom right
        Z = nlevidwt(Yt, layers)
        ratios[2][layer] = 1/ np.std(Z)

        # Update right edge and d
        edge = edge // 2
        d = d // 2
        
    
    # Top left
    d = d * 2
    Yt = np.zeros((image_size, image_size))
    Yt[d, d] = impulse_amplitude
    Z = nlevidwt(Yt, layers)
    ratios[0, layers] = 1/ np.std(Z)
    ratios = ratios / ratios.max()
    ratios[1, layers] = None
    ratios[2, layers] = None

    return ratios
