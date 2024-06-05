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
from typing import Tuple
from SSIM import calculate_ssim

# Initialise three images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

Xl = lighthouse - 128.0
Xb = bridge - 128.0
Xf = flamingo - 128.0

def dctbpp(Yr, N):
    # Your code here
    h, w = Yr.shape
    total_bits = 0
    d = h//N
    for i in range(0, h, d):
        for j in range(0, w, d):
            Ys = Yr[i: i+d, j: j+d]
            total_bits += bpp(Ys) * (d ** 2)
    return total_bits

def nlevdwt(X, n):
    assert(n >= 1)
    m=256
    Y=dwt(X)
    for i in range(1, n):
        m = m//2
        Y[:m, :m] = dwt(Y[:m, :m])
    return Y

def nlevidwt(Y, n):
    m = 256 // 2 ** n
    Z = Y.copy()
    for i in range(n):
        m = m*2
        Z[:m, :m] = idwt(Z[:m, :m])
    return Z

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtent: an array of shape `(3, n+1)` containing the entropies
    """
    def quantise_and_get_entropy(Y_sub, dwtstep, k, i):
        q = dwtstep[k, i]
        Yq_sub = quantise(Y_sub, q)
        ent = bpp(Yq_sub) * Yq_sub.shape[0] * Yq_sub.shape[1]
        return Yq_sub, ent

    n = dwtstep.shape[1] - 1
    Yq = Y.copy()
    ent_M = np.zeros(Y.shape)
    dwtent = np.zeros(dwtstep.shape)
    
    m = 512
    for i in range(n):
        m = m//2
        
        # Top right
        Yq[m//2:m, 0:m//2], dwtent[0, i] = quantise_and_get_entropy(Yq[m//2:m, 0:m//2], dwtstep, 0, i)
        ent_M[m//2:m, 0:m//2] = dwtent[0, i]

        # Bottom Left
        Yq[0:m//2, m//2:m], dwtent[1, i] = quantise_and_get_entropy(Yq[0:m//2, m//2:m], dwtstep, 1, i)
        ent_M[0:m//2, m//2:m] = dwtent[1, i]

        # Bottom right
        Yq[m//2:m, m//2:m], dwtent[2, i] = quantise_and_get_entropy(Yq[m//2:m, m//2:m], dwtstep, 2, i)
        ent_M[m//2:m, m//2:m] = dwtent[2, i]

    # Top left
    Yq[0:m//2, 0:m//2], dwtent[0, n] = quantise_and_get_entropy(Yq[0:m//2, 0:m//2], dwtstep, 0, n)
    ent_M[0:m//2, 0:m//2] = dwtent[0, n]
    
    return Yq, dwtent

def get_step_ratios(layers):
    """Get equal MSE DWT step ratios."""
    image_size = 256
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

def dwt_rms_error(step, X, n):
    Y = nlevdwt(X, n)
    dwtstep = get_step_ratios(n) * step
    Yq, dwtent = quantdwt(Y, dwtstep)
    Z = nlevidwt(Yq, n)
    rms_error = np.std(X - Z)
    rmsX = np.std(X-quantise(X, 17))
    return abs(rms_error- rmsX)

def optimize_dwt_step_size(X, n):
    result = minimize_scalar(
        dwt_rms_error, 
        args=(X, n),
        bounds=(0.1, 50), 
        method='bounded'
        )
    return result.x

# # Can enter X or Xb or different pics, and different n levels
# n = 5
# X_test = Xl
# step_opt = optimize_dwt_step_size(X_test, n)
# # step = 25
# print(f'optimized reference step size is: {step_opt:.4f}')

# Y = nlevdwt(X_test, n)
# dwtstep = get_step_ratios(n) * step_opt
# Yq, dwtent = quantdwt(Y, dwtstep)
# Z = nlevidwt(Yq, n)
# total_bits = dwtent.sum()
# print(f'Total number of bits: {total_bits:.2f}')
# Total_bits_Xq = bpp(quantise(X_test, 17)) * X_test.size
# Compression_ratio_const_step_size = Total_bits_Xq / total_bits
# print(f'Compression Ratio Equal-MSE-Scheme: {Compression_ratio_const_step_size:.4f}')
# ssim_score = calculate_ssim(X_test, Z)
# print(f"SSIM between the images: {ssim_score:.4f}")

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# fig.suptitle('Comparison of compressed and original images')
# plot_image(X_test, ax=axs[0])
# axs[0].set(title='original')
# plot_image(Z, ax=axs[1])
# axs[1].set(title='compressed')
# plt.show()
