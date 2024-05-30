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
from cued_sf2_lab.jpeg import (
    jpegenc, jpegdec, quant1, quant2, huffenc, huffdflt, huffdes, huffgen)

# Initialise three images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

Xl = lighthouse - 128.0
Xb = bridge - 128.0
Xf = flamingo - 128.0

def huffman_bits_gap(step, X, N):
    qstep = step
    vlc, hufftab = jpegenc(X, qstep, N, N)
    total_bits = sum(vlc[:, 1])
    return abs(total_bits - 40960.0)

def optimize_huffman_step_size(X, N):
    result = minimize_scalar(
        huffman_bits_gap, 
        args=(X, N),
        bounds=(0.1, 50), 
        method='bounded'
        )
    return result.x

step_opt = optimize_huffman_step_size(Xl, 4)
print(step_opt)
vlc, hufftab = jpegenc(Xl, step_opt, 4, 4)
Z_jpec = jpegdec(vlc, step_opt, 4, 4)
fig, ax = plt.subplots()
plot_image(Z_jpec, ax=ax)
plt.show()
jpeg_rms_error = np.std(Xl - Z_jpec)
print(f'MSE for jpeg is: {jpeg_rms_error}')