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
from jpeg_modi import jpegdec_lbt, jpegenc_lbt, jpegdec1, jpegenc1, jpegdec2, jpegenc2
from SSIM import calculate_ssim

# Initialise three images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')
compete4, _ = load_mat_img(img='SF2_competition_image_2024.mat', img_info='X')
compete3, _ = load_mat_img(img='SF2_competition_image_2023.mat', img_info='X')

Xl = lighthouse - 128.0
Xb = bridge - 128.0
Xf = flamingo - 128.0
X4 = compete4 - 128.0
X3 = compete3 - 128.0

def huffman_bits_gap(step, X, N, M):
    qstep = step
    vlc = jpegenc(X, qstep, N, M, opthuff=True, log=False)[0] # Change dct/lbt here
    total_bits = sum(vlc[:, 1])
    diff = total_bits - 40960.0 + 1424.0 + 5.0 # opthuff needs the extra 1424 bits
    # diff = total_bits - 40960.0 + 5.0
    if diff > 0:
        diff = diff * 100000
    return abs(diff)

def optimize_huffman_step_size(X, N, M):
    result = minimize_scalar(
        huffman_bits_gap, 
        args=(X, N, M),
        bounds=(2, 80), 
        method='bounded'
        )
    return result.x

# Decide which picture and N, M values
X_test = Xl
n = 8
m = 8
step_opt = optimize_huffman_step_size(X_test, n, m)
# print(step_opt)
# step_opt = 28

### For LBT tests
# vlc, hufftab = jpegenc_lbt(X_test, step_opt, n, m, opthuff=True)
# Z_lbt = jpegdec_lbt(vlc, step_opt, n, m, hufftab=hufftab)
# vlc, hufftab = jpegenc_lbt(X_test, step_opt, n, m)
# Z_lbt = jpegdec_lbt(vlc, step_opt, n, m)
# jpeg_rms_error = np.std(X_test - Z_lbt)
# print(f'MSE for jpeg is: {jpeg_rms_error:.4f}')
# ## Calculate SSIM
# ssim_score = calculate_ssim(X_test, Z_lbt)
# print(f"SSIM between the images: {ssim_score:.4f}")
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# fig.suptitle('Comparison of compressed and original images')
# plot_image(X_test, ax=axs[0])
# axs[0].set(title='original')
# plot_image(Z_lbt, ax=axs[1])
# axs[1].set(title='compressed')
# plt.show()

### For DCT tests
vlc, hufftab = jpegenc(X_test, step_opt, n, m, opthuff=True)
Z = jpegdec(vlc, step_opt, n, m, hufftab=hufftab)
# vlc, hufftab = jpegenc(X_test, step_opt, n, m)
# Z = jpegdec(vlc, step_opt, n, m)
jpeg_rms_error = np.std(X_test - Z)
## Calculate RMS Error
print(f'MSE for jpeg is: {jpeg_rms_error}')
## Calculate SSIM
ssim_score = calculate_ssim(X_test, Z)
print(f"SSIM between the images: {ssim_score:.4f}")
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle('Comparison of compressed and original images')
plot_image(X_test, ax=axs[0])
axs[0].set(title='original')
plot_image(Z, ax=axs[1])
axs[1].set(title='compressed')
plt.show()

# fig, ax = plt.subplots()
# plot_image(Z, ax=ax)
# plt.show()