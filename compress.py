import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import rowdec
from cued_sf2_lab.laplacian_pyramid import rowint
from cued_sf2_lab.laplacian_pyramid import bpp
from scipy.optimize import minimize_scalar
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.jpeg import (
    jpegenc, jpegdec, quant1, quant2, huffenc, huffdflt, huffdes, huffgen)

# Initialise three images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

# fig, axs = plt.subplots(1, 3)
# plot_image(lighthouse, ax=axs[0])
# plot_image(bridge, ax=axs[1])
# plot_image(flamingo, ax=axs[2])
# plt.show()

Xl = lighthouse - 128.0
Xb = bridge - 128.0
Xf = flamingo - 128.0

### 6-laplacian-pyramid ###
def py4enc(X, h):
    # Initialize Y list
    Y_list = []
    current_X = X

    # Iteration to get a list of Y
    for _ in range(4):
        # Step 1: Decimate X in row and col
        lowpass_X = rowdec(current_X, h)
        lowpass_X = rowdec(lowpass_X.T, h).T

        # Step 2: Interpolate lowpass_X back to current image size
        interpolated_X = rowint(lowpass_X, 2*h)
        interpolated_X = rowint(interpolated_X.T, 2*h).T

        # Step 3: Compute the highpass image
        highpass_image = current_X - interpolated_X

        # Store the highpass image into list Y
        Y_list.append(highpass_image)
        
        current_X = lowpass_X
    
    # The final tiny lowpass image
    tiny_lowpass_image = current_X

    return Y_list[0], Y_list[1], Y_list[2], Y_list[3], tiny_lowpass_image

def py4dec(Y0, Y1, Y2, Y3, X4, h):
    # Initialize the list for reconstructed images
    lowpass_image = []
    current_image = X4

    #Interate to reconstruct the 4 layers
    for i, highpass_image in enumerate([Y3, Y2, Y1, Y0]):
        # Step 1: Interpolate the current lowpass image
        interpolated_image = rowint(current_image, 2*h)
        interpolated_image = rowint(interpolated_image.T, 2*h).T

        # Step 2: Add the highpass image
        next_image = interpolated_image + highpass_image

        # Store the reconstructed lowpass image
        lowpass_image.append(next_image)

        # Update the current image for the next iteration
        current_image = next_image

    return lowpass_image

# Energy measurement of an impulse in each layer
def measure_energy_contribution(h, image_size=256, pyramid_layers=5):
    energies = []
    impulse_amplitude = 100

    for layer in range (pyramid_layers):
        # Create test pyramid with zero images
        Y0c, Y1c, Y2c, Y3c, X4c = [np.zeros((image_size >> i, image_size >> i)) for i in range(pyramid_layers)]
        test_layers = [Y0c, Y1c, Y2c, Y3c, X4c]

        # Place an impulse in the center of the current layer
        layer_center = test_layers[layer].shape[0] // 2
        test_layers[layer][layer_center, layer_center] = impulse_amplitude

        # Reconstruct the image
        Z3c, Z2c, Z1c, Z0c = py4dec(*test_layers, h)

        # Measure the total energy in the reconstructed image Z0
        energy = np.sum(Z0c**2)
        energies.append(energy)

    return energies

### 7-the-discrete-cosine-transform (DCT) ###
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

### 9-the-discrete-wavelet-transform (DWT) ###
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

def quantdwt(Y, dwtstep):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtent: an array of shape `(3, n+1)` containing the entropies
    """
    n = dwtstep.shape[1] - 1
    Yq = Y.copy()
    dwtent = np.zeros((3, n+1))
    height, width = Y.shape

    for i in range(n+1):        
        mid_row, mid_col = height // 2, width // 2
        Yq[:mid_row, mid_col:] = quantise(Y[:mid_row, mid_col:], dwtstep[0, i]) # top right
        Yq[mid_row:, :mid_col] = quantise(Y[mid_row:, :mid_col], dwtstep[1, i]) # bottom left
        Yq[mid_row:, mid_col:] = quantise(Y[mid_row:, mid_col:], dwtstep[2, i]) # bottom right
        dwtent[0, i] = bpp(Yq[:mid_row, mid_col:]) * Yq[:mid_row, mid_col:].size
        dwtent[1, i] = bpp(Yq[mid_row:, :mid_col]) * Yq[mid_row:, :mid_col].size
        dwtent[2, i] = bpp(Yq[mid_row:, mid_col:]) * Yq[mid_row:, mid_col:].size
        height = mid_row
        width = mid_col

    Yq[:mid_row, :mid_col] = quantise(Y[:mid_row, :mid_col], dwtstep[0, n])
    return Yq, dwtent

# Energy measurement of an impulse in each layer
def measure_energy_contribution1(image_size=256, layers=3):
    energies = np.zeros((3, layers + 1))
    impulse_amplitude = 100

    # Iterate to find impulse centres
    right_edge = image_size  # initialise the right edge value
    d = image_size // 4  # initialise the distance of centre to the right edge
    for layer in range(layers + 1):
        # Initialize Yt and Zt
        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        # Implement impulse
        Yt[d][right_edge - d] = impulse_amplitude # Top right
        m = image_size // (2**layers)
        for _ in range(layers+1):
            Yt[:m, :m] = idwt(Yt[:m, :m])
            m = 2 * m
        energies[0][layer] = np.sum(Yt**2)

        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        Yt[right_edge - d][d] = impulse_amplitude # Botton left
        m = image_size // (2**layers)
        for _ in range(layers+1):
            Yt[:m, :m] = idwt(Yt[:m, :m])
            m = 2 * m
        energies[1][layer] = np.sum(Yt**2)

        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        Yt[right_edge - d][right_edge - d] = impulse_amplitude # Botton right
        m = image_size // (2**layers)
        for _ in range(layers+1):
            Yt[:m, :m] = idwt(Yt[:m, :m])
            m = 2 * m
        energies[2][layer] = np.sum(Yt**2)

        # Update right edge and d
        right_edge = right_edge // 2
        d = d // 2
        
    return energies

# DCT
C8 = dct_ii(8)
Yl = colxfm(colxfm(Xl, C8).T, C8).T
step = 10
rise1 = step * 1.5
Yq = quantise(Yl, step, rise1)
N = 8
Yr = regroup(Yq, N)

# Total number of bits using dctbpp
dctbpp_bits = dctbpp(Yr, N)
print(f'Total number of bits using dctbpp: {dctbpp_bits:2f}')

Z = colxfm(colxfm(Yq.T, C8.T).T, C8.T)
dct_rms_error = np.std(Xl - Z)
print(f'MSE for DCT is: {dct_rms_error}')

# Huffman coding
qstep = 10
vlc, hufftab = jpegenc(Z, qstep)
Z_jpec = jpegdec(vlc, qstep)
jpeg_rms_error = np.std(Xl - Z_jpec)
print(f'MSE for jpeg is: {jpeg_rms_error}')

fig, ax = plt.subplots()
plot_image(Z_jpec, ax=ax)
plt.show()

# C4 = dct_ii(4)
# Yl4 = colxfm(colxfm(Xl, C4).T, C4).T
# Yl4x2 = colxfm(colxfm(Yl4, C4).T, C4).T