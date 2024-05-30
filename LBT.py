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