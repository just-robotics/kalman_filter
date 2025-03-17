import numpy as np


def kalman_filter(sigma_ksi, sigma_n, y_enc, y_gps, est_e_last):
    est_e = est_e_last + sigma_ksi / (2 * sigma_n**2) *\
    (np.sqrt(sigma_ksi**2 + 4 * sigma_n**2) - sigma_ksi) *\
    ((y_enc - y_gps) - est_e_last)
    return est_e