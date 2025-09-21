import numpy as np


def mol2vol(val, molar_volume):
    # convert mole fractions to volume fractions
    molar_volume = np.asarray(molar_volume)
    z_arr = np.array([val, 1-val]) if type(val) in [float, np.float64, np.float32] else np.asarray(val)
    vbar = z_arr @ molar_volume
    if not isinstance(vbar, np.ndarray):
        vbar = np.array([vbar])
    v_arr = z_arr * molar_volume / vbar.reshape(vbar.shape[0], 1)
    v_arr = v_arr.reshape(z_arr.shape)
    return float(v_arr[0]) if type(val) in [float, np.float64, np.float32] else v_arr

