import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from scipy.interpolate import PchipInterpolator

import skrf as rf
import warnings
import os
from pathlib import Path

import serdespy as sdp

import stat_eye

# Setup plot saving (this will intercept plt.show() and save plots instead)
from plot_save_utils import setup_plot_saving
setup_plot_saving()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global variables dictionary to mimic MATLAB's global scope
g = {
    'pulse_signal': None,
    'f': None,
    'pulse_resp_ch': None,
    'ratio_oversampling': None,
    'ui': None,
    'os': None,
    'H_ch': None,
    'tx_launch_amp': None,
    'pulse_signal_length': None,
    'num_pre_cursor': None,
    'num_post_cursor': None,
    'rterm_source': None,
    'rterm_sink': None,
}

################### Functions #########################

def S_ser_r(R_s, f, zo):
    Z_series = R_s + np.zeros_like(f, dtype=complex)
    term_ser_s = np.zeros((len(f), 2, 2), dtype=complex)
    common_den = Z_series + 2 * zo
    term_ser_s[:, 0, 0] = Z_series / common_den
    term_ser_s[:, 0, 1] = 2 * zo / common_den
    term_ser_s[:, 1, 0] = 2 * zo / common_den
    term_ser_s[:, 1, 1] = Z_series / common_den
    return term_ser_s

def S_ser_l(L_s, f, zo):
    # Note: The term (000e-3*L_s)/100e-12 in the original code evaluates to 0.
    Z_series = 1j * 2 * np.pi * L_s * f
    term_ser_s = np.zeros((len(f), 2, 2), dtype=complex)
    common_den = Z_series + 2 * zo
    term_ser_s[:, 0, 0] = Z_series / common_den
    term_ser_s[:, 0, 1] = 2 * zo / common_den
    term_ser_s[:, 1, 0] = 2 * zo / common_den
    term_ser_s[:, 1, 1] = Z_series / common_den
    return term_ser_s

def S_shn_c(C_s, f, zo):
    Y_shunt = 1j * 2 * np.pi * C_s * f
    y0 = 1 / zo
    term_shn_s = np.zeros((len(f), 2, 2), dtype=complex)
    common_den = Y_shunt + 2 * y0
    term_shn_s[:, 0, 0] = -Y_shunt / common_den
    term_shn_s[:, 0, 1] = 2 * y0 / common_den
    term_shn_s[:, 1, 0] = 2 * y0 / common_den
    term_shn_s[:, 1, 1] = -Y_shunt / common_den
    return term_shn_s

def S_shn_rc(C_s, R_s, f, zo):
    Y_shunt = 1 / R_s + (1j * 2 * np.pi * C_s * f)
    y0 = 1 / zo
    term_shn_s = np.zeros((len(f), 2, 2), dtype=complex)
    common_den = Y_shunt + 2 * y0
    term_shn_s[:, 0, 0] = -Y_shunt / common_den
    term_shn_s[:, 0, 1] = 2 * y0 / common_den
    term_shn_s[:, 1, 0] = 2 * y0 / common_den
    term_shn_s[:, 1, 1] = -Y_shunt / common_den
    return term_shn_s

def S_shn_lc(C_s, M_s, f, zo):
    # Adding a small epsilon to avoid division by zero at f=0
    f_safe = f + 1e-12
    Z_shunt = (1j * 2 * np.pi * M_s) * f_safe + 1 / ((1j * 2 * np.pi * C_s) * f_safe)
    Y_shunt = 1 / Z_shunt
    y0 = 1 / zo
    term_shn_s = np.zeros((len(f), 2, 2), dtype=complex)
    common_den = Y_shunt + 2 * y0
    term_shn_s[:, 0, 0] = -Y_shunt / common_den
    term_shn_s[:, 0, 1] = 2 * y0 / common_den
    term_shn_s[:, 1, 0] = 2 * y0 / common_den
    term_shn_s[:, 1, 1] = -Y_shunt / common_den
    return term_shn_s
    
def S2ABCD(S, zo):
    S11 = S[:, 0, 0]
    S12 = S[:, 0, 1]
    S21 = S[:, 1, 0]
    S22 = S[:, 1, 1]
    
    A = ((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21)
    B = ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21) * zo
    C = ((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21) / zo
    D = ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)
    
    return np.vstack([A, B, C, D])

def cascABCD(ABCD1, ABCD2):
    A1, B1, C1, D1 = ABCD1[0, :], ABCD1[1, :], ABCD1[2, :], ABCD1[3, :]
    A2, B2, C2, D2 = ABCD2[0, :], ABCD2[1, :], ABCD2[2, :], ABCD2[3, :]

    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = C1 * B2 + D1 * D2
    
    return np.vstack([A, B, C, D])

def RC_SOURCE(r_s, c_s, f, zo):
    term_ser_s = S_ser_r(r_s, f, zo)
    ABCD_ser_s = S2ABCD(term_ser_s, zo)
    
    term_shn_s = S_shn_c(c_s, f, zo)
    ABCD_shn_s = S2ABCD(term_shn_s, zo)
    
    return cascABCD(ABCD_ser_s, ABCD_shn_s)

def TCOIL_SOURCE(r_s, c_die, c_esd, c_pin, L1, L2, km, f, zo):
    term_ser_s = S_ser_r(r_s, f, zo)
    ABCD_ser_s = S2ABCD(term_ser_s, zo)

    term_shn_s = S_shn_c(c_die, f, zo)
    ABCD_shn_s = S2ABCD(term_shn_s, zo)
    ABCD_s = cascABCD(ABCD_ser_s, ABCD_shn_s)

    M = km * np.sqrt(L1 * L2)
    L1_rem = L1 - M
    S_L1_s = S_ser_l(L1_rem, f, zo)
    ABCD_L1_s = S2ABCD(S_L1_s, zo)
    ABCD_s = cascABCD(ABCD_s, ABCD_L1_s)

    S_C_M = S_shn_lc(c_esd, M, f, zo)
    ABCD_C_M_s = S2ABCD(S_C_M, zo)
    ABCD_s = cascABCD(ABCD_s, ABCD_C_M_s)

    L2_rem = L2 - M
    S_L2_s = S_ser_l(L2_rem, f, zo)
    ABCD_L2_s = S2ABCD(S_L2_s, zo)
    ABCD_s = cascABCD(ABCD_s, ABCD_L2_s)

    S_cpin_s = S_shn_c(c_pin, f, zo)
    ABCD_cpin_s = S2ABCD(S_cpin_s, zo)
    
    return cascABCD(ABCD_s, ABCD_cpin_s)

def TCOIL_split_SOURCE(r_s, c_die, c_esd1, c_esd2, c_pin, L1, L2, L3, km, f, zo):
    term_ser_s = S_ser_r(r_s, f, zo)
    ABCD_ser_s = S2ABCD(term_ser_s, zo)

    term_shn_s = S_shn_c(c_die, f, zo)
    ABCD_shn_s = S2ABCD(term_shn_s, zo)
    ABCD_s = cascABCD(ABCD_ser_s, ABCD_shn_s)

    M12 = km * np.sqrt(L1 * L2)
    M23 = km * np.sqrt(L2 * L3)
    
    L1_rem = L1 - M12 
    S_L1_s = S_ser_l(L1_rem, f, zo)
    ABCD_L1_s = S2ABCD(S_L1_s, zo)
    ABCD_s = cascABCD(ABCD_s, ABCD_L1_s)

    S_C_M1 = S_shn_lc(c_esd1, M12, f, zo) #effect of M31 is not known yet
    ABCD_C_M_s1 = S2ABCD(S_C_M1, zo)
    ABCD_s = cascABCD(ABCD_s, ABCD_C_M_s1)

    L2_rem = L2 - M12 - M23
    S_L2_s = S_ser_l(L2_rem, f, zo)
    ABCD_L2_s = S2ABCD(S_L2_s, zo)
    ABCD_s = cascABCD(ABCD_s, ABCD_L2_s)

    S_C_M2 = S_shn_lc(c_esd2, M23, f, zo)
    ABCD_C_M_s2 = S2ABCD(S_C_M2, zo)
    ABCD_s = cascABCD(ABCD_s, ABCD_C_M_s2)

    L3_rem = L3 - M23
    S_L3_s = S_ser_l(L3_rem, f, zo)
    ABCD_L3_s = S2ABCD(S_L3_s, zo)
    ABCD_s = cascABCD(ABCD_s, ABCD_L3_s)

    S_cpin_s = S_shn_c(c_pin, f, zo)
    ABCD_cpin_s = S2ABCD(S_cpin_s, zo)
    
    return cascABCD(ABCD_s, ABCD_cpin_s)

def RC_LOAD(c_l, f, zo):
    term_shn_l = S_shn_c(c_l, f, zo)
    return S2ABCD(term_shn_l, zo)

def TCOIL_LOAD(c_die, c_esd, c_pin, L1, L2, km, f, zo):
    term_shn = S_shn_c(c_pin, f, zo)
    ABCD_shn = S2ABCD(term_shn, zo)

    M = km * np.sqrt(L1 * L2)

    L1_rem = L1 - M
    S_L1_l = S_ser_l(L1_rem, f, zo)
    ABCD_L1_l = S2ABCD(S_L1_l, zo)
    ABCD_l = cascABCD(ABCD_shn, ABCD_L1_l)

    S_C_M = S_shn_lc(c_esd, M, f, zo)
    ABCD_C_M_l = S2ABCD(S_C_M, zo)
    ABCD_l = cascABCD(ABCD_l, ABCD_C_M_l)

    L2_rem = L2 - M
    S_L2_l = S_ser_l(L2_rem, f, zo)
    ABCD_L2_l = S2ABCD(S_L2_l, zo)
    ABCD_l = cascABCD(ABCD_l, ABCD_L2_l)

    S_cdie_l = S_shn_c(c_die, f, zo)
    ABCD_cpin_l = S2ABCD(S_cdie_l, zo)
    
    return cascABCD(ABCD_l, ABCD_cpin_l)

def ABCD2TF(ABCD, zo, R_L):
    A, B, C, D = ABCD[0, :], ABCD[1, :], ABCD[2, :], ABCD[3, :]
    
    den = A + B / zo + C * zo + D
    S11 = (A + B / zo - C * zo - D) / den
    S21 = 2 / den
    # The following variables are calculated but not used in the original code's return logic
    # S12 = 2 * (A * D - B * C) / den
    # S22 = (-A + B / zo - C * zo + D) / den
    # Gamma_L = (R_L - zo) / (R_L + zo)
    # Gamma_in = S11 + (S12 * S21 * Gamma_L) / (1 - S22 * Gamma_L)
    # V2_V1_intermediate = (2 * R_L * S21) / ((zo + R_L) * (1 + Gamma_in))
    
    # The original MATLAB code overwrites V2_V1 with this final line
    V2_V1 = 1 / (A + B / R_L)
    return V2_V1, S21, S11

# Helper for interpolation
def impinterp(P, n):
    if n <= 1:
        return P
    x = np.arange(len(P))
    xi = np.arange(0, len(P), 1/n)
    interp_func = PchipInterpolator(x, P)
    return interp_func(xi)
    
# Helper for circular convolution
def cconv(a, b, n):
    return np.fft.ifft(np.fft.fft(a, n) * np.fft.fft(b, n)).real

def sdd21(sp):
    # scikit-rf Network object s-parameters are indexed [freq, port1, port2]
    # Ports are 0-indexed: 1->0, 2->1, 3->2, 4->3
    return 0.5 * (sp[:, 1, 0] - sp[:, 1, 2] - sp[:, 3, 0] + sp[:, 3, 2])

# Placeholder for a function not provided in the original code
def frd_imp(H, f_rad):
    # This function likely calculates the impulse response from frequency domain data
    # Assuming H is complex frequency response and f_rad is angular frequency
    # We can get the time-domain impulse response via IFFT.
    imp = np.fft.irfft(H)
    # The sampling frequency is determined by the frequency step
    f_hz = f_rad / (2 * np.pi)
    if len(f_hz) > 1:
        Fs_ntwk = 2 * f_hz[-1] # Nyquist
    else:
        Fs_ntwk = 1.0
    return imp, Fs_ntwk

def gen_channel(**kwargs):
    p = {
        'fnyq': 0, 'df': 0, 'r_s': 0, 'c_die_s': 0, 'L1_s': 0, 'c_esd1_s': 0, 'L2_s': 0,
        'c_esd2_s': 0,'L3_s': 0, 'c_pad_s': 0, 'km_s': 0, 'c_pad_l': 0, 'L1_l': 0, 'c_esd_l': 0,
        'L2_l': 0, 'c_die_l': 0, 'km_l': 0, 'r_l': 0, 'die_s': '', 'pkg_s': '',
        'die_l': '', 'pkg_l': '', 'ch': '', 's_tcoil': 0, 'l_tcoil': 0, 's_tcoil_split': 0, 'l_tcoil_split': 0,
        'die_s_portswap': 0, 'pkg_s_portswap': 0, 'die_l_portswap': 0,
        'pkg_l_portswap': 0, 'ch_portswap': 0
    }
    p.update(kwargs)

    data = {'s4p': {}, 'h': {}, 'hf': {}, 'zo': {}}

    # Read S-Params
    for id_ in ['ch', 'die_s', 'pkg_s', 'die_l', 'pkg_l']:
        if p[id_]:
            try:
                network = rf.Network(p[id_])
                data['s4p'][id_] = network.s
                data['hf'][id_] = network.f
                data['zo'][id_] = network.z0[0,0] # Assuming uniform impedance

                if p[f'{id_}_portswap']:
                    s = data['s4p'][id_]
                    s[:, [1, 2], :] = s[:, [2, 1], :]
                    s[:, :, [1, 2]] = s[:, :, [2, 1]]
                    data['s4p'][id_] = s

                if p['fnyq'] == 0: p['fnyq'] = data['hf'][id_][-1]
                if p['df'] == 0: p['df'] = data['hf'][id_][1] - data['hf'][id_][0]
            except Exception as e:
                print(f"Could not read or process s-parameter file: {p[id_]}\n{e}")
                return None, None, None, None

    # Derive SISO frequency response data
    if not p['ch']:
        zo = 50
        f = np.arange(0, 70e9 + 1e6, 1e6)
    else:
        zo = data['zo']['ch']
        f = np.arange(0, p['fnyq'] + p['df'], p['df'])
    
    # Source Termination
    if p['s_tcoil']:
        if p['s_tcoil_split']:
            ABCD_source = TCOIL_split_SOURCE(p['r_s'], p['c_die_s'], p['c_esd1_s'], p['c_esd2_s'], p['c_pad_s'], p['L1_s'], p['L2_s'], p['L3_s'], p['km_s'], f, zo)
        else:
            ABCD_source = TCOIL_SOURCE(p['r_s'], p['c_die_s'], p['c_esd1_s']+p['c_esd2_s'], p['c_pad_s'], p['L1_s'], p['L2_s'], p['km_s'], f, zo)
    else:
        c_total_s = p['c_die_s'] + p['c_esd1_s'] + p['c_esd2_s'] + p['c_pad_s']
        ABCD_source = RC_SOURCE(p['r_s'], c_total_s, f, zo)
    
    H_term_s, S21_s, S11_s = ABCD2TF(ABCD_source, zo, zo)
    data['h']['term_s'] = H_term_s

    # Load Termination
    if p['l_tcoil']:
        if p['l_tcoil_split']:
            ABCD_l = TCOIL_LOAD(p['c_die_l'], p['c_esd_l'], p['c_pad_l'], p['L1_l'], p['L2_l'], p['km_l'], f, zo)
        else:
            ABCD_l = TCOIL_LOAD(p['c_die_l'], p['c_esd_l'], p['c_pad_l'], p['L1_l'], p['L2_l'], p['km_l'], f, zo)
    else:
        c_total_l = p['c_pad_l']
        ABCD_l = RC_LOAD(c_total_l, f, zo)
    
    H_term_l, S21_l, S11_l = ABCD2TF(ABCD_l, zo, p['r_l'])
    data['h']['term_l'] = H_term_l
    
    # SDD21 from S4P
    for id_ in ['ch', 'die_s', 'pkg_s', 'die_l', 'pkg_l']:
        if p[id_]:
            h_raw = sdd21(data['s4p'][id_])
            # Interpolate to common frequency vector f
            interp_func_real = PchipInterpolator(data['hf'][id_], h_raw.real)
            interp_func_imag = PchipInterpolator(data['hf'][id_], h_raw.imag)
            data['h'][id_] = interp_func_real(f) + 1j * interp_func_imag(f)
        else:
            data['h'][id_] = np.ones(len(f))

    # Derive Network
    H = (data['h']['term_s'] *
         data['h']['die_s'] *
         data['h']['pkg_s'] *
         data['h']['ch'] *
         data['h']['pkg_l'] *
         data['h']['die_l'] *
         data['h']['term_l'])
    
    H[0] = H[0].real
    
    return H, f, S11_s, S11_l

def get_crossings(signal, os):
    """Finds all zero-crossing indices in an oversampled signal."""
    # Find where the sign changes, indicating a crossing
    # We subtract the mean to handle any DC offset
    sign_changes = np.diff(np.sign(signal - np.mean(signal))) != 0
    crossing_indices = np.where(sign_changes)[0]
    
    # Filter out consecutive crossings due to noise (debounce)
    # A real transition should be separated by at least half a UI
    if len(crossing_indices) > 1:
        min_dist = os // 2
        valid_crossings = [crossing_indices[0]]
        for i in range(1, len(crossing_indices)):
            if (crossing_indices[i] - valid_crossings[-1]) > min_dist:
                valid_crossings.append(crossing_indices[i])
        return np.array(valid_crossings)
        
    return crossing_indices

def main():
    # Constants from the main script
    PLOT_FREQ_RESP = True
    PLOT_PULSE_RESP = True
    ADD_RAND_JITTER = False

    # Global Variables
    data_rate = 56.1e9 #NRZ
    f_nyq = data_rate / 2
    g['ui'] = 1 / data_rate
    g['os'] = 128 #samples per symbol
    g['tx_launch_amp'] = 0.6
    g['num_pre_cursor'] = 1
    g['num_post_cursor'] = 4
    g['rterm_source'] = 50
    g['rterm_sink'] = 50
    
    print("Variable Initialized.\n")
    
    # Input Pulse Generation
    Ts = g['ui'] / g['os'] # Time step
    pulse_response_length = 100
    total_data_width = pulse_response_length * g['ui']
    pulse_start = 3 * g['ui']
    
    t = np.arange(0, total_data_width, Ts)
    g['pulse_signal'] = np.zeros_like(t)
    
    start_index = int(pulse_start / Ts)
    end_index = int(start_index + (1 * g['ui']) / Ts)
    g['pulse_signal'][start_index:end_index] = g['tx_launch_amp']
    g['pulse_signal_length'] = int(total_data_width / Ts)
    print("Input pulse is defined.\n")

    #generate binary data
    data = sdp.prbs13(1)
    #generate Baud-Rate sampled signal from data
    signal_BR = sdp.nrz_input_BR(data)
    #oversampled signal
    signal_ideal = 0.5*g['tx_launch_amp'] * np.repeat(signal_BR, g['os'])
    print("PRBS signal train is generated.\n")
    #TX signal with jitter
    # gaussian_jitter(signal_ideal, UI,n_symbols,samples_per_symbol,stdev):
    if ADD_RAND_JITTER:
        signal_jitter = sdp.gaussian_jitter(signal_ideal, g['ui'], len(data), g['os'], stdev=1000e-15)
        print("Random Jitter is added.\n")
    else:
        signal_jitter = signal_ideal
    
    # TX to RX link
    print("Creating Transfer Function from end to end.\n")

    # Define project paths relative to this script's location
    script_dir = Path(__file__).resolve().parent
    s_param_dir = script_dir / "Channels"
    
    if not s_param_dir.is_dir():
        print(f"ERROR: S-parameter directory not found at '{s_param_dir}'")
        print("Please ensure the 'Channels' directory exists in the same directory as this script.")
        return

    H_base, f_base, S11_s_base, S11_l_base = gen_channel(
        # Source
        r_s=g['rterm_source'],
        c_die_s=150e-15,
        L1_s=100e-12,
        c_esd1_s=200e-15,
        L2_s=300e-12,
        c_esd2_s=200e-15,
        L3_s=300e-12,
        c_pad_s=100e-15,
        km_s=-0.5,
        # Sink
        c_pad_l=100e-15,
        L1_l=50e-12,
        c_esd_l=400e-15,
        L2_l=200e-12,
        c_die_l=150e-15,
        km_l=-0.4,
        r_l=g['rterm_sink'],
        # Source Package
        # pkg_s=os.path.join(s_param_dir, 'PKG100GEL_95ohm_30mm_50ohmPort.s4p'),
        # Sink Package
        # pkg_l=os.path.join(s_param_dir, 'PKG100GEL_95ohm_30mm_50ohmPort.s4p'),
        # ch=os.path.join(s_param_dir, '100G_PAM4_Cisco_c2c_thru_ch1.s4p'),
        s_tcoil=False,
        s_tcoil_split = False,
        l_tcoil=False,
        l_tcoil_split = False,
        pkg_s_portswap=False,
        pkg_l_portswap=False
    )

    g['H_ch'], g['f'], S11_s, S11_l = gen_channel(
        # Source
        r_s=g['rterm_source'],
        c_die_s=150e-15,
        L1_s=250e-12,
        c_esd1_s=200e-15,
        L2_s=100e-12,
        c_esd2_s=200e-15,
        L3_s=150e-12,
        c_pad_s=100e-15,
        km_s=-0.4,
        # Sink
        c_pad_l=100e-15,
        L1_l=50e-12,
        c_esd_l=400e-15,
        L2_l=200e-12,
        c_die_l=150e-15,
        km_l=-0.4,
        r_l=g['rterm_sink'],
        # Source Package
        pkg_s=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p',
        # Sink Package
        pkg_l=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p',
        # Channel
        # ch=s_param_dir / '100G_PAM4_Cisco_c2c_thru_ch1.s4p',
        s_tcoil=False,
        s_tcoil_split = True,
        l_tcoil=False,
        l_tcoil_split = False,
        pkg_s_portswap=True,
        pkg_l_portswap=True
    )
    if g['H_ch'] is None: return # Exit if s-params failed to load
    print("Full link: Transfer Function evaluation completed.\n")
    
    # Impulse Response
    imp_ch_raw, Fs_ntwk = frd_imp(g['H_ch'], g['f'] * 2 * np.pi)
    Fs = 1/Ts
    g['ratio_oversampling'] = round(Fs / (2 * Fs_ntwk))
    print("Full link: Impulse Response evaluation completed.\n")
    
    # Output
    if PLOT_FREQ_RESP:
        fi_nyq = np.argmin(np.abs(g['f'] - f_nyq))
        H_ch_loss_at_nyquist = 20 * np.log10(np.abs(g['H_ch'][fi_nyq]))
        
        fi_rate = np.argmin(np.abs(g['f'] - data_rate))
        H_ch_loss_at_rate = 20 * np.log10(np.abs(g['H_ch'][fi_rate]))
        
        print(f"Loss at {f_nyq/1e9:.1f} GHz for end to end full link: {H_ch_loss_at_nyquist:.2f}dB")
        print(f"Loss at {data_rate/1e9:.1f} GHz for end to end full link: {H_ch_loss_at_rate:.2f}dB\n")

        plt.figure()
        plt.plot(g['f']/1e9, 20 * np.log10(np.abs(g['H_ch'])), label="Channel Loss")
        plt.plot(f_base/1e9, 20 * np.log10(np.abs(H_base)), label="TX-RX Link")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Frequency Response")
        plt.legend()
        plt.grid(True)
        
    if PLOT_PULSE_RESP:
        imp_ch = impinterp(np.fft.irfft(g['H_ch']), g['ratio_oversampling'])
        imp_ch /= np.sum(np.abs(imp_ch))
        
        #Single One response
        pulse_resp_ch = cconv(g['pulse_signal'], imp_ch, g['pulse_signal_length'])

        #Signal train after channel
        signal_filtered = sp.signal.fftconvolve(signal_jitter, imp_ch, mode="full")
        
        plt.figure()
        plt.plot(t/1e-12, g['pulse_signal'], label="Input Pulse")
        plt.plot(t/1e-12, pulse_resp_ch, label="Channel Response")
        plt.title(f"Pulse Response of End-to-End Channel for {pulse_response_length}UI duration")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        # Stat EYE
        print("\nSTAT-EYE measurements:")
        n_precursor = 6
        n_postcursor = 12
        EYE_TRAVERSING_TGRID = 127
        EYE_TRAVERSING_VGRID = 801
        EYE_TRAVERSING_VRANGE = stat_eye.recommend_vspan_include_main(pulse_resp_ch, g['ui'], Ts, n_pre=n_precursor, n_post=n_postcursor, n_tau=EYE_TRAVERSING_TGRID)

        taus_ui, V, isi_pdfs, main = stat_eye.build_isi_pdf_grid_from_pulse_anchored(pulse_resp_ch, g['ui'], Ts, n_precursor=n_precursor, n_postcursor=n_postcursor, n_tau=EYE_TRAVERSING_TGRID, v_bins=EYE_TRAVERSING_VGRID, v_span=EYE_TRAVERSING_VRANGE, prune_energy=0.9999, prune_abs_floor=0.0)
        stat_eye.plot_eye_center_isi_bar_from_grid(taus_ui, V, isi_pdfs)    
        stat_eye.compare_eye_heights_at_center(pulse_resp_ch, g['ui'], Ts, n_pre=n_precursor, n_post=n_postcursor, ber_target=1e-6)

        isi_avg = stat_eye.apply_tau_kernel_dual_dirac(isi_pdfs, sigma_rj_ui=0.05, dj_pp_ui=0.00)

        t_ui, Vg, BER = stat_eye.build_ber_grid_from_isi_streaming(taus_ui, V, isi_avg, main, sigma_v=0.0, chunk_tau=4, dtype=np.float32)
        stat_eye.plot_ber_contours_ui01(t_ui, Vg, BER, levels=(1e-6,1e-9,1e-12,1e-15))
                
    plt.show()

if __name__ == "__main__":
    main()
