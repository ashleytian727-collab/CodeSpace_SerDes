import numpy as np

from sparam_modeling import impinterp, cconv


def gen_ctle_resp(f, aac, adc, fz, fp2):
    """
    Single-zero CTLE frequency response.

    This is a direct translation of the Octave function:
        gen_ctle_resp(f, aac, adc, fz, fp2)
    """
    fp1 = fz * (10.0 ** ((aac - adc) / 20.0))
    wz = 2 * np.pi * fz
    wp2 = 2 * np.pi * fp2
    wp1 = 2 * np.pi * fp1
    k = (10.0 ** (adc / 20.0)) * wp1 * wp2 / wz

    w = 2 * np.pi * f
    s = 1j * w
    H = k * (s + wz) / ((s + wp1) * (s + wp2))
    return H


def gen_ctle_resp_2z(f, adc, fp1, fz2, fp2, fp3):
    """
    Two-zero CTLE frequency response.

    Direct translation of the Octave function:
        gen_ctle_resp_2z(f, adc, fp1, fz2, fp2, fp3)
    """
    fz1 = fp1 * (10.0 ** (adc / 20.0))
    wz1 = 2 * np.pi * fz1
    wz2 = 2 * np.pi * fz2
    wp1 = 2 * np.pi * fp1
    wp2 = 2 * np.pi * fp2
    wp3 = 2 * np.pi * fp3
    k = (10.0 ** (adc / 20.0)) * wp1 * wp2 * wp3 / (wz1 * wz2)

    w = 2 * np.pi * f
    s = 1j * w
    H = k * ((s + wz1) * (s + wz2)) / ((s + wp1) * (s + wp2) * (s + wp3))
    return H


def _phi_metrics(cursors, num_pre_cursor, num_post_cursor):
    """
    Helper implementing the PHI_1 / PHI_2 metrics from the Octave code.

    PHI_1(c, L1, L2) = 1 - sum(|c[L1], c[L1+1]|) / sum(|c|)
    PHI_2(c, L1, L2) = 1 - c[L1] / sum(|c|)
    where indexing has been converted from 1-based (Octave) to 0-based (Python).
    """
    denom = np.sum(np.abs(cursors))
    if denom == 0:
        return 0.0, 0.0

    # Main cursor is at index num_pre_cursor, first post-cursor at num_pre_cursor + 1
    main_and_first_post = np.sum(np.abs(cursors[num_pre_cursor:num_pre_cursor + 2]))
    phi_1 = 1.0 - main_and_first_post / denom
    phi_2 = 1.0 - cursors[num_pre_cursor] / denom
    return float(phi_1), float(phi_2)


def phi(aac, adc, fz, fp2, g, print_progress=True):
    """
    CTLE adaptation objective (single-zero CTLE).

    Direct translation of the Octave function:
        ctle_adapt_obj = phi(aac, adc, fz, fp2)

    The additional argument `g` is a dictionary carrying the variables
    that were `global` in the Octave code:
        g['os']
        g['pulse_signal_length']
        g['pulse_signal']
        g['f']
        g['H_ch']
        g['ratio_oversampling']
        g['num_pre_cursor']
        g['num_post_cursor']

    Returns:
        1D NumPy array: concatenation of the sampled cursors and PHI_2.
    """
    os = g['os']
    pulse_signal_length = g['pulse_signal_length']
    pulse_signal = g['pulse_signal']
    f = g['f']
    H_ch = g['H_ch']
    ratio_oversampling = g['ratio_oversampling']
    num_pre_cursor = g['num_pre_cursor']
    num_post_cursor = g['num_post_cursor']

    # Frequency-domain CTLE response
    H_ctle = gen_ctle_resp(f, aac, adc, fz, fp2)

    # Impulse response: channel * CTLE, then IRFFT and interpolation
    imp = np.fft.irfft(H_ch * H_ctle)
    imp = impinterp(imp, ratio_oversampling)
    norm = np.sum(np.abs(imp))
    if norm != 0:
        imp = imp / norm

    # Pulse response via circular convolution
    pls = cconv(pulse_signal, imp, pulse_signal_length)

    # Find main cursor (maximum pulse response)
    c0_idx = int(np.argmax(pls))

    # Sample cursors at UI intervals: from -num_pre_cursor to +num_post_cursor
    start = c0_idx - num_pre_cursor * os
    stop = c0_idx + num_post_cursor * os
    if start < 0 or stop >= len(pls):
        raise ValueError("Cursor window exceeds pulse response length.")

    cursor_indices = np.arange(start, stop + 1, os, dtype=int)
    cursors = pls[cursor_indices]

    # Compute PHI_2 metric and form adaptation object
    _, phi_2 = _phi_metrics(cursors, num_pre_cursor, num_post_cursor)
    ctle_adapt_obj = np.concatenate([cursors, np.array([phi_2])])

    if print_progress:
        print("-", end="", flush=True)

    return ctle_adapt_obj


def phi_2z(adc, fp1, fz2, fp2, fp3, g, print_progress=True):
    """
    CTLE adaptation objective (two-zero CTLE).

    Direct translation of the Octave function:
        ctle_adapt_obj = phi_2z(adc, fp1, fz2, fp2, fp3)

    The additional argument `g` is a dictionary carrying the variables
    that were `global` in the Octave code:
        g['os']
        g['pulse_signal_length']
        g['pulse_signal']
        g['f']
        g['H_ch']
        g['ratio_oversampling']
        g['num_pre_cursor']
        g['num_post_cursor']

    Returns:
        1D NumPy array: concatenation of the sampled cursors and PHI_2.
    """
    os = g['os']
    pulse_signal_length = g['pulse_signal_length']
    pulse_signal = g['pulse_signal']
    f = g['f']
    H_ch = g['H_ch']
    ratio_oversampling = g['ratio_oversampling']
    num_pre_cursor = g['num_pre_cursor']
    num_post_cursor = g['num_post_cursor']

    # Frequency-domain CTLE response (two zeros)
    H_ctle = gen_ctle_resp_2z(f, adc, fp1, fz2, fp2, fp3)

    # Impulse response: channel * CTLE, then IRFFT and interpolation
    imp = np.fft.irfft(H_ch * H_ctle)
    imp = impinterp(imp, ratio_oversampling)
    norm = np.sum(np.abs(imp))
    if norm != 0:
        imp = imp / norm

    # Pulse response via circular convolution
    pls = cconv(pulse_signal, imp, pulse_signal_length)

    # Find main cursor (maximum pulse response)
    c0_idx = int(np.argmax(pls))

    # Sample cursors at UI intervals: from -num_pre_cursor to +num_post_cursor
    start = c0_idx - num_pre_cursor * os
    stop = c0_idx + num_post_cursor * os
    if start < 0 or stop >= len(pls):
        raise ValueError("Cursor window exceeds pulse response length.")

    cursor_indices = np.arange(start, stop + 1, os, dtype=int)
    cursors = pls[cursor_indices]

    # Compute PHI_2 metric and form adaptation object
    _, phi_2 = _phi_metrics(cursors, num_pre_cursor, num_post_cursor)
    ctle_adapt_obj = np.concatenate([cursors, np.array([phi_2])])

    if print_progress:
        print("-", end="", flush=True)

    return ctle_adapt_obj


# Convenience alias mirroring the standalone impinterp in the Octave code
def impinterp_local(P, n):
    """
    Local wrapper for impulse response interpolation by factor n >= 1.

    This mirrors the anonymous impinterp function at the end of the Octave code:
        impinterp = @(P,n) interp1(..., 'pchip', 0)
    and simply delegates to sparam_modeling.impinterp.
    """
    return impinterp(P, n)

