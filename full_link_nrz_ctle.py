"""
Example script demonstrating how to:
- Build an end-to-end SerDes channel using s-parameters
- Compute the channel impulse response
- Evaluate a CTLE objective function using the ctle_modeling helper

This is intentionally similar in structure to full_link_part1.py,
but focused on CTLE modeling and the `phi` objective.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from pathlib import Path
import sys

# Ensure local serdespy copy is importable (from ./serdespy-main)
_script_dir = Path(__file__).resolve().parent
try:
    import serdespy as sdp
except ModuleNotFoundError:
    serdespy_root = _script_dir / "serdespy-main"
    if serdespy_root.is_dir():
        sys.path.insert(0, str(serdespy_root))
        import serdespy as sdp
    else:
        raise

from sparam_modeling import gen_channel, frd_imp, cconv, impinterp
from ctle_modeling import gen_ctle_resp, phi


# Global-style dictionary to mirror the MATLAB "global" variables
g = {
    "pulse_signal": None,
    "f": None,
    "H_ch": None,
    "ratio_oversampling": None,
    "ui": None,
    "os": None,
    "tx_launch_amp": None,
    "pulse_signal_length": None,
    "num_pre_cursor": None,
    "num_post_cursor": None,
}


def main():
    # Configuration flags
    PLOT_FREQ_RESP = True
    PLOT_PULSE_RESP = True
    PLOT_CTLE_FREQ_RESP = True
    ADD_RAND_JITTER = False

    # Basic link parameters
    data_rate = 56e9  # NRZ data rate
    f_nyq = data_rate / 2
    g["ui"] = 1 / data_rate
    g["os"] = 128  # samples per symbol
    g["tx_launch_amp"] = 0.6
    g["num_pre_cursor"] = 1
    g["num_post_cursor"] = 4

    print("Variables initialized.\n")

    # Time grid and input pulse definition (1 UI pulse)
    Ts = g["ui"] / g["os"]  # time step
    pulse_response_length = 100  # in UIs
    total_data_width = pulse_response_length * g["ui"]
    pulse_start = 3 * g["ui"]

    t = np.arange(0, total_data_width, Ts)
    g["pulse_signal"] = np.zeros_like(t)
    start_index = int(pulse_start / Ts)
    end_index = int(start_index + (1 * g["ui"]) / Ts)
    g["pulse_signal"][start_index:end_index] = g["tx_launch_amp"]
    g["pulse_signal_length"] = int(total_data_width / Ts)

    print("Input pulse defined.\n")

    # Generate PRBS data and oversampled NRZ waveform (for optional time-domain plots)
    data = sdp.prbs13(1)
    signal_BR = sdp.nrz_input_BR(data)
    signal_ideal = 0.5 * g["tx_launch_amp"] * np.repeat(signal_BR, g["os"])

    if ADD_RAND_JITTER:
        signal_jitter = sdp.gaussian_jitter(
            signal_ideal,
            g["ui"],
            len(data),
            g["os"],
            stdev=1000e-15,
        )
        print("Random jitter added to transmit signal.\n")
    else:
        signal_jitter = signal_ideal

    print("PRBS signal generated.\n")

    # Build end-to-end channel using s-parameters
    script_dir = Path(__file__).resolve().parent
    s_param_dir = script_dir / "Channels"
    if not s_param_dir.is_dir():
        print(f"ERROR: S-parameter directory not found at '{s_param_dir}'")
        print("Please ensure the 'Channels' directory exists in the same directory as this script.")
        return

    # Example: t-coil based link (same structure as full_link_part1)
    H_ch, f, _, _ = gen_channel(
        # Source
        r_s=50,
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
        r_l=50,
        pkg_s=s_param_dir / "PKG100GEL_95ohm_30mm_50ohmPort.s4p",
        pkg_l=s_param_dir / "PKG100GEL_95ohm_30mm_50ohmPort.s4p",
        # ch=s_param_dir / "100G_PAM4_Cisco_c2c_thru_ch1.s4p",
        s_tcoil=True,
        s_tcoil_split=True,
        l_tcoil=False,
        l_tcoil_split=True,
        pkg_s_portswap=True,
        pkg_l_portswap=True,
        ch_portswap=False,
    )

    if H_ch is None:
        # gen_channel already printed the reason
        return

    g["H_ch"] = H_ch
    g["f"] = f

    print("Full link transfer function evaluated.\n")

    # Impulse response of the channel (no CTLE yet)
    imp_ch_raw, Fs_ntwk = frd_imp(g["H_ch"], g["f"] * 2 * np.pi)
    Fs = 1 / Ts
    g["ratio_oversampling"] = round(Fs / (2 * Fs_ntwk))

    print(f"Channel impulse response evaluated. Oversampling ratio: {g['ratio_oversampling']}\n")

    # Optional: frequency response of the channel alone
    if PLOT_FREQ_RESP:
        fi_nyq = np.argmin(np.abs(g["f"] - f_nyq))
        H_ch_loss_at_nyquist = 20 * np.log10(np.abs(g["H_ch"][fi_nyq]))
        fi_rate = np.argmin(np.abs(g["f"] - data_rate))
        H_ch_loss_at_rate = 20 * np.log10(np.abs(g["H_ch"][fi_rate]))

        print(f"Loss at {f_nyq/1e9:.1f} GHz for end-to-end link: {H_ch_loss_at_nyquist:.2f} dB")
        print(f"Loss at {data_rate/1e9:.1f} GHz: {H_ch_loss_at_rate:.2f} dB\n")

        plt.figure()
        plt.plot(g["f"] / 1e9, 20 * np.log10(np.abs(g["H_ch"])), label="Channel")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Channel Frequency Response")
        plt.grid(True)
        plt.legend()

    # Compute pulse response through the channel
    if PLOT_PULSE_RESP:
        imp_ch = impinterp(np.fft.irfft(g["H_ch"]), g["ratio_oversampling"])
        imp_ch /= np.sum(np.abs(imp_ch))
        pulse_resp_ch = cconv(g["pulse_signal"], imp_ch, g["pulse_signal_length"])

        plt.figure()
        plt.plot(pulse_resp_ch)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.title("Pulse Response Through Channel (No CTLE)")
        plt.grid(True)

    # ------------------------------------------------------------------
    # CTLE modeling and objective evaluation using ctle_modeling.phi
    # ------------------------------------------------------------------

    # Example CTLE parameters (you can tune these as desired)
    aac = -3.0      # AC-coupling gain (dB at low freq)
    adc = 6.0       # DC boost (peaking) in dB
    fz = f_nyq / 4  # zero frequency (Hz)
    fp2 = f_nyq     # high-frequency pole (Hz)

    # Evaluate CTLE adaptation objective
    ctle_adapt_obj = phi(aac, adc, fz, fp2, g)
    cursors = ctle_adapt_obj[:-1]
    phi2_val = ctle_adapt_obj[-1]

    print("CTLE adaptation objective evaluated using ctle_modeling.phi().")
    print(f"  Cursors (pre/main/post): {cursors}")
    print(f"  PHI_2 metric: {phi2_val:.4f}\n")

    # Optional: plot CTLE frequency response and combined channel+CTLE
    if PLOT_CTLE_FREQ_RESP:
        H_ctle = gen_ctle_resp(g["f"], aac, adc, fz, fp2)
        H_total = g["H_ch"] * H_ctle

        plt.figure()
        plt.plot(g["f"] / 1e9, 20 * np.log10(np.abs(H_ctle)), label="CTLE")
        plt.plot(g["f"] / 1e9, 20 * np.log10(np.abs(H_total)), label="Channel + CTLE")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("CTLE and Combined Channel Response")
        plt.grid(True)
        plt.legend()

    # Optional: apply CTLE in time domain to the NRZ waveform
    imp_total = np.fft.irfft(g["H_ch"] * gen_ctle_resp(g["f"], aac, adc, fz, fp2))
    imp_total = impinterp(imp_total, g["ratio_oversampling"])
    imp_total /= np.sum(np.abs(imp_total))

    signal_ctle = sp.signal.fftconvolve(signal_jitter, imp_total, mode="full")
    signal_ctle = signal_ctle[: len(signal_jitter)]

    plt.figure()
    time = Ts * np.arange(len(signal_ctle))
    plt.plot(time[0:5000], signal_ctle[0:5000])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("NRZ Signal After Channel + CTLE")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
