import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def generate_clock_signal(
    clock_freq_hz: float,
    duration_ui: int = 1000,
    samples_per_ui: int = 128,
    rj_rms_ui: float = 0.001,
    dj_freq_hz: float = 0.0,
    dj_peak_ui: float = 0.0,
    lpf_bw_factor: float = 10.0
):
    """
    Generates a complementary clock signal with optional RJ and DJ, passed through a low-pass filter.
    All timing parameters are specified in terms of UI (Unit Interval = 1/clock_freq).

    Args:
        clock_freq_hz (float): The desired clock frequency in Hz.
        duration_ui (int): The total duration of the signal in UI (Unit Intervals).
        samples_per_ui (int): Number of samples per UI (oversampling factor).
        rj_rms_ui (float): The RMS value of Random Jitter as a fraction of UI.
        dj_freq_hz (float): The frequency of the Sinusoidal Jitter in Hz.
        dj_peak_ui (float): The peak amplitude of the Sinusoidal Jitter as a fraction of UI.
        lpf_bw_factor (float): The factor by which to multiply the clock frequency to get the
                               low-pass filter's cutoff frequency.

    Returns:
        tuple: A tuple containing:
            - t (np.ndarray): The time vector for the signals (in seconds).
            - clk_0 (np.ndarray): The 0° clock signal.
            - clk_90 (np.ndarray): The 90° clock signal.
            - clk_180 (np.ndarray): The 180° clock signal.
            - clk_270 (np.ndarray): The 270° clock signal.
            - f_welch (np.ndarray): Frequency offsets for the phase noise profile (in Hz).
            - phase_noise_dbchz (np.ndarray): SSB Phase Noise in dBc/Hz.
            - ui (float): The Unit Interval in seconds.
    """
    if not 1e9 <= clock_freq_hz <= 30e9:
        raise ValueError("Frequency must be between 1 GHz and 30 GHz.")
    
    if duration_ui < 1:
        raise ValueError("Duration must be at least 1 UI.")
    
    if samples_per_ui < 2:
        raise ValueError("Samples per UI must be at least 2 (Nyquist requirement).")

    # Calculate UI (Unit Interval)
    ui = 1.0 / clock_freq_hz  # UI in seconds
    
    # Calculate sampling frequency based on oversampling
    fs_hz:float = samples_per_ui * clock_freq_hz
    
    # Calculate total duration in seconds
    duration_s = duration_ui * ui
    
    # 1. Create the time vector
    num_samples = duration_ui * samples_per_ui
    t = np.arange(0, num_samples) * (ui / samples_per_ui)

    # 2. Generate jitter signal j(t)
    jitter_signal = np.zeros(num_samples)
    
    # Add sinusoidal jitter (DJ) - convert from UI to seconds
    if dj_freq_hz > 0 and dj_peak_ui > 0:
        dj_peak_s = dj_peak_ui * ui
        jitter_signal += dj_peak_s * np.sin(2 * np.pi * dj_freq_hz * t)
    
    # Add random jitter (RJ) - convert from UI to seconds
    if rj_rms_ui > 0:
        rj_rms_s = rj_rms_ui * ui
        rj_noise = np.random.normal(0, 1, num_samples)
        # Scale noise to have the desired RMS value
        rj_noise = rj_noise * (rj_rms_s / np.sqrt(np.mean(rj_noise**2)))
        jitter_signal += rj_noise

    # 3. Generate the jittery clock
    phase_jittered = 2 * np.pi * clock_freq_hz * (t + jitter_signal)
    clk_p_ideal = np.sin(phase_jittered)
    clk_n_ideal = -clk_p_ideal

    # 4. Create and apply the low-pass filter
    lpf_cutoff_hz = clock_freq_hz * lpf_bw_factor
    # Use a 2nd order Butterworth filter for a gentle rolloff
    b, a = signal.butter(2, lpf_cutoff_hz, btype='low', fs=fs_hz)
    clk_p = signal.lfilter(b, a, clk_p_ideal)
    clk_n = signal.lfilter(b, a, clk_n_ideal)

    # Generate quadrature clocks (0°, 90°, 180°, 270°)
    # 0°: clk_p (already have it)
    clk_0 = clk_p
    
    # 90°: cos(phase) = sin(phase + π/2)
    clk_90_ideal = np.cos(phase_jittered)
    clk_90 = signal.lfilter(b, a, clk_90_ideal)
    
    # 180°: -sin(phase) = sin(phase + π)
    clk_180 = clk_n
    
    # 270°: -cos(phase) = sin(phase + 3π/2)
    clk_270_ideal = -clk_90_ideal
    clk_270 = signal.lfilter(b, a, clk_270_ideal)

    # 5. Calculate the Phase Noise Profile
    # Method: Direct phase deviation from jitter signal (more direct approach)
    # The jitter adds phase modulation: phase_deviation = 2π * freq_hz * jitter_signal
    phase_deviation_from_jitter = 2 * np.pi * clock_freq_hz * jitter_signal
    
    # Calculate the Power Spectral Density (PSD) of the phase deviation using Welch
    # Use appropriate window size for good frequency resolution
    # We want to resolve the 100 MHz DJ modulation, so window should be small enough
    nperseg = min(int(fs_hz / 1e6), len(t) // 4)  # ~1 MHz frequency resolution
    f_welch, psd_phase_deviation = signal.welch(
        phase_deviation_from_jitter,
        fs=fs_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,  # 50% overlap for better averaging
        scaling='density',
        window='blackmanharris'  # Better spectral leakage properties than Hann
    )
    
    # Convert to Single-Sideband (SSB) phase noise in dBc/Hz
    # L(f) = 10 * log10( 0.5 * S_phi(f) )
    # Avoid log of zero by using a small epsilon
    phase_noise_dbchz = 10 * np.log10(np.maximum(0.5 * psd_phase_deviation, 1e-20))

    return t, clk_0, clk_90, clk_180, clk_270, f_welch, phase_noise_dbchz, ui


if __name__ == '__main__':
    # --- Example Usage (UI-based parameters) ---
    # Parameters
    clock_freq = 10e9      # 10 GHz
    duration_ui = 2000     # 2000 UI duration
    samples_per_ui = 256   # 256 samples per UI (2.56 THz sampling)
    
    # Jitter components (as fractions of UI)
    rj_rms_ui = 0.01       # 1% of UI RMS
    dj_amp_ui = 0.025      # 2.5% of UI peak
    dj_freq = 100e6        # 100 MHz modulation frequency

    print(f"Generating {clock_freq/1e9} GHz clock...")
    print(f"  Duration: {duration_ui} UI")
    print(f"  Samples per UI: {samples_per_ui}")
    print(f"  RJ RMS: {rj_rms_ui*100:.2f}% of UI")
    print(f"  DJ Peak: {dj_amp_ui*100:.2f}% of UI")

    t, clk_0, clk_90, clk_180, clk_270, f_noise, pn_dbchz, ui = generate_clock_signal(
        clock_freq_hz=clock_freq,
        duration_ui=duration_ui,
        samples_per_ui=samples_per_ui,
        rj_rms_ui=rj_rms_ui,
        dj_freq_hz=dj_freq,
        dj_peak_ui=dj_amp_ui
    )

    print(f"Generation complete. UI = {ui*1e12:.3f} ps")
    print(f"  Total duration: {t[-1]*1e9:.2f} ns")
    print(f"  Total samples: {len(t)}")
    print("Plotting results...")

    # --- Verification / Testing ---
    print("\n" + "="*60)
    print("VERIFICATION: Checking phase noise contains expected peaks")
    print("="*60)
    
    # Find peak near the DJ frequency (100 MHz)
    if dj_freq > 0:
        mask_dj = (f_noise > dj_freq * 0.5) & (f_noise < dj_freq * 1.5)
        if np.any(mask_dj):
            peak_idx = np.argmax(pn_dbchz[mask_dj])
            peak_freq = f_noise[np.where(mask_dj)[0][peak_idx]]
            peak_level = pn_dbchz[np.where(mask_dj)[0][peak_idx]]
            print(f"\n✓ DJ (Deterministic Jitter) Peak:")
            print(f"    Expected frequency: {dj_freq/1e6:.1f} MHz")
            print(f"    Found frequency: {peak_freq/1e6:.2f} MHz")
            print(f"    Peak level: {peak_level:.1f} dBc/Hz")
            print(f"    Expected amplitude: {dj_amp_ui*100:.2f}% UI = {dj_amp_ui*ui*1e12:.2f} ps")
        else:
            print(f"\n✗ DJ peak not found near {dj_freq/1e6:.1f} MHz")
    
    # Estimate RJ floor (broadband noise floor at high offset frequencies)
    if rj_rms_ui > 0:
        # RJ creates a broadband floor; check high frequency region (e.g., 500 MHz - 1 GHz)
        mask_rj = (f_noise > 500e6) & (f_noise < 1e9)
        if np.any(mask_rj):
            rj_floor_avg = np.mean(pn_dbchz[mask_rj])
            print(f"\n✓ RJ (Random Jitter) Floor:")
            print(f"    Expected RMS: {rj_rms_ui*100:.2f}% UI = {rj_rms_ui*ui*1e12:.2f} ps")
            print(f"    Measured broadband floor (500 MHz - 1 GHz): {rj_floor_avg:.1f} dBc/Hz")
    
    print("\n" + "="*60)

    # --- Plotting ---
    
    # 1. Plot a small segment of the quadrature clock signals
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"{clock_freq/1e9} GHz Quadrature Clocks with Jitter (RJ={rj_rms_ui*100:.1f}% UI, DJ={dj_amp_ui*100:.1f}% UI)", fontsize=14)

    # Display 10 clock cycles
    num_cycles_to_plot = 10
    plot_end_index = int(num_cycles_to_plot * samples_per_ui)
    
    # Plot all 4 quadrature clocks
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t[:plot_end_index] * 1e12, clk_0[:plot_end_index], label='Clock 0°', linewidth=1.5, color='blue')
    ax1.plot(t[:plot_end_index] * 1e12, clk_90[:plot_end_index], label='Clock 90°', linewidth=1.5, color='red', alpha=0.8)
    ax1.set_title(f"Quadrature Clocks 0° and 90° ({num_cycles_to_plot} cycles)")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t[:plot_end_index] * 1e12, clk_180[:plot_end_index], label='Clock 180°', linewidth=1.5, color='green')
    ax2.plot(t[:plot_end_index] * 1e12, clk_270[:plot_end_index], label='Clock 270°', linewidth=1.5, color='orange', alpha=0.8)
    ax2.set_title(f"Quadrature Clocks 180° and 270° ({num_cycles_to_plot} cycles)")
    ax2.set_xlabel("Time (ps)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # 2. Plot the Phase Noise Profile
    ax3 = plt.subplot(3, 1, 3)
    # Plot from 1 kHz to 1 GHz offset
    ax3.semilogx(f_noise, pn_dbchz, linewidth=2, color='darkblue')
    ax3.set_title("Phase Noise Profile (SSB)")
    ax3.set_xlabel("Frequency Offset (Hz)")
    ax3.set_ylabel("SSB Phase Noise (dBc/Hz)")
    ax3.grid(True, which='both', alpha=0.3)
    ax3.set_xlim(1e3, 1e9)
    ax3.set_ylim(-160, -40)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Ensure plots directory exists
    from pathlib import Path
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    plt.savefig("plots/clock_generator_output.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'plots/clock_generator_output.png'")
    
    # To avoid issues in headless environments, we save the plot instead of showing it.
    # If you are in a graphical environment, you can uncomment the next line.
    # plt.show()
