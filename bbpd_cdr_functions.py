import numpy as np
import scipy.signal
import warnings

# --- (Other functions like create_divided_clock, half_rate_sampler, deserialize_to_matrix, deserialize_2_to_20, bang_bang_phase_detector remain UNCHANGED) ---

# --- UPDATED synchronous_majority_voter ---

def synchronous_majority_voter(decision_matrix, decimation_factor, pipeline_latency_words=1):
    """
    Implements two-stage majority vote and models pipeline latency.
    
    NOTE: The 'divided_clock_signal' and 'sample_rate' arguments have been removed 
    as the simulation currently models the synchronous re-timing using discrete array 
    indexing and latency padding, not the analog clock waveform.

    Args:
        decision_matrix (np.ndarray): (N_words, 19) matrix of BBPD decisions.
        decimation_factor (int): The number of words to combine for one final vote.
        pipeline_latency_words (int): Latency added in Word Clock cycles (number of votes).

    Returns:
        np.ndarray: The decimated and majority-voted output (-1, 0, or 1).
    """
    if decimation_factor <= 0:
        warnings.warn("Decimation factor must be positive. Returning empty array.", UserWarning)
        return np.array([])
        
    # 1. Intra-word majority vote (collapse 19 decisions into a phase error score per word)
    word_sums = np.sum(decision_matrix, axis=1)
    
    # --- Inter-Word Synchronous Decimation (Accumulation) ---
    
    # Trim to multiple of decimation_factor
    N_words_to_vote = len(word_sums)
    num_blocks = N_words_to_vote // decimation_factor
    if num_blocks == 0:
        return np.array([])
        
    trimmed_length = num_blocks * decimation_factor
    trimmed_sums = word_sums[:trimmed_length]
    
    # Reshape into blocks
    blocks = trimmed_sums.reshape((num_blocks, decimation_factor))
    
    # 3. Final Vote (sum across the blocks)
    final_sums = np.sum(blocks, axis=1)
    voted_output = np.sign(final_sums).astype(int)
    
    # --- Latency Modeling ---
    # Add final stage voting latency, measured in final vote periods
    if pipeline_latency_words > 0:
        # The latency here is modeled in the final voted output cycles (num_blocks)
        latency_vote_cycles = pipeline_latency_words 
        dummy_votes = np.zeros(latency_vote_cycles, dtype=voted_output.dtype)
        
        # Prepend the dummy votes
        voted_output = np.concatenate((dummy_votes, voted_output), axis=0)
    
    return voted_output

# Helper function definition for completeness
def decimator_majority_vote(input_stream, decimation_factor):
    """
    Performs majority voting decimation on a stream of samples.
    """
    if decimation_factor <= 0:
        raise ValueError("Decimation factor must be positive.")
    
    num_blocks = len(input_stream) // decimation_factor
    if num_blocks == 0:
        return np.array([])
        
    trimmed_length = num_blocks * decimation_factor
    trimmed_stream = input_stream[:trimmed_length]
    
    blocks = trimmed_stream.reshape((num_blocks, decimation_factor))
    sums = np.sum(blocks, axis=1)
    decimated_output = np.sign(sums).astype(int)
    return decimated_output

# --- (Other utility functions) ---
def create_divided_clock(clock_signal, division_factor):
    """
    Creates a divided clock from an input clock waveform.
    """
    rising_edges = np.where((clock_signal[:-1] < 0.5) & (clock_signal[1:] >= 0.5))[0]
    divided_clock = np.zeros_like(clock_signal)
    output_level = 0
    
    # Apply a gentle filter for a more realistic square wave shape
    b, a = scipy.signal.butter(2, 0.05, 'low')
    
    for i in range(len(rising_edges)):
        if i % division_factor == 0:
            output_level = 1 - output_level
            
        start_idx = rising_edges[i]
        end_idx = rising_edges[i+1] if i + 1 < len(rising_edges) else len(clock_signal)
        
        divided_clock[start_idx:end_idx] = output_level
        
    return scipy.signal.filtfilt(b, a, divided_clock)

def half_rate_sampler(data_signal, clock_signal, history, sample_rate, setup_time=0.0e-12, clk_to_data_delay=10e-12, sensitivity=5e-3):
    """
    Models a sample-and-hold circuit triggered on the positive clock edge. 
    """
    sample_and_hold_output = history[-1] * np.ones_like(data_signal)
    threshold = 0.5
    rising_edge_indices = np.where((clock_signal[:-1] < threshold) & (clock_signal[1:] >= threshold))[0]

    if len(rising_edge_indices) == 0:
        return sample_and_hold_output 

    setup_samples = int(setup_time * sample_rate)
    clk_to_data_samples = int(clk_to_data_delay * sample_rate)

    # Initial state comes from the history.
    current_held_bit = np.sign(history[-1]) 
    if current_held_bit == 0: current_held_bit = -1 # Handle zero case if needed

    for i in range(len(rising_edge_indices)):
        current_edge_idx = rising_edge_indices[i]
        hold_start_idx = current_edge_idx + clk_to_data_samples
        
        if i + 1 < len(rising_edge_indices):
            next_edge_idx = rising_edge_indices[i+1]
            hold_end_idx = next_edge_idx + clk_to_data_samples
        else:
            hold_end_idx = len(data_signal)

        data_sample_idx = current_edge_idx - setup_samples

        if data_sample_idx < 0 or hold_start_idx >= len(data_signal):
            continue
        
        sampled_value = data_signal[data_sample_idx]
        # Hysteresis (Guard Band) Logic
        # ---------------------------------------------------------
        # If we are currently LOW (-1), we need a strong positive signal to switch HIGH
        if current_held_bit == -1:
            if sampled_value > sensitivity:
                current_held_bit = 1
            else:
                current_held_bit = -1 # Stay Low (Guard band protection)

        # If we are currently HIGH (1), we need a strong negative signal to switch LOW
        else: # current_held_bit == 1
            if sampled_value < -sensitivity:
                current_held_bit = -1
            else:
                current_held_bit = 1 # Stay High (Guard band protection)
        # ---------------------------------------------------------        
        actual_hold_end = min(hold_end_idx, len(data_signal))
        sample_and_hold_output[hold_start_idx:actual_hold_end] = current_held_bit
        
    return sample_and_hold_output
   
def deserialize_to_matrix(sah_d0, sah_d1, sah_e0, sah_e1, align_clock):
    """
    Aligns four S&H streams (D0, D1, E0, E1) to a reference clock.
    """
    threshold = 0.5
    align_clk_edges = np.where((align_clock[:-1] < threshold) & (align_clock[1:] >= 0.5))[0]
    sample_indices = align_clk_edges + 1

    max_len = min(len(sah_d0), len(sah_d1), len(sah_e0), len(sah_e1))
    valid_mask = sample_indices < max_len
    sample_indices = sample_indices[valid_mask]

    ## Due to finite clock to Q delay of the samplers, this will reurn previously sampled data by i, q, ib and qb clock.
    d0_aligned = sah_d0[sample_indices]
    d1_aligned = sah_d1[sample_indices]
    e0_aligned = sah_e0[sample_indices]
    e1_aligned = sah_e1[sample_indices]
    
    hr_data = np.vstack((d0_aligned, d1_aligned)).T
    hr_edge = np.vstack((e0_aligned, e1_aligned)).T
    
    return hr_data, hr_edge

def deserialize_2_to_20(hr_data, hr_edge, factor=10, pipeline_latency_words=2):
    """
    Deserializes 2-bit wide parallel data into 2*factor-bit wide parallel data (e.g., 20-bit),
    modeling pipeline latency by prepending dummy words.
    """
    num_words = len(hr_data) // factor
    if num_words == 0:
        return np.array([]), np.array([])
    
    trimmed_len = num_words * factor
    
    # Process Data
    trimmed_data = hr_data[:trimmed_len]
    blocks_data = trimmed_data.reshape((num_words, factor, 2))
    deserialized_data = np.array([block.flatten() for block in blocks_data])

    # Process Edge
    trimmed_edge = hr_edge[:trimmed_len]
    blocks_edge = trimmed_edge.reshape((num_words, factor, 2))
    deserialized_edge = np.array([block.flatten() for block in blocks_edge])
    
    # --- Latency Modeling ---
    if pipeline_latency_words > 0:
        word_width = 2 * factor
        # Create dummy words (zeros) for the latency period
        dummy_data = np.zeros((pipeline_latency_words, word_width), dtype=deserialized_data.dtype)
        dummy_edge = np.zeros((pipeline_latency_words, word_width), dtype=deserialized_edge.dtype)
        
        # Prepend the dummy words to model pipeline latency
        deserialized_data = np.concatenate((dummy_data, deserialized_data), axis=0)
        deserialized_edge = np.concatenate((dummy_edge, deserialized_edge), axis=0)
        
    return deserialized_data, deserialized_edge

def bang_bang_phase_detector(data_words, edge_words, pipeline_latency_words=1):
    """
    Implements BBPD on 20-bit data/edge words, yielding 19 decisions per word, 
    and models pipeline latency by prepending dummy decisions.
    """
    N_words, WORD_WIDTH = data_words.shape
    if N_words == 0 or WORD_WIDTH != 20:
        return np.array([])
        
    # --- BBPD Logic ---
    D_pre = data_words[:, 1:20] 
    D_post = data_words[:, 0:19]
    E_sense = edge_words[:, 1:20] 
    
    decision_matrix = np.zeros_like(D_pre, dtype=int)
    
    # Mask out words that are zero (due to latency padding)
    valid_word_mask = (np.sum(np.abs(data_words), axis=1) != 0)
    
    D_pre_valid = D_pre[valid_word_mask]
    D_post_valid = D_post[valid_word_mask]
    E_sense_valid = E_sense[valid_word_mask]

    transition_mask_valid = (D_pre_valid != D_post_valid)
    
    D_pre_trans = D_pre_valid[transition_mask_valid]
    D_post_trans = D_post_valid[transition_mask_valid]
    E_sense_trans = E_sense_valid[transition_mask_valid]
    
    early_mask_trans = (E_sense_trans == D_post_trans)
    late_mask_trans = (E_sense_trans == D_pre_trans)
    
    decisions_trans = np.zeros(np.sum(transition_mask_valid), dtype=int)
    decisions_trans[early_mask_trans] = 1   # EARLY
    decisions_trans[late_mask_trans] = -1  # LATE
    
    temp_matrix = decision_matrix[valid_word_mask]
    temp_matrix[transition_mask_valid] = decisions_trans
    decision_matrix[valid_word_mask] = temp_matrix
    
    # --- Latency Modeling ---
    if pipeline_latency_words > 0:
        N_decisions_per_word = 19
        dummy_decisions = np.zeros((pipeline_latency_words, N_decisions_per_word), dtype=decision_matrix.dtype)
        
        # Prepend the dummy decisions
        decision_matrix = np.concatenate((dummy_decisions, decision_matrix), axis=0)
        
    return decision_matrix

import numpy as np
import scipy.signal
import warnings
# (Utility functions remain unchanged)

# --- CDR Loop Filter Blocks (Digital Loop Filter) ---

def proportional_gain(voter_output, Kp, latency_words=1):
    """
    Implements the Proportional Path (PHUG) in the DLF: Output = Kp * Voter_Output.
    
    Args:
        voter_output (np.ndarray): The phase error signal from the voter (-1, 0, 1).
        Kp (float): Proportional gain.
        latency_words (int): Latency in Word Clock cycles.

    Returns:
        np.ndarray: The proportional control signal.
    """
    prop_output = Kp * voter_output
    
    # Latency Modeling
    if latency_words > 0:
        dummy_output = np.zeros(latency_words, dtype=prop_output.dtype)
        prop_output = np.concatenate((dummy_output, prop_output), axis=0)
        
    return prop_output


def frequency_integrator(voter_output, Kf, initial_code=0.0, latency_words=1):
    """
    Implements the Integral Path (FRUG -> Frequency Integrator).
    Output[n] = Output[n-1] + Kf * Voter_Output[n].
    
    Args:
        voter_output (np.ndarray): The phase error signal from the voter (-1, 0, 1).
        Kf (float): Integral/Frequency gain.
        initial_code (float): Starting value for the frequency accumulator.
        latency_words (int): Latency in Word Clock cycles.

    Returns:
        np.ndarray: The accumulated frequency control code.
    """
    N_words = len(voter_output)
    freq_acc_output = np.zeros(N_words, dtype=float)
    current_acc = initial_code
    
    # Integrate the error signal (FRUG portion is implicitly Kf * error)
    for n in range(N_words):
        current_acc += Kf * voter_output[n]
        freq_acc_output[n] = current_acc
        
    # Latency Modeling
    if latency_words > 0:
        dummy_output = np.zeros(latency_words, dtype=freq_acc_output.dtype)
        
        # NOTE: The dummy output must reflect the initial state for the final accumulator.
        # We fill the last dummy entry with initial_code to ensure continuity.
        dummy_output[-1] = initial_code 
        
        freq_acc_output = np.concatenate((dummy_output, freq_acc_output), axis=0)
        
    return freq_acc_output


def phase_accumulator_and_quantizer(prop_path, int_path_accumulated, initial_phase_code=0.0, total_pi_codes=128, latency_words=1):
    """
    Combines Proportional and Integral paths, accumulates them to track Phase position,
    and quantizes the output to integer PI codes.
    
    Args:
        prop_path: Vector of Proportional corrections (Kp * Error).
        int_path_accumulated: Vector of Frequency offsets (Accumulated Ki * Error).
        initial_phase_code: Starting PI code value (float).
        total_pi_codes: Max codes for wrapping/clipping.
        latency_words: Pipeline latency to model.

    Returns:
        pi_codes_int: The final integer control words for the PI.
    """
    # Ensure inputs are same length
    N = min(len(prop_path), len(int_path_accumulated))
    prop_path = prop_path[:N]
    int_path_accumulated = int_path_accumulated[:N]
    
    pi_codes_int = np.zeros(N, dtype=int)
    
    # State variable: The accurate floating-point phase position
    current_phase_val = float(initial_phase_code)
    
    for i in range(N):
        # 1. Calculate the 'Step' (Velocity) requested by the loop filter
        # Step = (Proportional Jump) + (Frequency Offset)
        delta_phase = prop_path[i] + int_path_accumulated[i]
        
        # 2. Update the 'Position' (Phase)
        current_phase_val += delta_phase
        
        # 3. Quantize to Integer (for the Hardware PI)
        # We round to nearest integer for the output, but KEEP the float 'current_phase_val'
        # for the next iteration. This preserves the fractional error.
        code_out = int(np.round(current_phase_val))
        
        # Clip to valid range (0 to 127)
        # (In a real rotating PI, you would use modulo: code_out % total_pi_codes)
        code_out = max(0, min(code_out, total_pi_codes - 1))
        
        pi_codes_int[i] = code_out
        
    # --- Latency Modeling ---
    if latency_words > 0:
        # Prepend dummy codes (using initial value) to model latency
        dummy_codes = np.full(latency_words, int(initial_phase_code), dtype=int)
        pi_codes_int = np.concatenate((dummy_codes, pi_codes_int), axis=0)
        
    return pi_codes_int

import numpy as np
import matplotlib.pyplot as plt
import control as ct  # A standard library for control systems analysis

# In rx_cdr_functions.py

def analyze_cdr_stability(Kp_nominal: float, Ki_nominal: float, ui_period: float, 
                          bits_per_word: int = 20, loop_latency_words: int = 4, 
                          total_gain_factor: float = 1.0) -> dict:
    """
    Analyzes the stability and performance metrics of a Type-II DLF based on a 
    linearized discrete-time model. Uses an overall loop gain factor 
    A = Kbb * Kd * Kdpc to scale the nominal Kp and Ki.
    
    Kp_eff = Kp_nominal * total_gain_factor
    Ki_eff = Ki_nominal * total_gain_factor

    Args:
        Kp_nominal (float): Proportional loop filter gain (Kp_nom).
        Ki_nominal (float): Integral loop filter gain (Ki_nom).
        ui_period (float): Unit Interval (UI) period in seconds (1/data_rate).
        bits_per_word (int): Number of bits processed per loop cycle (T_word).
        loop_latency_words (int): Total loop latency in word cycles (M).
        total_gain_factor (float): The combined physical gain (Kbb * Kd * Kdpc).

    Returns:
        dict: A dictionary containing the stability parameters.
    """
    
    # Calculate effective loop filter gains by applying the total physical loop gain
    Kp = Kp_nominal * total_gain_factor
    Ki = Ki_nominal * total_gain_factor
    
    # --- Standard Discrete-Time 2nd-Order Digital PLL Formulas (Neglecting Latency) ---
    
    if Ki <= 0:
        return {"error": "Ki_effective must be positive for Type-II loop stability analysis."}

    T_word = bits_per_word * ui_period # T_ref or Ts
    
    # 1. Natural Frequency (omega_n)
    # omega_n = (1/T_word) * sqrt(Ki_eff)
    omega_n_rad_per_sec = (1.0 / T_word) * np.sqrt(Ki)
    f_n_Hz = omega_n_rad_per_sec / (2 * np.pi)

    # 2. Damping Factor (zeta)
    # zeta = Kp_eff / (2 * sqrt(Ki_eff))
    zeta = Kp / (2.0 * np.sqrt(Ki))

    # 3. -3dB Loop Bandwidth (f_3dB) - continuous-time approximation of a 2nd-order system
    f_3db_Hz = 0.0
    if zeta > 0:
        term1 = 1 + 2 * zeta**2
        term2 = np.sqrt(term1**2 + 1)
        f_3db_Hz = f_n_Hz * np.sqrt(term1 + term2)
    else:
        f_3db_Hz = f_n_Hz * 1.0

    # 4. Gain Peaking (GP)
    gain_peaking_dB = 0.0
    if 0.0 < zeta < 0.707:
        gain_peaking_dB = 20 * np.log10(1.0 / (2 * zeta * np.sqrt(1 - zeta**2)))

    return {
        "T_word": T_word,
        "f_n_Hz": f_n_Hz,
        "omega_n_rad_per_sec": omega_n_rad_per_sec,
        "zeta": zeta,
        "f_3db_Hz": f_3db_Hz,
        "gain_peaking_dB": gain_peaking_dB,
        "loop_latency_words": loop_latency_words
    }
# In rx_cdr_functions.py

def plot_cdr_analysis(Kp_nominal: float, Ki_nominal: float, T_word: float, M: int, 
                       total_gain_factor: float, plot_max_freq: float = None):
    """
    Plots both the Open-Loop Bode diagram (Magnitude and Phase) and the 
    Closed-Loop Jitter Transfer Function (JTF) Magnitude.
    """
    if ct is None:
        print("Warning: The 'python-control' library is required for Bode analysis and is not available.")
        return

    # Calculate effective gains
    Kp_eff = Kp_nominal * total_gain_factor
    Ki_eff = Ki_nominal * total_gain_factor

    # 1. Define the Open-Loop Transfer Function G(z)H(z)
    
    # Numerator N_GH: [ Kp_eff*T_word, (Ki_eff - Kp_eff)*T_word ]
    GH_num = [Kp_eff * T_word, (Ki_eff - Kp_eff) * T_word]
    
    # Denominator D_GH: z^(M+1) - z^M
    GH_den = np.zeros(M + 2)
    GH_den[0] = 1.0     # Coefficient for z^(M+1)
    GH_den[1] = -1.0    # Coefficient for z^M
    GH_den = GH_den.tolist()
    
    GH_z = ct.TransferFunction(GH_num, GH_den, dt=T_word)
    
    # --- Open-Loop Analysis ---
    fs = 1.0 / T_word
    omega_max = 2.0 * np.pi * fs / 2.0
    if plot_max_freq is not None:
         omega_max = min(omega_max, 2 * np.pi * plot_max_freq)
         
    omega_range = np.logspace(-2, np.log10(omega_max), 500)
    mag, phase, omega = ct.bode_plot(GH_z, omega=omega_range, plot=False)
    gm, pm, wcg, wcp = ct.margin(GH_z)
    
    gm_dB = 20 * np.log10(gm)
    pm_deg = pm
    wcg_Hz = wcg / (2 * np.pi)
    wcp_Hz = wcp / (2 * np.pi)
    
    # 2. Define the Closed-Loop Transfer Function H(z)
    
    # H(z) = N_GH / (D_GH + N_GH)
    
    # Denominator of H(z) is D_GH + N_GH (polynomial addition)
    # The 'control' library's tf_add function handles polynomial length differences
    # by zero-padding the shorter polynomial, so this is safe.
    H_num = GH_num

    # Denominator of H(z) is D_GH + N_GH (polynomial addition)
    # The 'control' library does not have a public tf_add for coefficients.
    # We must manually pad the polynomials to the maximum length and add them.
    
    # Convert lists to NumPy arrays
    N_poly = np.array(GH_num)
    D_poly = np.array(GH_den)
    
    max_len = max(len(N_poly), len(D_poly))
    
    # Pad the polynomials with leading zeros to match the maximum length
    N_padded = np.pad(N_poly, (max_len - len(N_poly), 0), 'constant')
    D_padded = np.pad(D_poly, (max_len - len(D_poly), 0), 'constant')
    
    # Add the polynomials to get the closed-loop denominator D_H
    H_den = (D_padded + N_padded).tolist()

    H_z = ct.TransferFunction(H_num, H_den, dt=T_word)     
    # --- Closed-Loop JTF Analysis ---
    
    # Calculate magnitude of H(z) in dB
    _, JTF_mag_dB, _ = ct.bode_plot(H_z, omega=omega_range, plot=False)
    
    # Find Gain Peaking
    max_peaking_dB = np.max(JTF_mag_dB)
    
    # Find -3dB JTF Bandwidth
    # Find the index where magnitude first drops below -3dB
    bandwidth_idx = np.where(JTF_mag_dB < -3)[0]
    f_3db_JTF_Hz = 0.0
    if len(bandwidth_idx) > 0:
        # Interpolate between the last point above -3dB and the first point below
        idx1 = bandwidth_idx[0] - 1
        idx2 = bandwidth_idx[0]
        f_3db_JTF_Hz = np.interp(-3, [JTF_mag_dB[idx2], JTF_mag_dB[idx1]], [omega[idx2], omega[idx1]]) / (2 * np.pi)
    
    # 3. Plotting
    
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    fig.suptitle(f'CDR Transfer Function Analysis (Kp={Kp_nominal}, Ki={Ki_nominal}, A={total_gain_factor:.3e}, M={M})', fontsize=14)

    # Plot 1: Open-Loop Magnitude (dB)
    axes[0].semilogx(omega / (2 * np.pi), mag, label='Open-Loop Magnitude')
    axes[0].plot([omega[0] / (2 * np.pi), omega[-1] / (2 * np.pi)], [0, 0], 'k--', label='0 dB Ref')
    axes[0].plot(wcg_Hz, 0, 'go')
    axes[0].plot([wcg_Hz, wcg_Hz], [axes[0].get_ylim()[0], 0], 'g--')
    axes[0].set_ylabel('G(z)H(z) Magnitude (dB)')
    axes[0].grid(which='both', linestyle='--')
    
    # Plot 2: Open-Loop Phase (deg)
    axes[1].semilogx(omega / (2 * np.pi), phase, label='Open-Loop Phase')
    axes[1].plot([omega[0] / (2 * np.pi), omega[-1] / (2 * np.pi)], [-180, -180], 'k--', label='-180° Ref')
    axes[1].plot(wcg_Hz, -180 + pm_deg, 'go')
    axes[1].plot(wcp_Hz, -180, 'ro')
    axes[1].plot([wcg_Hz, wcg_Hz], [-180 + pm_deg, -180], 'g--')
    axes[1].set_ylabel('G(z)H(z) Phase (deg)')
    axes[1].grid(which='both', linestyle='--')
    
    # Plot 3: Closed-Loop Jitter Transfer Function (JTF)
    axes[2].semilogx(omega / (2 * np.pi), JTF_mag_dB, 'b', label='Closed-Loop JTF |H(z)|')
    axes[2].plot([omega[0] / (2 * np.pi), omega[-1] / (2 * np.pi)], [0, 0], 'k--', label='0 dB Ref')
    axes[2].plot(f_3db_JTF_Hz, -3, 'mx', label=f'BW={f_3db_JTF_Hz*1e-6:.2f} MHz')
    axes[2].plot(omega[np.argmax(JTF_mag_dB)] / (2*np.pi), max_peaking_dB, 'r*', label=f'Peak={max_peaking_dB:.2f} dB')
    axes[2].set_ylabel('JTF Magnitude (dB)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].legend(loc='upper left')
    axes[2].grid(which='both', linestyle='--')

    # Add Margin Text
    fig.text(0.92, 0.75, 
             f'Open-Loop Margins:\nPM: {pm_deg:.2f}°\nGM: {gm_dB:.2f} dB\nwcg: {wcg_Hz*1e-6:.2f} MHz', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), 
             transform=axes[0].transAxes)
             
    fig.text(0.92, 0.25, 
             f'Closed-Loop:\nBW: {f_3db_JTF_Hz*1e-6:.2f} MHz\nPeak: {max_peaking_dB:.2f} dB', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), 
             transform=axes[2].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig('cdr_transfer_function_analysis.png')
    plt.close(fig)

def plot_bode_analysis(Kp_nominal: float, Ki_nominal: float, T_word: float, M: int, 
                       total_gain_factor: float, plot_max_freq: float = None):
    """
    Plots the Bode diagram (Magnitude and Phase) of the Type-II CDR's open-loop 
    transfer function G(z)H(z) including loop latency z^(-M) and all physical gains.
    
    Args:
        Kp_nominal (float): Nominal Proportional loop filter gain.
        Ki_nominal (float): Nominal Integral loop filter gain.
        T_word (float): Sampling period (T_ref = T_word) in seconds.
        M (int): Loop latency in word cycles.
        total_gain_factor (float): The combined physical gain (Kbb * Kd * Kdpc).
        plot_max_freq (float, optional): Maximum frequency for the plot in Hz.
    """
    if ct is None:
        print("Warning: The 'python-control' library is required for Bode analysis and is not available.")
        return

    # Calculate effective gains
    Kp_eff = Kp_nominal * total_gain_factor
    Ki_eff = Ki_nominal * total_gain_factor

    # 1. Define the components of the transfer function
    
    # Combined Open Loop G(z)H(z) numerator: [ Kp_eff*T_word, (Ki_eff - Kp_eff)*T_word ]
    GH_num = [Kp_eff * T_word, (Ki_eff - Kp_eff) * T_word]
    
    # Denominator: z^(M+1) - z^M
    # This polynomial represents: 1*z^(M+1) - 1*z^M + 0*z^(M-1) + ... + 0*z^0
    GH_den = np.zeros(M + 2)
    GH_den[0] = 1.0     # Coefficient for z^(M+1)
    GH_den[1] = -1.0    # Coefficient for z^M
    GH_den = GH_den.tolist()
    
    GH_z = ct.TransferFunction(GH_num, GH_den, dt=T_word)
    
    # 2. Bode Plot Generation
    fs = 1.0 / T_word
    omega_max = 2.0 * np.pi * fs / 2.0 # Max frequency up to Nyquist limit
    
    if plot_max_freq is not None:
         omega_max = min(omega_max, 2 * np.pi * plot_max_freq)
         
    # Generate Bode plot data
    mag, phase, omega = ct.bode_plot(GH_z, omega=np.logspace(-2, np.log10(omega_max), 500), plot=False)
    
    # Analyze stability metrics from the frequency response data
    gm, pm, wcg, wcp = ct.margin(GH_z)
    
    # Convert margins to dB and degrees
    gm_dB = 20 * np.log10(gm)
    pm_deg = pm
    
    # 3. Plotting
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    
    # Magnitude Plot
    axes[0].semilogx(omega / (2 * np.pi), mag, label='Magnitude')
    axes[0].plot([omega[0] / (2 * np.pi), omega[-1] / (2 * np.pi)], [0, 0], 'k--', label='0 dB Reference')
    
    # Mark Gain Crossover Frequency (wcg)
    wcg_Hz = wcg / (2 * np.pi)
    axes[0].plot(wcg_Hz, 0, 'go', label=f'wcg={wcg_Hz*1e-6:.2f} MHz')
    axes[0].plot([wcg_Hz, wcg_Hz], [axes[0].get_ylim()[0], 0], 'g--')
    
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].grid(which='both', linestyle='--')
    axes[0].set_title(f'CDR Open-Loop Bode Plot (Kp_nom={Kp_nominal}, Ki_nom={Ki_nominal}, A={total_gain_factor:.3e}, Latency M={M})')
    
    # Phase Plot
    axes[1].semilogx(omega / (2 * np.pi), phase, label='Phase')
    axes[1].plot([omega[0] / (2 * np.pi), omega[-1] / (2 * np.pi)], [-180, -180], 'k--', label='-180° Reference')

    # Mark Phase Crossover Frequency (wcp)
    wcp_Hz = wcp / (2 * np.pi)
    axes[1].plot(wcp_Hz, -180, 'ro', label=f'wcp={wcp_Hz*1e-6:.2f} MHz')
    axes[1].plot([wcp_Hz, wcp_Hz], [axes[1].get_ylim()[0], -180], 'r--')
    
    # Mark Phase Margin (PM)
    pm_line_y = -180 + pm_deg
    axes[1].plot(wcg_Hz, pm_line_y, 'go')
    axes[1].plot([wcg_Hz, wcg_Hz], [pm_line_y, -180], 'g--')
    
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (deg)')
    axes[1].grid(which='both', linestyle='--')
    
    # Add Margin Text
    fig.text(0.92, 0.5, 
             f'Stability Margins:\nPM: {pm_deg:.2f}°\nGM: {gm_dB:.2f} dB', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), 
             transform=axes[0].transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('cdr_bode_plot.png')
    plt.close(fig)