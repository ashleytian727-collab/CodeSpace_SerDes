
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import scipy as sp
from phase_interpolator import PhaseInterpolator, sin_to_square
import rx_cdr_functions as cdr

def bbpd_cdr_loop_tran(
    DESERIALIZATION_FACTOR,
    g,
    signal_filtered,
    PHASE_INTG_DITHER_BITS,
    TOTAL_LOOP_LATENCY_WORDS,
    PI_NUM_BITS,
    rx_clk_i,
    rx_clk_i_b,
    rx_clk_q,
    rx_clk_q_b,
    t_cdr,
    sample_rate,
    RX_CLOCK_FREQUENCY,
    SAMPLER_C2Q,
    SAMPLER_SENSE,
    KP_GAIN,
    KI_GAIN,
    FREQ_INTG_DITHER_BITS,
    TOTAL_PI_CODES,
    VOTER_DECIMATION_FACTOR,
):
    samples_per_word = DESERIALIZATION_FACTOR * (2 * g['os']) # 2x since half rate sampling and os is number of samples per bit/symbol
    total_available_samples = len(signal_filtered)
    num_words_to_simulate = total_available_samples // samples_per_word
    num_words_to_simulate = min(num_words_to_simulate, 5000)

    print(f"Starting Closed-Loop Simulation for {num_words_to_simulate} words...")
    # --- Initialize State Variables
    INITIAL_PI_CODE = 32.0
    current_pi_code_float = INITIAL_PI_CODE * (2**PHASE_INTG_DITHER_BITS) # Start in middle of range (Code 64)
    current_pi_code_int = int(current_pi_code_float/(2**PHASE_INTG_DITHER_BITS))
    freq_integrator_state = 0.0
    # --- Latency Queue (FIFO)
    # ---  --- Stores pi_codes to apply in future cycles
    pi_code_fifo = deque([current_pi_code_int] * TOTAL_LOOP_LATENCY_WORDS, maxlen=TOTAL_LOOP_LATENCY_WORDS)

    # ---  --- History Recorders
    history_pi_codes = []
    history_freq_code = []
    history_phase_error = []
    history_phase_float = []

    rxpi_sq_i_final = []

    history_sah_output_d0 = -1 * np.ones(samples_per_word)
    history_sah_output_d1 = -1 * np.ones(samples_per_word)
    history_sah_output_e0 = -1 * np.ones(samples_per_word)
    history_sah_output_e1 = -1 * np.ones(samples_per_word)
    # ---  --- Instantiate PI Object
    pi = PhaseInterpolator(num_bits=PI_NUM_BITS)

    #Time-Marching Loop ---
    plot_once = False
    window_count = 0
    window_to_print = 1000

    for i in range(num_words_to_simulate): #Loop is going through every 20 bits or a word or byte_clock
        if i % 100 == 0:
            print(f"Processing Word {i}/{num_words_to_simulate} | Current PI Code: {current_pi_code_int}")
            window_count +=1

        if window_count == window_to_print:
            plot_once = True
            window_count += 1
        
        # ... Determine Time Window for this Word
        start_idx = i * samples_per_word
        end_idx = start_idx + samples_per_word
        
        # Add a small safety buffer (padding) for the slicer/sampler edges, so we don't lose the first/last edge of the block
        pad = g['os'] * 0
        safe_start = max(0, start_idx - pad)
        safe_end = min(len(signal_filtered), end_idx + pad)
        
        # Extract Slices
        clk_i_slice = rx_clk_i[safe_start:safe_end]
        clk_ib_slice = rx_clk_i_b[safe_start:safe_end]
        clk_q_slice = rx_clk_q[safe_start:safe_end]
        clk_qb_slice = rx_clk_q_b[safe_start:safe_end]
        t_cdr_slice = t_cdr[safe_start:safe_end]
        
        # ... Actuate Phase Interpolator
        # Apply the PI code that is currently valid (from the head of the latency FIFO)
        active_pi_code = pi_code_fifo[0] # Read current applied code; this will take maxlen=TOTAL_LOOP_LATENCY_WORDS to apply newly calculated code
        
        rxpi_i, rxpi_ib = pi.interpolate_phase(clk_i_slice, clk_ib_slice, clk_q_slice, clk_qb_slice, active_pi_code)
        # 90 deg offset logic dependent on PI implementation
        rxpi_q, rxpi_qb = pi.interpolate_phase(clk_i_slice, clk_ib_slice, clk_q_slice, clk_qb_slice, (active_pi_code + (2**PI_NUM_BITS))%(4*2**PI_NUM_BITS)) 

        # Convert to Square (Slicing)
        rxpi_sq_i  = sin_to_square(rxpi_i, sample_rate, RX_CLOCK_FREQUENCY)
        rxpi_sq_ib = sin_to_square(rxpi_ib, sample_rate, RX_CLOCK_FREQUENCY)
        rxpi_sq_q  = sin_to_square(rxpi_q, sample_rate, RX_CLOCK_FREQUENCY)
        rxpi_sq_qb = sin_to_square(rxpi_qb, sample_rate, RX_CLOCK_FREQUENCY)

        rxpi_sq_i_final.extend(rxpi_sq_i)

        # ... Sample & Deserialize
        # Call the new sample-and-hold function
        sah_output_d0 = cdr.half_rate_sampler(signal_filtered[safe_start:safe_end], rxpi_sq_i,  history_sah_output_d0, sample_rate, setup_time=0e-12, clk_to_data_delay=SAMPLER_C2Q, sensitivity=SAMPLER_SENSE)
        sah_output_d1 = cdr.half_rate_sampler(signal_filtered[safe_start:safe_end], rxpi_sq_ib, history_sah_output_d1, sample_rate, setup_time=0e-12, clk_to_data_delay=SAMPLER_C2Q, sensitivity=SAMPLER_SENSE)
        sah_output_e0 = cdr.half_rate_sampler(signal_filtered[safe_start:safe_end], rxpi_sq_q,  history_sah_output_e0, sample_rate, setup_time=0e-12, clk_to_data_delay=SAMPLER_C2Q, sensitivity=SAMPLER_SENSE)
        sah_output_e1 = cdr.half_rate_sampler(signal_filtered[safe_start:safe_end], rxpi_sq_qb, history_sah_output_e1, sample_rate, setup_time=0e-12, clk_to_data_delay=SAMPLER_C2Q, sensitivity=SAMPLER_SENSE)

        history_sah_output_d0 = sah_output_d0
        history_sah_output_d1 = sah_output_d1
        history_sah_output_e0 = sah_output_e0
        history_sah_output_e1 = sah_output_e1

        # Note: We must be careful to only "count" the bits that belong to this word.
        # The sampler might return extra bits due to the padding.
        # However, deserialize_2_to_20 is robust to length.
        hr_data, hr_edge = cdr.deserialize_to_matrix(sah_output_d0, sah_output_d1, sah_output_e0, sah_output_e1, rxpi_sq_i)

        # Force exactly 1 word (20 bits) processing
        # If we captured more/less due to jitter/padding, we take the center block or first valid block.
        # For simulation stability, we assume the windowing captured at least one full word.
        deser_data, deser_edge = cdr.deserialize_2_to_20(hr_data, hr_edge, factor=DESERIALIZATION_FACTOR, pipeline_latency_words=0) #pi_code_fifo models the latency
        
        if len(deser_data) == 0:
            # Skip if sampling failed (e.g. signal end)
            history_pi_codes.append(active_pi_code)
            history_phase_error.append(0)
            continue
            
        # Take just the first word found in this slice
        curr_data_word = deser_data[0:1] 
        curr_edge_word = deser_edge[0:1]
        
        # ... BBPD & Voter
        bbpd_out = cdr.bang_bang_phase_detector(curr_data_word, curr_edge_word, pipeline_latency_words=0)  #pi_code_fifo models the latency
        voter_out = cdr.synchronous_majority_voter(bbpd_out, decimation_factor=VOTER_DECIMATION_FACTOR, pipeline_latency_words=0)  #pi_code_fifo models the latency
        
        # Extract scalar error (-1, 0, 1)
        error_val = voter_out[0] if len(voter_out) > 0 else 0
        
        # ... Digital Loop Filter (State Update)
        # Proportional: add +/-error_val at every byte_clock or loop iteration
        prop_step = KP_GAIN * error_val
        
        # Frequency Integrator (Accumulate State)
        freq_integrator_state += KI_GAIN * error_val
        # Truncate frequency integrator dither bits
        freq_integrator_state = freq_integrator_state/(2**FREQ_INTG_DITHER_BITS)

        # Calculate the total phase movement requested this cycle:
        total_phase_movement = prop_step + freq_integrator_state
        
        # Phase Accumulate
        current_pi_code_float += total_phase_movement
        
        # Quantize, Dither-truncate & Wrap
        next_pi_code_int = int(np.round(current_pi_code_float)/(2**PHASE_INTG_DITHER_BITS))
        
        # Handle Wrap-Around (Rotational PI)
        # If code goes 255 -> 256, it should wrap to 0. If 0 -> -1, wrap to 255.
        next_pi_code_int = next_pi_code_int % TOTAL_PI_CODES
        
        # Handle Float Wrap (to keep float variable bounded)
        current_pi_code_float = current_pi_code_float % TOTAL_PI_CODES

        # F. Latency Update
        # The new code calculated now will apply N cycles later.
        pi_code_fifo.append(next_pi_code_int) # Enqueue new code
        current_pi_code_int = next_pi_code_int # Update loop variable tracking
        
        # G. Record History
        history_pi_codes.append(active_pi_code)
        history_freq_code.append(freq_integrator_state)
        history_phase_error.append(error_val)
        history_phase_float.append(current_pi_code_float)

        if plot_once:
            print(f"length of pi clock: {len(rxpi_sq_i)}")
            print(f"length of sampler output: {len(sah_output_d1)}")
            print(f"Size of hr data: {hr_data.shape}")
            print(hr_data)
            print(f"Size of dser data: {deser_data.shape}")
            print(deser_data)
            print(deser_edge)
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(t_cdr_slice,signal_filtered[safe_start:safe_end], label="Data Signal")
            plt.plot(t_cdr_slice,rxpi_sq_i, label="In pahse HRCLK")
            plt.plot(t_cdr_slice,rxpi_sq_q, label ="Q phase HRCLK")
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.plot(t_cdr_slice,rxpi_sq_i, label="In pahse HRCLK")
            plt.plot(t_cdr_slice,sah_output_d0, label="In pahse HRDATA")
            plt.plot(t_cdr_slice,sah_output_e0, label="In pahse HRDATA")
            plt.legend()
            plt.grid(True)
            plot_once = False

    print('------------------------------------------------------\n')        
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history_pi_codes)
    plt.title(f"CDR Lock Behavior (Kp={KP_GAIN}, Ki={KI_GAIN})")
    plt.ylabel(f"PI Code (0-{TOTAL_PI_CODES-1})")
    plt.grid(True)
        
    plt.subplot(2, 1, 2)
    # Moving average of phase error density
    window = 50
    err_density = np.convolve(history_phase_error, np.ones(window)/window, mode='valid')
    plt.plot(err_density)
    #plt.plot(history_phase_error)
    plt.title("Smoothed Phase Error Density")
    plt.ylabel("Avg Error (-1=Late, +1=Early)")
    plt.xlabel("Word Cycles")
    plt.grid(True)
        
    plt.tight_layout()
    plt.show()

    return(rxpi_sq_i_final)
