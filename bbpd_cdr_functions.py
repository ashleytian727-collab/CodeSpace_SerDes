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
