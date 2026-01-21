"""
Phase Interpolator for Quadrature Clock Signals

This module provides a phase interpolator that takes 4 quadrature clocks (0°, 90°, 180°, 270°)
and generates an interpolated clock signal by mixing adjacent phase clocks with a ratio 
determined by an input code.

The interpolator divides the full 360° phase range into 4 quadrants:
- Quadrant 0: Mix clk_0 and clk_90 (0° to 90°)
- Quadrant 1: Mix clk_90 and clk_180 (90° to 180°)
- Quadrant 2: Mix clk_180 and clk_270 (180° to 270°)
- Quadrant 3: Mix clk_270 and clk_0 (270° to 360°/0°)

Each quadrant is further divided into 2^num_bits steps for fine phase control.

ENHANCED FEATURES (v2.0):
- DNL (Differential Non-Linearity) modeling with multiple profile shapes
- INL (Integral Non-Linearity) modeling with multiple profile shapes
- Complementary clock output (180° out of phase)
- Comprehensive error analysis and visualization capabilities
"""

import numpy as np


def _phase_interpolate_impl(clk_0, clk_90, clk_180, clk_270, num_bits, code, 
                            dnl_profile=None, inl_profile=None, complementary=False):
    """
    Generate an interpolated clock signal by mixing adjacent quadrature clocks.
    
    The function creates a phase-shifted clock by blending two adjacent quadrature
    clocks based on the input code. The code determines both the quadrant and the
    mixing ratio within that quadrant. Optional DNL and INL profiles can model
    real-world non-linearity effects.
    
    Parameters
    ----------
    clk_0 : ndarray
        Clock signal at 0° phase
    clk_90 : ndarray
        Clock signal at 90° phase
    clk_180 : ndarray
        Clock signal at 180° phase
    clk_270 : ndarray
        Clock signal at 270° phase
    num_bits : int
        Number of bits for phase resolution. Total phases = 4 * 2^num_bits
    code : int
        Control code determining the output phase and mixing ratio.
        - 0 to 2^num_bits - 1: Mix clk_0 and clk_90
        - 2^num_bits to 2*2^num_bits - 1: Mix clk_90 and clk_180
        - 2*2^num_bits to 3*2^num_bits - 1: Mix clk_180 and clk_270
        - 3*2^num_bits to 4*2^num_bits - 1: Mix clk_270 and clk_0
    dnl_profile : ndarray, optional
        DNL (Differential Non-Linearity) profile. Array of length (4*2^num_bits)
        with DNL error in degrees for each code value. If None, ideal linear response.
    inl_profile : ndarray, optional
        INL (Integral Non-Linearity) profile. Array of length (4*2^num_bits)
        with cumulative INL error in degrees for each code value. If None, no INL.
    complementary : bool, optional
        If True, returns both regular and complementary clocks (180° out of phase).
        Default is False.
    
    Returns
    -------
    clk_interp : ndarray
        Interpolated clock signal with phase determined by code
    clk_interp_n : ndarray or None
        Complementary clock (180° out of phase) if complementary=True, else None
    phase_degrees : float
        Output phase in degrees (0° to 360°)
    phase_error : float
        Phase error due to DNL/INL modeling in degrees
    mixing_ratio : float
        Ratio parameter used for mixing (0 to 1)
    
    Raises
    ------
    ValueError
        If code is out of valid range or num_bits is invalid
        If DNL/INL profiles have incorrect length
    
    Examples
    --------
    Generate a 10 GHz clock and create 256 interpolated phases:
    
    >>> from clock_generator import generate_clock_signal
    >>> from phase_interpolator import _phase_interpolate_impl
    >>> 
    >>> # Generate quadrature clocks
    >>> t, clk_0, clk_90, clk_180, clk_270, f, pn, ui = generate_clock_signal(
    ...     clock_freq_hz=10e9,
    ...     duration_ui=100,
    ...     samples_per_ui=256
    ... )
    >>> 
    >>> # Create interpolated clock at 45° with complementary output
    >>> num_bits = 8  # 256 phases per quadrant
    >>> code = 2**7   # Middle of first quadrant (45°)
    >>> clk_interp, clk_interp_n, phase, error, ratio = _phase_interpolate_impl(
    ...     clk_0, clk_90, clk_180, clk_270, num_bits, code, complementary=True
    ... )
    >>> print(f"Output phase: {phase:.1f}°, Phase error: {error:.3f}°")
    >>> print(f"Mixing ratio: {ratio:.3f}")
    >>> print(f"Complementary clock available: {clk_interp_n is not None}")
    """
    
    # Validate num_bits
    if not isinstance(num_bits, (int, np.integer)) or num_bits < 1:
        raise ValueError(f"num_bits must be a positive integer, got {num_bits}")
    
    # Calculate the maximum code value
    max_code = 4 * (2 ** num_bits) - 1
    
    # Validate code range (accept both Python int and numpy integer types)
    if not isinstance(code, (int, np.integer)) or code < 0 or code > max_code:
        raise ValueError(
            f"code must be an integer in range [0, {max_code}], got {code}"
        )
    
    # Ensure all clock arrays are numpy arrays with same length
    clk_0 = np.asarray(clk_0)
    clk_90 = np.asarray(clk_90)
    clk_180 = np.asarray(clk_180)
    clk_270 = np.asarray(clk_270)
    
    if not (len(clk_0) == len(clk_90) == len(clk_180) == len(clk_270)):
        raise ValueError("All clock signals must have the same length")
    
    # Define quadrant boundaries and corresponding clock pairs
    quadrant_size = 2 ** num_bits
    
    if code < quadrant_size:
        # Quadrant 0: Mix clk_0 and clk_90 (0° to 90°)
        local_code = code
        clk_a = clk_0
        clk_b = clk_90
        phase_start = 0.0
        phase_range = 90.0
        quadrant_label = "0→90"
        
    elif code < 2 * quadrant_size:
        # Quadrant 1: Mix clk_90 and clk_180 (90° to 180°)
        local_code = code - quadrant_size
        clk_a = clk_90
        clk_b = clk_180
        phase_start = 90.0
        phase_range = 90.0
        quadrant_label = "90→180"
        
    elif code < 3 * quadrant_size:
        # Quadrant 2: Mix clk_180 and clk_270 (180° to 270°)
        local_code = code - 2 * quadrant_size
        clk_a = clk_180
        clk_b = clk_270
        phase_start = 180.0
        phase_range = 90.0
        quadrant_label = "180→270"
        
    else:
        # Quadrant 3: Mix clk_270 and clk_0 (270° to 360°/0°)
        local_code = code - 3 * quadrant_size
        clk_a = clk_270
        clk_b = clk_0
        phase_start = 270.0
        phase_range = 90.0
        quadrant_label = "270→0"
    
    # Calculate mixing ratio (0 = 100% clk_a, 1 = 100% clk_b)
    mixing_ratio = local_code / quadrant_size
    
    # Linear interpolation (weighted sum) of two adjacent quadrature clocks
    # clk_interp = (1 - ratio) * clk_a + ratio * clk_b
    clk_interp = (1.0 - mixing_ratio) * clk_a + mixing_ratio * clk_b
    
    # Calculate output phase in degrees (ideal, without non-linearity)
    phase_ideal = phase_start + mixing_ratio * phase_range
    phase_ideal = phase_ideal % 360.0
    
    # Apply DNL and INL corrections
    phase_error = 0.0
    if dnl_profile is not None:
        dnl_profile = np.asarray(dnl_profile)
        if len(dnl_profile) != 4 * quadrant_size:
            raise ValueError(
                f"DNL profile length ({len(dnl_profile)}) must match "
                f"total phases (4 * 2^{num_bits} = {4 * quadrant_size})"
            )
        phase_error += dnl_profile[code]
    
    if inl_profile is not None:
        inl_profile = np.asarray(inl_profile)
        if len(inl_profile) != 4 * quadrant_size:
            raise ValueError(
                f"INL profile length ({len(inl_profile)}) must match "
                f"total phases (4 * 2^{num_bits} = {4 * quadrant_size})"
            )
        phase_error += inl_profile[code]
    
    # Apply phase error to output phase
    phase_degrees = phase_ideal + phase_error
    phase_degrees = phase_degrees % 360.0
    
    # Generate complementary clock if requested (180° out of phase)
    clk_interp_n = None
    if complementary:
        clk_interp_n = -clk_interp
    
    return clk_interp, clk_interp_n, phase_degrees, phase_error, mixing_ratio


class phase_interpolate:
    """
    Phase interpolator class for CDR/sampling applications.
    
    This class provides an object-oriented interface for phase interpolation,
    designed for use in Clock and Data Recovery (CDR) loops and high-speed
    sampling applications. It maintains configuration and provides methods
    for generating interpolated clocks at specified phases.
    
    The interpolator uses weighted mixing of four quadrature clock inputs
    (0°, 90°, 180°, 270°) to generate outputs at arbitrary phases determined
    by a digital control code.
    
    Parameters
    ----------
    num_bits : int
        Phase resolution in bits. Total phase codes = 4 * 2^num_bits.
        For example, num_bits=6 gives 256 total phase codes.
    dnl_profile : ndarray, optional
        Differential Non-Linearity error profile for realistic PI mismatch.
        Should have length = 4 * 2^num_bits.
    inl_profile : ndarray, optional
        Integral Non-Linearity error profile.
    
    Examples
    --------
    Create a phase interpolator with 6-bit resolution (256 codes):
    
    >>> pi = phase_interpolate(num_bits=6)
    
    Generate interpolated clock at code 128:
    
    >>> clk_i, clk_i_b = pi.interpolate_phase(clk_0, clk_0_b, clk_90, clk_90_b, code=128)
    
    Generate both I and Q phase clocks:
    
    >>> clk_i, clk_i_b, clk_q, clk_q_b = pi.interpolate_phase_quad(
    ...     clk_0, clk_0_b, clk_90, clk_90_b, code=128
    ... )
    """
    
    def __init__(self, num_bits, dnl_profile=None, inl_profile=None):
        """
        Initialize phase interpolator with configuration.
        
        Parameters
        ----------
        num_bits : int
            Phase resolution in bits
        dnl_profile : ndarray, optional
            DNL error profile
        inl_profile : ndarray, optional
            INL error profile
        """
        self.num_bits = num_bits
        self.dnl_profile = dnl_profile
        self.inl_profile = inl_profile
        self.max_code = 4 * (2 ** num_bits) - 1
        self.total_codes = 4 * (2 ** num_bits)
    
    def interpolate_phase(self, clk_0, clk_0_b, clk_90, clk_90_b, code):
        """
        Generate interpolated clock at specified code with complementary output.
        
        This method is the primary interface for CDR applications. It takes
        four quadrature clock inputs (true and complement pairs) and generates
        an interpolated clock at the phase specified by the control code.
        
        Parameters
        ----------
        clk_0 : ndarray
            Clock signal at 0° phase (true)
        clk_0_b : ndarray
            Clock signal at 0° phase (complement/bar)
        clk_90 : ndarray
            Clock signal at 90° phase (true)
        clk_90_b : ndarray
            Clock signal at 90° phase (complement/bar)
        code : int or float
            Control code [0, 4*2^num_bits - 1] specifying the output phase.
            code=0 → 0°, code=total_codes/4 → 90°, etc.
        
        Returns
        -------
        clk_interp : ndarray
            Interpolated clock signal at the requested phase
        clk_interp_b : ndarray
            Complementary interpolated clock (180° out of phase)
        
        Raises
        ------
        ValueError
            If code is outside valid range or clock arrays have mismatched lengths
        
        Examples
        --------
        >>> clk_i, clk_i_b = pi.interpolate_phase(clk_0, clk_0_b, clk_90, clk_90_b, 64)
        """
        # Validate code
        code_int = int(code) % self.total_codes
        
        clk_i, clk_i_b, _, _, _ = _phase_interpolate_impl(
            clk_0, clk_90, clk_0_b, clk_90_b, self.num_bits, code_int,
            dnl_profile=self.dnl_profile, inl_profile=self.inl_profile,
            complementary=True
        )
        return clk_i, clk_i_b
    
    def interpolate_phase_quad(self, clk_0, clk_0_b, clk_90, clk_90_b, code):
        """
        Generate both in-phase and quadrature interpolated clocks.
        
        This method generates two sets of complementary clocks offset by 90°.
        Useful for half-rate sampling or quadrature detection applications.
        
        Parameters
        ----------
        clk_0 : ndarray
            Clock signal at 0° phase (true)
        clk_0_b : ndarray
            Clock signal at 0° phase (complement)
        clk_90 : ndarray
            Clock signal at 90° phase (true)
        clk_90_b : ndarray
            Clock signal at 90° phase (complement)
        code : int or float
            Control code for the in-phase clock [0, 4*2^num_bits - 1]
        
        Returns
        -------
        clk_i : ndarray
            In-phase interpolated clock
        clk_i_b : ndarray
            In-phase complementary clock
        clk_q : ndarray
            Quadrature interpolated clock (90° offset from in-phase)
        clk_q_b : ndarray
            Quadrature complementary clock
        
        Examples
        --------
        >>> clk_i, clk_i_b, clk_q, clk_q_b = pi.interpolate_phase_quad(
        ...     clk_0, clk_0_b, clk_90, clk_90_b, code=64
        ... )
        """
        code_int = int(code) % self.total_codes
        # Offset code by 90° (one quadrant)
        code_90_offset = (code_int + (2 ** self.num_bits)) % self.total_codes
        
        # In-phase
        clk_i, clk_i_b, _, _, _ = _phase_interpolate_impl(
            clk_0, clk_90, clk_0_b, clk_90_b, self.num_bits, code_int,
            dnl_profile=self.dnl_profile, inl_profile=self.inl_profile,
            complementary=True
        )
        
        # Quadrature (90° offset)
        clk_q, clk_q_b, _, _, _ = _phase_interpolate_impl(
            clk_0, clk_90, clk_0_b, clk_90_b, self.num_bits, code_90_offset,
            dnl_profile=self.dnl_profile, inl_profile=self.inl_profile,
            complementary=True
        )
        
        return clk_i, clk_i_b, clk_q, clk_q_b
    
    def get_phase_degrees(self, code):
        """
        Convert control code to phase in degrees.
        
        Parameters
        ----------
        code : int
            Control code [0, 4*2^num_bits - 1]
        
        Returns
        -------
        phase_deg : float
            Phase in degrees [0, 360)
        """
        code_int = int(code) % self.total_codes
        return (code_int / self.total_codes) * 360.0


def generate_interpolated_bank(clk_0, clk_90, clk_180, clk_270, num_bits, 
                               dnl_profile=None, inl_profile=None, complementary=False):
    """
    Generate a bank of all possible interpolated clocks for a given num_bits setting.
    
    This function generates all 4 * 2^num_bits interpolated clock phases covering
    the full 360° range with uniform phase spacing. Optional DNL and INL profiles
    can model real-world non-linearity effects.
    
    Parameters
    ----------
    clk_0 : ndarray
        Clock signal at 0° phase
    clk_90 : ndarray
        Clock signal at 90° phase
    clk_180 : ndarray
        Clock signal at 180° phase
    clk_270 : ndarray
        Clock signal at 270° phase
    num_bits : int
        Number of bits for phase resolution
    dnl_profile : ndarray, optional
        DNL error profile. Array of length (4*2^num_bits) with DNL error in degrees.
    inl_profile : ndarray, optional
        INL error profile. Array of length (4*2^num_bits) with INL error in degrees.
    complementary : bool, optional
        If True, also generate complementary clocks (180° out of phase).
        Default is False.
    
    Returns
    -------
    clk_bank : ndarray
        2D array of shape (4 * 2^num_bits, len(clk_0)) containing all interpolated
        clock signals. Index i corresponds to code=i.
    clk_bank_n : ndarray or None
        Complementary clock bank if complementary=True, else None
    phases : ndarray
        1D array of output phases in degrees for each interpolated clock
    phase_errors : ndarray
        1D array of phase errors due to DNL/INL for each code
    codes : ndarray
        1D array of code values used (0 to 4*2^num_bits - 1)
    
    Examples
    --------
    Generate all 256 phases for num_bits=6:
    
    >>> clk_bank, clk_bank_n, phases, errors, codes = generate_interpolated_bank(
    ...     clk_0, clk_90, clk_180, clk_270, num_bits=6
    ... )
    >>> print(f"Generated {len(clk_bank)} interpolated clocks")
    >>> print(f"Phase spacing: {phases[1] - phases[0]:.3f}°")
    """
    
    num_phases = 4 * (2 ** num_bits)
    sample_length = len(clk_0)
    
    # Pre-allocate arrays
    clk_bank = np.zeros((num_phases, sample_length))
    clk_bank_n = np.zeros((num_phases, sample_length)) if complementary else None
    phases = np.zeros(num_phases)
    phase_errors = np.zeros(num_phases)
    codes = np.arange(num_phases)
    
    # Generate all interpolated clocks
    for code in range(num_phases):
        clk, clk_n, phase, error, _ = _phase_interpolate_impl(
            clk_0, clk_90, clk_180, clk_270, num_bits, code,
            dnl_profile=dnl_profile, inl_profile=inl_profile, 
            complementary=complementary
        )
        clk_bank[code] = clk
        phases[code] = phase
        phase_errors[code] = error
        if complementary and clk_n is not None:
            clk_bank_n[code] = clk_n
    
    return clk_bank, clk_bank_n, phases, phase_errors, codes


def sin_to_square(signal, sample_rate, clock_freq, method='zerocross', threshold=0.0, slew_rate=None):
    """
    Convert a sinusoidal signal to a square wave using various methods.
    
    This function converts continuous sinusoidal clock signals to digital square
    wave signals using different conversion techniques, modeling realistic
    comparator behavior.
    
    Parameters
    ----------
    signal : ndarray
        Input sinusoidal signal to convert
    sample_rate : float
        Sampling rate in Hz
    clock_freq : float
        Clock frequency in Hz
    method : str, optional
        Conversion method to use. Options:
        - 'zerocross': Zero-crossing detector (fastest edge)
        - 'threshold': Comparator with fixed threshold level
        - 'slew': Slew-rate limited comparator output
        - 'hysteresis': Schmitt trigger with hysteresis
        - 'quantized': Quantized output with limited levels
        Default is 'zerocross'
    threshold : float, optional
        Threshold level for comparator-based methods. Default is 0.0 (mid-point)
    slew_rate : float, optional
        Slew rate for 'slew' method in units/sample. If None, uses fastest slew.
        Default is None
    
    Returns
    -------
    square : ndarray
        Output square wave signal (typically ±1 or 0/1)
    edge_indices : ndarray
        Indices where rising edges occur
    transition_times : ndarray
        Estimated transition times (in samples) with sub-sample precision
    
    Raises
    ------
    ValueError
        If method is not recognized or invalid parameters provided
    
    Examples
    --------
    Convert a sinusoidal clock to square wave using zero-crossing detector:
    
    >>> from clock_generator import generate_clock_signal
    >>> from phase_interpolator import sin_to_square
    >>> 
    >>> # Generate a sinusoidal clock
    >>> t, clk_0, _, _, _, _, _, _ = generate_clock_signal(
    ...     clock_freq_hz=10e9,
    ...     duration_ui=10,
    ...     samples_per_ui=256
    ... )
    >>> 
    >>> # Convert to square wave
    >>> square_wave, edges, trans_times = sin_to_square(clk_0, 2.56e12, 10e9, method='zerocross')
    >>> print(f"Generated {len(edges)} rising edges")
    >>> print(f"Square wave range: [{square_wave.min():.2f}, {square_wave.max():.2f}]")
    """
    
    signal = np.asarray(signal)
    
    if method == 'zerocross':
        # Zero-crossing detector: output transitions where signal crosses threshold
        # Find zero crossings with sub-sample precision
        zero_crosses = np.where(np.diff(np.sign(signal - threshold)))[0]
        
        # Interpolate for more precise crossing times using linear interpolation
        transition_times = np.zeros(len(zero_crosses))
        for i, idx in enumerate(zero_crosses):
            if idx < len(signal) - 1:
                # Linear interpolation to find exact crossing point
                y0 = signal[idx] - threshold
                y1 = signal[idx + 1] - threshold
                transition_times[i] = idx - y0 / (y1 - y0) if (y1 - y0) != 0 else idx
            else:
                transition_times[i] = idx
        
        # Generate square wave from transitions
        square = np.ones_like(signal, dtype=float)
        current_state = 1 if signal[0] > threshold else -1
        square[0] = current_state
        
        for idx in zero_crosses:
            if idx < len(signal):
                current_state *= -1
                square[idx:] = current_state
        
        # Find rising edges (transitions from -1 to +1)
        rising_edges = zero_crosses[::2] if len(zero_crosses) > 0 else np.array([])
        
    elif method == 'threshold':
        # Simple comparator: output high when signal > threshold, else low
        square = np.where(signal > threshold, 1.0, -1.0)
        
        # Find rising edges (transitions from -1 to +1)
        rising_edges = np.where(np.diff(square) > 0)[0]
        transition_times = rising_edges.astype(float)
        
    elif method == 'slew':
        # Slew-rate limited comparator output
        if slew_rate is None:
            slew_rate = 2.0  # Default fast slew rate
        
        square = np.zeros_like(signal, dtype=float)
        current_output = -1.0 if signal[0] < threshold else 1.0
        target_output = current_output
        square[0] = current_output
        
        for i in range(1, len(signal)):
            # Update target based on input
            target_output = 1.0 if signal[i] > threshold else -1.0
            
            # Slew towards target
            diff = target_output - current_output
            if abs(diff) > slew_rate:
                current_output += np.sign(diff) * slew_rate
            else:
                current_output = target_output
            
            square[i] = current_output
        
        # Find rising edges
        rising_edges = np.where(np.diff(square) > 0)[0]
        transition_times = rising_edges.astype(float)
        
    elif method == 'hysteresis':
        # Schmitt trigger with hysteresis
        upper_threshold = threshold + 0.1
        lower_threshold = threshold - 0.1
        
        square = np.zeros_like(signal, dtype=float)
        current_state = -1.0 if signal[0] < lower_threshold else 1.0
        square[0] = current_state
        
        for i in range(1, len(signal)):
            if current_state > 0:  # Currently high
                if signal[i] < lower_threshold:
                    current_state = -1.0
            else:  # Currently low
                if signal[i] > upper_threshold:
                    current_state = 1.0
            square[i] = current_state
        
        # Find rising edges
        rising_edges = np.where(np.diff(square) > 0)[0]
        transition_times = rising_edges.astype(float)
        
    elif method == 'quantized':
        # Quantized levels
        levels = 3  # Number of quantization levels
        normalized = (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)
        quantized = np.round(normalized * (levels - 1)) / (levels - 1)
        square = 2.0 * quantized - 1.0  # Convert to [-1, 1] range
        
        # Find rising edges
        rising_edges = np.where(np.diff(square) > 0)[0]
        transition_times = rising_edges.astype(float)
        
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of: "
                        "'zerocross', 'threshold', 'slew', 'hysteresis', 'quantized'")
    
    return square, rising_edges, transition_times


def create_dnl_profile(num_bits, dnl_magnitude=0.5, shape='random'):
    """
    Create a DNL (Differential Non-Linearity) error profile.
    
    DNL is the deviation of an individual step from the ideal step size.
    This function generates realistic DNL profiles for phase interpolators.
    
    Parameters
    ----------
    num_bits : int
        Phase resolution (total phases = 4 * 2^num_bits)
    dnl_magnitude : float
        Maximum DNL error in degrees (typical: 0.1-1.0°)
    shape : str
        Profile shape - 'random', 'sine', 'sawtooth', or 'monotonic'
        - 'random': Random walk DNL errors (most realistic)
        - 'sine': Sinusoidal DNL pattern
        - 'sawtooth': Sawtooth pattern (quadrant-dependent)
        - 'monotonic': Monotonic increase (linear drift)
    
    Returns
    -------
    dnl_profile : ndarray
        DNL errors in degrees for each code (length: 4 * 2^num_bits)
    
    Examples
    --------
    >>> dnl = create_dnl_profile(num_bits=6, dnl_magnitude=0.5, shape='random')
    >>> print(f"DNL profile shape: {dnl.shape}")
    >>> print(f"Max DNL: {np.max(np.abs(dnl)):.3f}°")
    """
    num_phases = 4 * (2 ** num_bits)
    
    if shape == 'random':
        # Random walk - realistic for real hardware
        dnl = np.random.randn(num_phases) * (dnl_magnitude / 3)
        dnl = np.cumsum(dnl - np.mean(dnl))  # Zero-mean cumulative
        dnl = (dnl / np.max(np.abs(dnl))) * dnl_magnitude  # Scale to magnitude
        
    elif shape == 'sine':
        # Sinusoidal pattern across all phases
        phase_vals = np.linspace(0, 4 * np.pi, num_phases)
        dnl = dnl_magnitude * np.sin(phase_vals) / 2
        
    elif shape == 'sawtooth':
        # Sawtooth pattern per quadrant
        quadrant_size = 2 ** num_bits
        dnl = np.zeros(num_phases)
        for q in range(4):
            phase_in_quad = np.linspace(0, 1, quadrant_size)
            dnl[q*quadrant_size:(q+1)*quadrant_size] = (
                dnl_magnitude * (phase_in_quad - 0.5)
            )
        
    elif shape == 'monotonic':
        # Linear drift across entire phase range
        dnl = np.linspace(-dnl_magnitude/2, dnl_magnitude/2, num_phases)
        
    else:
        raise ValueError(f"Unknown DNL shape: {shape}")
    
    return dnl.astype(np.float64)


def create_inl_profile(num_bits, inl_magnitude=1.0, shape='random'):
    """
    Create an INL (Integral Non-Linearity) error profile.
    
    INL is the cumulative error from ideal phase as we sweep through codes.
    This represents systematic non-linearity in the phase relationship.
    
    Parameters
    ----------
    num_bits : int
        Phase resolution (total phases = 4 * 2^num_bits)
    inl_magnitude : float
        Maximum INL error in degrees (typical: 0.5-5.0°)
    shape : str
        Profile shape - 'random', 'sine', 'quadratic', or 'linear'
        - 'random': Random walk INL (most realistic)
        - 'sine': Sinusoidal INL pattern
        - 'quadratic': Parabolic INL (compression/expansion)
        - 'linear': Linear INL (monotonic drift)
    
    Returns
    -------
    inl_profile : ndarray
        INL errors in degrees for each code (length: 4 * 2^num_bits)
    
    Examples
    --------
    >>> inl = create_inl_profile(num_bits=6, inl_magnitude=1.0, shape='sine')
    >>> print(f"INL profile shape: {inl.shape}")
    >>> print(f"Max INL: {np.max(np.abs(inl)):.3f}°")
    """
    num_phases = 4 * (2 ** num_bits)
    
    if shape == 'random':
        # Random walk INL - realistic for real hardware
        inl = np.cumsum(np.random.randn(num_phases))
        inl = (inl / np.max(np.abs(inl))) * inl_magnitude if np.max(np.abs(inl)) > 0 else inl
        
    elif shape == 'sine':
        # Sinusoidal INL pattern
        phase_vals = np.linspace(0, 4 * np.pi, num_phases)
        inl = inl_magnitude * np.sin(phase_vals)
        
    elif shape == 'quadratic':
        # Parabolic INL (phase compression/expansion)
        normalized = np.linspace(-1, 1, num_phases)
        inl = inl_magnitude * (normalized ** 2 - 0.5)
        
    elif shape == 'linear':
        # Linear INL across entire range
        inl = np.linspace(-inl_magnitude/2, inl_magnitude/2, num_phases)
        
    else:
        raise ValueError(f"Unknown INL shape: {shape}")
    
    return inl.astype(np.float64)


if __name__ == '__main__':
    """
    Demonstration and verification of the phase interpolator.
    """
    from clock_generator import generate_clock_signal
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Generate quadrature clocks
    print("="*70)
    print("PHASE INTERPOLATOR - DEMONSTRATION (v2.0 with DNL/INL & Complementary)")
    print("="*70)
    
    clock_freq = 10e9
    duration_ui = 20
    samples_per_ui = 256
    
    print(f"\nGenerating {clock_freq/1e9} GHz quadrature clocks...")
    t, clk_0, clk_90, clk_180, clk_270, f_noise, pn, ui = generate_clock_signal(
        clock_freq_hz=clock_freq,
        duration_ui=duration_ui,
        samples_per_ui=samples_per_ui,
        rj_rms_ui=0.005,
        dj_freq_hz=100e6,
        dj_peak_ui=0.01
    )
    
    print(f"✓ Generated {len(t)} samples ({t[-1]*1e9:.2f} ns)")
    
    # Test 1: Single phase interpolation examples
    print("\n" + "="*70)
    print("TEST 1: Single Phase Interpolation Examples")
    print("="*70)
    
    num_bits = 8  # 256 phases per quadrant = 1024 total phases
    test_codes = [0, 128, 256, 384, 512, 640, 768, 896, 1023]
    
    print(f"\nnum_bits = {num_bits} ({2**num_bits} phases per quadrant, {4*2**num_bits} total)")
    print(f"\nTesting various codes:")
    print(f"{'Code':<6} {'Phase (°)':<12} {'Quadrant':<15} {'Mix Ratio':<12}")
    print("-" * 50)
    
    for code in test_codes:
        try:
            clk_interp, _, phase, _, ratio = _phase_interpolate_impl(
                clk_0, clk_90, clk_180, clk_270, num_bits, code
            )
            
            # Determine quadrant
            if code < 256:
                quadrant = "0→90"
            elif code < 512:
                quadrant = "90→180"
            elif code < 768:
                quadrant = "180→270"
            else:
                quadrant = "270→0"
            
            print(f"{code:<6} {phase:<12.2f} {quadrant:<15} {ratio:<12.4f}")
        except ValueError as e:
            print(f"{code:<6} ERROR: {e}")
    
    # Test 2: Generate full phase bank
    print("\n" + "="*70)
    print("TEST 2: Full Phase Bank Generation")
    print("="*70)
    
    num_bits_bank = 6  # 64 phases per quadrant = 256 total
    print(f"\nGenerating full phase bank with num_bits={num_bits_bank}...")
    
    clk_bank, _, phases, _, codes = generate_interpolated_bank(
        clk_0, clk_90, clk_180, clk_270, num_bits_bank
    )
    
    print(f"✓ Generated {len(clk_bank)} interpolated clocks")
    print(f"  Phase range: {phases[0]:.3f}° to {phases[-1]:.3f}°")
    phase_spacing = phases[1] - phases[0]
    print(f"  Phase spacing: {phase_spacing:.4f}°")
    print(f"  Expected spacing: {360.0 / len(clk_bank):.4f}°")
    
    # Test 3: Error handling
    print("\n" + "="*70)
    print("TEST 3: Error Handling")
    print("="*70)
    
    invalid_codes = [
        (8, -1, "negative code"),
        (8, 4096, "code too large"),
        (8, 4095.5, "non-integer code"),
    ]
    
    num_bits_test = 8
    print(f"\nTesting error cases with num_bits={num_bits_test}:")
    print(f"Valid range: 0 to {4*2**num_bits_test - 1}")
    
    for nb, code, description in invalid_codes:
        try:
            clk_interp, _, phase, _, ratio = _phase_interpolate_impl(
                clk_0, clk_90, clk_180, clk_270, nb, code
            )
            print(f"✗ {description:<25} - Should have raised error!")
        except (ValueError, TypeError) as e:
            print(f"✓ {description:<25} - Caught: {type(e).__name__}")
    
    # Test 4: Complementary clock output
    print("\n" + "="*70)
    print("TEST 4: Complementary Clock Output")
    print("="*70)
    
    num_bits_comp = 8
    code_comp = 128
    print(f"\nGenerating complementary clocks at code={code_comp}, num_bits={num_bits_comp}...")
    
    clk_p, clk_n, phase_comp, error_comp, ratio_comp = _phase_interpolate_impl(
        clk_0, clk_90, clk_180, clk_270, num_bits_comp, code_comp, complementary=True
    )
    
    print(f"✓ Complementary clock pair generated:")
    print(f"  Phase: {phase_comp:.2f}°")
    print(f"  Phase error: {error_comp:.4f}°")
    print(f"  Complementarity check: max(clk_p + clk_n) = {np.max(np.abs(clk_p + clk_n)):.6e}")
    
    # Test 5: DNL/INL profiles
    print("\n" + "="*70)
    print("TEST 5: DNL and INL Profile Generation")
    print("="*70)
    
    num_bits_profiles = 6
    dnl = create_dnl_profile(num_bits_profiles, dnl_magnitude=0.5, shape='random')
    inl = create_inl_profile(num_bits_profiles, inl_magnitude=1.0, shape='sine')
    
    print(f"\nDNL Profile (magnitude=0.5°, shape=random):")
    print(f"  Min: {np.min(dnl):.4f}°, Max: {np.max(dnl):.4f}°, Std: {np.std(dnl):.4f}°")
    print(f"\nINL Profile (magnitude=1.0°, shape=sine):")
    print(f"  Min: {np.min(inl):.4f}°, Max: {np.max(inl):.4f}°, Range: {np.max(inl)-np.min(inl):.4f}°")
    
    # Test 6: Visualization
    print("\n" + "="*70)
    print("TEST 6: Visualization")
    print("="*70)
    
    num_bits_viz = 6  # 64 phases per quadrant = 256 total
    num_phases_viz = 4 * (2 ** num_bits_viz)
    
    print(f"\nGenerating visualization with num_bits={num_bits_viz}...")
    print(f"Visualizing {num_phases_viz} interpolated phases")
    
    # Select 8 phases for visualization (every 32nd phase)
    viz_codes = np.arange(0, num_phases_viz, 32)
    num_to_plot = 10  # Plot 10 UI cycles
    plot_samples = num_to_plot * samples_per_ui
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Quadrature base clocks
    ax = axes[0]
    ax.plot(t[:plot_samples]*1e12, clk_0[:plot_samples], label='clk_0 (0°)', linewidth=1.5, alpha=0.8)
    ax.plot(t[:plot_samples]*1e12, clk_90[:plot_samples], label='clk_90 (90°)', linewidth=1.5, alpha=0.8)
    ax.plot(t[:plot_samples]*1e12, clk_180[:plot_samples], label='clk_180 (180°)', linewidth=1.5, alpha=0.8)
    ax.plot(t[:plot_samples]*1e12, clk_270[:plot_samples], label='clk_270 (270°)', linewidth=1.5, alpha=0.8)
    ax.set_title(f"Four Quadrature Clock Signals ({num_to_plot} UI cycles)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Selected interpolated clocks overlaid
    ax = axes[1]
    colors = plt.cm.hsv(np.linspace(0, 1, len(viz_codes)))
    
    for i, code in enumerate(viz_codes):
        clk_interp, _, phase, _, _ = _phase_interpolate_impl(
            clk_0, clk_90, clk_180, clk_270, num_bits_viz, code
        )
        label = f"code={code} ({phase:.1f}°)"
        ax.plot(t[:plot_samples]*1e12, clk_interp[:plot_samples], 
                label=label, linewidth=1.2, color=colors[i], alpha=0.7)
    
    ax.set_title(f"Interpolated Clock Signals (num_bits={num_bits_viz}, showing {len(viz_codes)} phases)", 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    plt.savefig("plots/phase_interpolator_demo.png", dpi=150, bbox_inches='tight')
    print("✓ Plot saved to 'plots/phase_interpolator_demo.png'")
    
    # Plot phase spacing uniformity
    fig, ax = plt.subplots(figsize=(12, 6))
    
    clk_bank, _, phases, _, codes = generate_interpolated_bank(
        clk_0, clk_90, clk_180, clk_270, num_bits_bank
    )
    
    phase_diffs = np.diff(phases)
    
    ax.plot(codes[:-1], phase_diffs, 'o-', linewidth=2, markersize=6, label='Phase spacing')
    ax.axhline(y=phase_spacing, color='r', linestyle='--', label=f'Expected spacing: {phase_spacing:.4f}°')
    ax.set_xlabel("Code", fontsize=11)
    ax.set_ylabel("Phase Difference (°)", fontsize=11)
    ax.set_title(f"Phase Interpolator Spacing Uniformity (num_bits={num_bits_bank})", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("plots/phase_spacing_uniformity.png", dpi=150, bbox_inches='tight')
    print("✓ Plot saved to 'plots/phase_spacing_uniformity.png'")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
