import numpy as np  
import matplotlib.pyplot as plt  
from scipy import signal  
  
# ==========================================  
# 1. USER CONFIGURATION (STRICTLY PRESERVED)  
# ==========================================  
g = {'ui': 1.0 / 56.1e9, 'os': 128}  
data_rate = 56.1e9  
RJ_SIGMA = 0.05 #UI  
# --- Digital Loop Filter (DLF) Constants ---  
KP_GAIN = 1  
KI_GAIN = 1  
PI_NUM_BITS = 6  
TOTAL_PI_CODES_PER_UI = (2**PI_NUM_BITS) #1UI per quadrant  
PHASE_INTG_DITHER_BITS = 3 #Dp  
FREQ_INTG_DITHER_BITS = 4 #Df  
DECIMATION_FACTOR = 20 # L UI  
SYMBOL_INTERVAL = DECIMATION_FACTOR/data_rate  
  
# Physical Gain Factors  
KDPC_GAIN = 1.0 / (2**(PI_NUM_BITS+PHASE_INTG_DITHER_BITS)) # Kdpc: Phase step size in UI per LSB  
KBB_GAIN = 1.0 / (RJ_SIGMA*np.sqrt(2*np.pi))                    # Kbb: BBPD gain (1/WordLength, WordLength=20)  
KD_GAIN = 0.50                 # Kd: Decimation gain (assumed 1.0)  
TOTAL_LOOP_GAIN_FACTOR = KBB_GAIN * KD_GAIN * KDPC_GAIN  
  
# Latency in Word Clock Cycles (M)  
TOTAL_LOOP_LATENCY_WORDS = 6 # M SYMBOL_INTERVAL  
  
def bbpd_dig_cdr_stab(data_rate, RJ_SIGMA, sweep_type, KP_GAIN, KI_GAIN, PI_NUM_BITS, PHASE_INTG_DITHER_BITS, FREQ_INTG_DITHER_BITS, DECIMATION_FACTOR, SYMBOL_INTERVAL, TOTAL_LOOP_GAIN_FACTOR, TOTAL_LOOP_LATENCY_WORDS):  
    # ==========================================  
    # 2. SWEEP CONFIGURATION  
    # ==========================================  
    # Choose parameter to sweep: 'KP_GAIN', 'KI_GAIN', or 'LATENCY'  
    #sweep_type = 'KP_GAIN'   
    UI_margin = (1 - 12*RJ_SIGMA)   # Jitter Tolerance Margin  
  
    if sweep_type == 'KP_GAIN':  
        sweep_values = [2**-1, 1, 2, 4, 8]  # Sweep Digital Kp Gain  
        fixed_Ki_dig = KI_GAIN  
        fixed_N      = TOTAL_LOOP_LATENCY_WORDS  
          
    elif sweep_type == 'KI_GAIN':  
        sweep_values = [2**-5, 2**-3, 1, 2, 4] # Sweep Digital Ki Gain  
        fixed_Kp_dig = KP_GAIN  
        fixed_N      = TOTAL_LOOP_LATENCY_WORDS  
  
    elif sweep_type == 'LATENCY':  
        sweep_values = [6, 8, 10] #SYMBOL_INTERVAL  
        fixed_Kp_dig = KP_GAIN  
        fixed_Ki_dig = KI_GAIN  
  
    else:  
        sweep_values =[1]  
        fixed_Ki_dig = KI_GAIN  
        fixed_Kp_dig = KP_GAIN  
        fixed_N      = TOTAL_LOOP_LATENCY_WORDS  
  
    # ==========================================  
    # 3. PLOTTING SETUP  
    # ==========================================  
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))  
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sweep_values)))  
  
    # Frequency Vector: 10 kHz to Nyquist  
    w = np.logspace(np.log10(1e4 * 2*np.pi/data_rate), np.log10(np.pi), 5000)  
    freq_hz = w * data_rate / (2 * np.pi)  
    if len(sweep_values) > 1:  
        print(f"Running sweep for {sweep_type}...")  
        print(f"Total Loop Gain Factor: {TOTAL_LOOP_GAIN_FACTOR:.6f}")  
    # ==========================================  
    # 4. ITERATION LOOP  
    # ==========================================  
    for i, val in enumerate(sweep_values):  
        # Map sweep values to current loop parameters  
        if sweep_type == 'KP_GAIN':  
            curr_Kp_dig, curr_Ki_dig, curr_N = val, fixed_Ki_dig, fixed_N  
            label_str = f"Kp_dig={val}"  
        elif sweep_type == 'KI_GAIN':  
            curr_Kp_dig, curr_Ki_dig, curr_N = fixed_Kp_dig, val, fixed_N  
            label_str = f"Ki_dig={val}"  
        elif sweep_type == 'LATENCY':  
            curr_Kp_dig, curr_Ki_dig, curr_N = fixed_Kp_dig, fixed_Ki_dig, int(val)  
            label_str = f"Latency={val}"  
        else:  
            curr_Kp_dig, curr_Ki_dig, curr_N = fixed_Kp_dig, fixed_Ki_dig, fixed_N  
            label_str = "All fixed"  
  
        # --- Calculate Effective Linear Gains ---  
        # K_param_eff = Digital_Value * Total_Analog_Gain_Factor  
        K_P_eff = curr_Kp_dig * TOTAL_LOOP_GAIN_FACTOR  
        K_I_eff = curr_Ki_dig * TOTAL_LOOP_GAIN_FACTOR * (1/2**PHASE_INTG_DITHER_BITS)  
  
        # --- Define Coefficients ---  
        # Numerator: Pad with N zeros for delay z^-N  
        # num_open length = N + 2  
        # L(z) = z^-N * [ (Kp_eff + Ki_eff) - Kp_eff * z^-1 ] / (1-z^-1)  
        num_open = np.zeros(curr_N + 2)  
        num_open[curr_N]     = (K_P_eff + K_I_eff)  
        num_open[curr_N + 1] = - K_P_eff  
  
        # Denominator: Integrator poles (1 - z^-1)  
        # Note: If your diagram had a double integrator (PI + Accumulator), it's (1-z^-1)^2  
        # Assuming standard Type-2 loop (Proportional + Integral path summing into DCO accumulator):  
        # The denominator of the open loop L(z) is (1 - z^-1) if the DCO is the only integrator.  
        # However, standard Type-2 usually implies the integral path has its own accumulator.  
        # Based on previous context (Type-2): Den = (1 - z^-1)^2  
        den_open = np.zeros(curr_N + 2)  
        den_open[0] = 1.0  
        den_open[1] = -2.0  
        den_open[2] = 1.0  
  
        # Closed Loop: H(z) = Num / (Den + Num)  
        num_closed = num_open  
        den_closed = den_open + num_open  
  
        # --- Frequency Response ---  
        sys_closed = signal.TransferFunction(num_closed, den_closed, dt=1/SYMBOL_INTERVAL)  
        _, h_closed = signal.dfreqresp(sys_closed, w=w)  
  
        # Magnitude (JTF)  
        mag_jtf_db = 20 * np.log10(np.abs(h_closed) + 1e-20)  
  
        # JTOL Calculation  
        s_response = 1 - h_closed  
        mag_error = np.abs(s_response) + 1e-20  
        jtol_ui = UI_margin / mag_error  
  
        # --- Plotting ---  
        ax1.semilogx(freq_hz, mag_jtf_db, linewidth=2, color=colors[i], label=label_str)  
        ax2.loglog(freq_hz, jtol_ui, linewidth=2, color=colors[i], label=label_str)  
  
        # Print Bandwidth  
        bw_idx = np.where(mag_jtf_db < -3.0)[0]  
        bw_val = freq_hz[bw_idx[0]]/1e6 if len(bw_idx) > 0 else 0  
        print(f"  {label_str}: BW ~ {bw_val:.2f} MHz")  
  
    # ==========================================  
    # 5. FINAL VISUALS & INFO BOX  
    # ==========================================  
    # Text Box Logic  
    if sweep_type == 'KP_GAIN':  
        fixed_str = f"Fixed:\nKi_dig={fixed_Ki_dig}\nLat={fixed_N}"  
    elif sweep_type == 'KI_GAIN':  
        fixed_str = f"Fixed:\nKp_dig={fixed_Kp_dig}\nLat={fixed_N}"  
    elif sweep_type == 'LATENCY':  
        fixed_str = f"Fixed:\nKp_dig={fixed_Kp_dig}\nKi_dig={fixed_Ki_dig}"  
    else:  
        fixed_str = f"Fixed:\nKp_dig={fixed_Kp_dig}\nKi_dig={fixed_Ki_dig}\nLat={fixed_N}"  
  
    # Add Derived Constant Info  
    info_str = '\n'.join([  
        r'System Constants:',  
        f'Data Rate: {data_rate/1e9:.1f} Gbps',  
        f'RJ Sigma: {RJ_SIGMA} UI',  
        f'Loop Gain Factor: {TOTAL_LOOP_GAIN_FACTOR:.2e}',  
        '-----------------',  
        fixed_str  
    ])  
  
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9)  
    ax1.text(0.02, 0.95, info_str, transform=ax1.transAxes, fontsize=10,  
            verticalalignment='top', bbox=props)  
  
    # Layout Settings  
    ax1.set_title(f'Closed Loop JTF (Sweeping {sweep_type})')  
    ax1.set_ylabel('Magnitude (dB)')  
    ax1.set_xlim([freq_hz[0], data_rate/2])  
    ax1.grid(which='both', linestyle='--', alpha=0.6)  
    ax1.legend(loc='lower left', fontsize=9)  
    ax1.axhline(0, color='k', linewidth=0.5, alpha=0.5)  
  
    ax2.set_title(f'Jitter Tolerance (Sweeping {sweep_type})')  
    ax2.set_xlabel('Frequency (Hz)')  
    ax2.set_ylabel('Tolerance (UIpp)')  
    ax2.set_xlim([freq_hz[0], data_rate/2])  
    ax2.grid(which='both', linestyle='--', alpha=0.6)  
    ax2.legend(loc='lower left', fontsize=9)  
    ax2.axhline(UI_margin, color='r', linestyle=':', label='Margin Limit')  
  
    plt.tight_layout()  
  
    return fig  
  