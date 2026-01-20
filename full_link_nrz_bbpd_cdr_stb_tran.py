import numpy as np  
from collections import deque  
import matplotlib.pyplot as plt  
import scipy as sp  
from matplotlib.ticker import FuncFormatter  
from scipy.interpolate import PchipInterpolator  
import warnings  
import os  
import sys  
# sys.path.append(r"/Users/tamaldas/Documents/Scripts/Python/serdespy-main/")  
sys.path.append(r"/Users/tamal.das2/Projects/SerDes/serdespy")  
import serdespy as sdp  
from clock_generator import generate_clock_with_jitter  
from phase_interpolator import PhaseInterpolator, create_dnl_profile, sin_to_square  
from sparam_functions import *  
from dynamic_plotting import *  
import rx_cdr_functions as cdr  
from bbpd_dig_cdr_stab import bbpd_dig_cdr_stab  
from bbpd_cdr_loop_tran import bbpd_cdr_loop_tran  
  
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
def main():  
    # Constants from the main script  
    ADD_RAND_JITTER = True  
    PLOT_FREQ_RESP = False  
    PLOT_PULSE_RESP = False # No longer plotting pulse response here  
    PLOT_RX_CLK_IN = False  
    PLOT_RX_PI_CLK = False  
    PLOT_RESULTS = True  
    PLOT_BODE_ANALYSIS = True  
  
    ###################  Global Variables ###################   
    data_rate = 56.1e9 #NRZ  
    f_nyq = data_rate / 2  
    g['ui'] = 1 / data_rate  
    g['os'] = 128 # Samples per symbol  
    g['tx_launch_amp'] = 0.6  
    g['rterm_source'] = 50  
    g['rterm_sink'] = 50  
    Ts = g['ui'] / g['os'] # Time step  
      
    ###################  TX local Variables ###################   
    RJ_SIGMA = 0.02 #UI  
    ###################  RX local variables ###################  
    # --- Sampler Parameters ---  
    SAMPLER_C2Q = 5e-12  
    SAMPLER_SENSE = 5e-3  
    # --- Digital Loop Filter (DLF) Constants ---  
    KP_GAIN = 2  
    KI_GAIN = 1  
    PI_NUM_BITS = 6  
    TOTAL_PI_CODES = 4 * (2**PI_NUM_BITS) # 256 total codes (0-255)  
    PHASE_INTG_DITHER_BITS = 3 #Dp  
    FREQ_INTG_DITHER_BITS = 4 #Df  
    DECIMATION_FACTOR = 20 # L UI  
    SYMBOL_INTERVAL = DECIMATION_FACTOR/data_rate  
    # --- Physical Gain Factors  
    KDPC_GAIN = 1.0 / (2**(PI_NUM_BITS+PHASE_INTG_DITHER_BITS)) # Kdpc: Phase step size in UI per LSB  
    KBB_GAIN = 1.0 / (RJ_SIGMA*np.sqrt(2*np.pi))                # Kbb: BBPD gain (1/WordLength, WordLength=20)  
    KD_GAIN = 0.50                                              # Kd: Decimation gain (assumed 1.0)  
    TOTAL_LOOP_GAIN_FACTOR = KBB_GAIN * KD_GAIN * KDPC_GAIN  
    # --- Latency in Word Clock Cycles (M)  
    DESERIALIZER_LATENCY_WORDS = 2   
    BBPD_LATENCY_WORDS = 1  
    VOTER_LATENCY_WORDS = 1  
    PROP_GAIN_LATENCY_WORDS = 1   
    FREQ_INT_LATENCY_WORDS = 1  
    PI_ACC_LATENCY_WORDS = 1  
    TOTAL_LOOP_LATENCY_WORDS = DESERIALIZER_LATENCY_WORDS + BBPD_LATENCY_WORDS + VOTER_LATENCY_WORDS + max(PROP_GAIN_LATENCY_WORDS, FREQ_INT_LATENCY_WORDS) + PI_ACC_LATENCY_WORDS # M SYMBOL_INTERVAL  
    # --- Decimation facotrs ---  
    DESERIALIZATION_FACTOR = 10   
    VOTER_DECIMATION_FACTOR = 1  
  
    ################### TX: Input Generation ###################   
    # --- generate binary data  
    data = sdp.prbs13(1)  
    data = np.concatenate((data,data,data,data,data,data), axis=0)  
    # ---  --- generate Baud-Rate sampled signal from data  
    signal_BR = sdp.nrz_input_BR(data)  
    # ---  --- oversampled signal  
    signal_ideal = 0.5*g['tx_launch_amp'] * np.repeat(signal_BR, g['os'])  
    print("PRBS signal train is generated.\n")  
    # --- TX signal with jitter  
    if ADD_RAND_JITTER:  
        signal_jitter = sdp.gaussian_jitter(signal_ideal, g['ui'], len(data), g['os'], stdev=1000e-15)  
        print("Random Jitter is added.\n")  
    else:  
        signal_jitter = signal_ideal  
  
    ###################  RX clock Generation ###################   
    duration = len(signal_jitter) * Ts  
    #print(f"RX sim Duration: {duration}seconds")  
    RJ_STD_DEV = RJ_SIGMA * g['ui']  
    DJ_FREQ = 100e6  
    DJ_AMPLITUDE = 0.00 * 2 * g['ui']  
    RX_CLOCK_FREQUENCY = data_rate/2  
    t_cdr, rx_clk_i, rx_clk_i_b, rx_clk_q, rx_clk_q_b, sample_rate = generate_clock_with_jitter(  
        clock_freq= RX_CLOCK_FREQUENCY,  
        duration=duration,  
        oversample_ratio=g['os'] * 2,  
        rj_std_dev=RJ_STD_DEV,  
        dj_freq=DJ_FREQ,  
        dj_amplitude=DJ_AMPLITUDE,  
        filter_bw_factor=1.5,  
        generate_q_clock=True  
    )  
    fig = bbpd_dig_cdr_stab(data_rate, RJ_SIGMA, '',   
                      KP_GAIN, KI_GAIN, PI_NUM_BITS, PHASE_INTG_DITHER_BITS, FREQ_INTG_DITHER_BITS, DECIMATION_FACTOR,   
                      SYMBOL_INTERVAL, TOTAL_LOOP_GAIN_FACTOR, TOTAL_LOOP_LATENCY_WORDS)  
    fig.savefig('bbpd_cdr_stability.png')  
  
    ###################  TX to RX link ###################    
    s_param_dir = "/Users/tamal.das2/Projects/SerDes/Channels/"   
    if not os.path.exists(s_param_dir):  
        print(f"ERROR: S-parameter directory not found at '{s_param_dir}'")  
        return  
  
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
        pkg_s=os.path.join(s_param_dir, 'PKG100GEL_95ohm_30mm_50ohmPort.s4p'),  
        # Sink Package  
        #pkg_l=os.path.join(s_param_dir, 'PKG100GEL_95ohm_30mm_50ohmPort.s4p'),  
        # Channel  
        #ch=os.path.join(s_param_dir, 'C2M_3p6in_100Ohms_thru1.s4p'),  
        s_tcoil=True,  
        s_tcoil_split = True,  
        l_tcoil=False,  
        l_tcoil_split = True,  
        pkg_s_portswap=True,  
        pkg_l_portswap=True  
    )  
      
    if g['H_ch'] is None: return   
    print("\nFull link: Transfer Function evaluation completed.\n")  
      
    imp_ch = impinterp(np.fft.irfft(g['H_ch']), round( (1/Ts) / (2*g['f'][-1])))  
    imp_ch /= np.sum(np.abs(imp_ch))  
      
    signal_filtered = sp.signal.fftconvolve(signal_jitter, imp_ch, mode="full")  
    signal_filtered = signal_filtered[0:len(signal_jitter)]  
  
    ###################  CDR LOOP TRAN ###################    
    rxpi_sq_i_final = bbpd_cdr_loop_tran(  
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
    )  
  
    ###################  EYE PLOTS ###################    
    #sdp.simple_eye(signal_ideal, g['os']*3, 100, Ts, "{}Gbps 2-PAM Signal".format(data_rate/1e9), res=100, linewidth=1.5)  
    idx_min = 700000  
    idx_max = 800000  
    arr = signal_filtered[idx_min:idx_max]  
    crossings = np.where(arr[:-1] * arr[1:] < 0)[0]  
    zero_cross = crossings[0] if len(crossings) > 0 else None  
    sdp.simple_eye(signal_filtered[idx_min+zero_cross+int(g['os']/2):], g['os']*2, 2000, Ts, "{}Gbps 2-PAM Signal after Channel".format(round(data_rate/1e9)),res=100)  
    sdp.simple_eye(np.array(rxpi_sq_i_final[900*20*g['os']:])*0.5, g['os']*2, 2000, Ts, "{}GHZ Half Rate Recovered Clock".format(round(RX_CLOCK_FREQUENCY/1e9)),res=100)  
  
if __name__ == "__main__":  
    main()