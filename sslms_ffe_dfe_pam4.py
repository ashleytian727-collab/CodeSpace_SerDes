
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def pam4_slicer(y_n, decision_levels=None):
    """
    PAM4 multi-level slicer.
    
    Args:
        y_n: Input sample
        decision_levels: List of 4 decision levels [L0, L1, L2, L3] in ascending order
                        Default: [-3, -1, 1, 3]
    
    Returns:
        Sliced symbol (-3, -1, 1, or 3)
    """
    if decision_levels is None:
        decision_levels = [-3, -1, 1, 3]
    
    if y_n <= decision_levels[0]:
        return -3
    elif y_n <= decision_levels[1]:
        return -1
    elif y_n <= decision_levels[2]:
        return 1
    else:
        return 3

def sslms_ffe_dfe_pam4(signal, os, modulation='NRZ', sampling_offset=0, 
                       num_ffe_taps=3, num_dfe_taps=5, 
                       mu_ffe=0.01, mu_dfe=0.01, 
                       algorithm='SSLMS', dLev_init=0.0, delta_dLev=1e-3,
                       sslms_start_iter=0, pam4_levels=None, 
                       plot=False, Ts=None):
    """
    SSLMS/LMS FFE+DFE equalizer with adaptive decision level for NRZ and PAM4 signals.
    
    Args:
        signal: Input signal (ndarray)
        os: Oversampling factor
        modulation: 'NRZ' for binary or 'PAM4' for quaternary modulation
        sampling_offset: Sample offset for downsampling
        num_ffe_taps: Number of FFE tap weights
        num_dfe_taps: Number of DFE tap weights
        mu_ffe: Learning rate for FFE taps
        mu_dfe: Learning rate for DFE taps
        algorithm: 'LMS' or 'SSLMS' (sign-sign LMS)
        dLev_init: Initial decision level
        delta_dLev: Hysteresis for decision level update
        sslms_start_iter: Iteration after which to apply SSLMS (before: LMS)
        pam4_levels: PAM4 decision levels [L0, L1, L2, L3]
        plot: Whether to generate plots
        Ts: Sampling period for time axis
    
    Returns:
        Tuple containing:
        - ffe_taps: Final FFE tap weights
        - dfe_taps: Final DFE tap weights
        - dLev: Final decision level (Vref)
        - ffe_taps_history: History of FFE tap values over time
        - dfe_taps_history: History of DFE tap values over time
        - dLev_history: History of decision level over time
        - equalized_signal: Full-resolution equalized signal
        - decisions: Full-resolution symbol decisions
    """  
    if modulation not in ['NRZ', 'PAM4']:
        raise ValueError("modulation must be 'NRZ' or 'PAM4'")
    
    if algorithm not in ['LMS', 'SSLMS']:
        raise ValueError("algorithm must be 'LMS' or 'SSLMS'")
    
    if pam4_levels is None:
        pam4_levels = [-3, -1, 1, 3]
    
    signal_downsampled = signal[sampling_offset::os]
    
    # FFE tap weights (feedforward, operates on input signal history)
    ffe_taps = np.zeros(num_ffe_taps)
    ffe_taps_history = np.zeros((num_ffe_taps, len(signal)))
    
    # DFE tap weights (feedback, operates on past decisions)
    dfe_taps = np.zeros(num_dfe_taps)
    dfe_taps_history = np.zeros((num_dfe_taps, len(signal)))
    
    # Adaptive decision level (for NRZ/PAM4)
    dLev = dLev_init
    dLev_history = np.zeros(len(signal))
    
    # History buffers
    x_history = np.zeros(num_ffe_taps)  # Input signal history for FFE
    d_history = np.zeros(num_dfe_taps)  # Decision history for DFE
    
    # Index of history arrays
    history_idx = sampling_offset
    iteration = 0
    
    for x_n in signal_downsampled:
        # FFE processing: y_ffe = FFE_output
        y_ffe = np.dot(ffe_taps, x_history) + x_n
        
        # DFE processing: y_dfe = y_ffe - DFE_feedback
        y_dfe = y_ffe - np.dot(dfe_taps, d_history)
        
        # Symbol decision based on modulation type
        if modulation == 'NRZ':
            # Binary slicer with adaptive threshold
            d_n = 1 if y_dfe > dLev else -1
        else:  # PAM4
            # Multi-level slicer
            d_n = pam4_slicer(y_dfe, pam4_levels)
        
        # Error signal
        if modulation == 'NRZ':
            e_cont = y_dfe - dLev * d_n
        else:  # PAM4
            e_cont = y_dfe - d_n
        
        # Adaptive filter updates
        if algorithm == 'SSLMS':
            if iteration > sslms_start_iter:
                # SSLMS: use sign of error
                error_sign = np.sign(e_cont)
                ffe_taps += mu_ffe * error_sign * x_history
                dfe_taps += mu_dfe * error_sign * d_history
        else:  # LMS
            # LMS: use full error
            ffe_taps += mu_ffe * e_cont * x_history
            dfe_taps += mu_dfe * e_cont * d_history
        
        # Decision level adaptation (for NRZ only, PAM4 levels are fixed)
        if modulation == 'NRZ':
            if iteration <= sslms_start_iter:
                dLev += 0.01 * e_cont
            else:
                up = 1 if (y_dfe * d_n > dLev + delta_dLev) else 0
                down = 1 if (y_dfe * d_n < dLev - delta_dLev) else 0
                dLev += 0.01 * (up - down)
        
        # Update history buffers
        x_history = np.roll(x_history, 1)
        x_history[0] = x_n
        
        d_history = np.roll(d_history, 1)
        d_history[0] = d_n
        
        # Store history
        if history_idx + os < len(signal):
            ffe_taps_history[:, history_idx + os] = ffe_taps
            dfe_taps_history[:, history_idx + os] = dfe_taps
            dLev_history[history_idx:history_idx + os] = dLev
        else:
            ffe_taps_history[:, history_idx] = ffe_taps[:]
            dfe_taps_history[:, history_idx] = dfe_taps[:]
            dLev_history[history_idx] = dLev
        
        history_idx += os
        iteration += 1
    
    # Optional plotting
    if plot:
        plots_dir = Path(__file__).resolve().parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Generate time axis
        if Ts is None:
            time_axis = np.arange(len(signal))
            time_label = 'Sample Index'
        else:
            time_axis = np.arange(len(signal)) * Ts
            time_label = 'Time (s)'
        
        # Plot 1: FFE tap weights history
        fig, ax1 = plt.subplots(figsize=(12, 5))
        for i in range(min(num_ffe_taps, 3)):
            ax1.plot(time_axis, ffe_taps_history[i, :], label=f'FFE Tap {i}', linewidth=1.5)
        ax1.set_xlabel(time_label)
        ax1.set_ylabel('Tap Value')
        ax1.set_title('Evolution of FFE Tap Weights')
        ax1.legend()
        ax1.grid(True)
        plot1_path = plots_dir / 'ffe_dfe_ffe_taps_history.png'
        fig.savefig(plot1_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot1_path}")
        
        # Plot 2: DFE tap weights history
        fig, ax2 = plt.subplots(figsize=(12, 5))
        for i in range(min(num_dfe_taps, 3)):
            ax2.plot(time_axis, dfe_taps_history[i, :], label=f'DFE Tap {i}', linewidth=1.5)
        ax2.set_xlabel(time_label)
        ax2.set_ylabel('Tap Value')
        ax2.set_title('Evolution of DFE Tap Weights')
        ax2.legend()
        ax2.grid(True)
        plot2_path = plots_dir / 'ffe_dfe_dfe_taps_history.png'
        fig.savefig(plot2_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot2_path}")
        
        # Plot 3: Signal with decision levels
        fig, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(time_axis, signal, label='Input Signal', linewidth=1, alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        if modulation == 'NRZ':
            ax3_twin = ax3.twinx()
            ax3_twin.plot(time_axis, dLev_history, label='dLev', color='red', linewidth=1.5)
            ax3_twin.plot(time_axis, -dLev_history, label='-dLev', color='red', linewidth=1.5, linestyle='--')
            ax3_twin.set_ylabel('Decision Level', color='red')
            ax3.set_title('Signal with Adaptive Decision Levels (NRZ)')
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
        else:  # PAM4
            for level in pam4_levels:
                ax3.axhline(y=level, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            ax3.set_title('Signal with PAM4 Decision Levels')
            ax3.legend(loc='upper right')
        
        ax3.set_xlabel(time_label)
        ax3.set_ylabel('Signal Amplitude', color='blue')
        ax3.grid(True, alpha=0.3)
        plot3_path = plots_dir / 'ffe_dfe_signal_with_levels.png'
        fig.savefig(plot3_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot3_path}")
    
    # Compute full-resolution equalized signal and decisions
    equalized_signal = np.zeros(len(signal))
    decisions = np.zeros(len(signal))
    
    # Initialize history buffers
    x_history_full = np.zeros(num_ffe_taps)
    d_history_full = np.zeros(num_dfe_taps)
    
    # Process all samples
    for i in range(len(signal)):
        # Get current tap weights
        if i >= ffe_taps_history.shape[1]:
            ffe_taps_final = ffe_taps_history[:, -1]
            dfe_taps_final = dfe_taps_history[:, -1]
            dLev_final = dLev_history[-1]
        else:
            ffe_taps_final = ffe_taps_history[:, i]
            dfe_taps_final = dfe_taps_history[:, i]
            dLev_final = dLev_history[i]
        
        # FFE processing
        y_ffe = np.dot(ffe_taps_final, x_history_full) + signal[i]
        
        # DFE processing
        y_eq = y_ffe - np.dot(dfe_taps_final, d_history_full)
        equalized_signal[i] = y_eq
        
        # Make symbol decision
        if modulation == 'NRZ':
            d_i = 1 if y_eq > dLev_final else -1
        else:  # PAM4
            d_i = pam4_slicer(y_eq, pam4_levels)
        
        decisions[i] = d_i
        
        # Update history buffers
        x_history_full = np.roll(x_history_full, 1)
        x_history_full[0] = signal[i]
        
        d_history_full = np.roll(d_history_full, 1)
        d_history_full[0] = d_i
    
    return ffe_taps, dfe_taps, dLev, ffe_taps_history, dfe_taps_history, dLev_history, equalized_signal, decisions





sslms_ffe_dfe_pam4(
    signal,                    # 输入信号
    os=4,                      # 过采样因子
    modulation='NRZ',          # 'NRZ' 或 'PAM4'
    num_ffe_taps=3,            # FFE级数
    num_dfe_taps=5,            # DFE级数
    mu_ffe=0.01,               # FFE学习率
    mu_dfe=0.01,               # DFE学习率
    algorithm='SSLMS',         # 'LMS' 或 'SSLMS'
    plot=True                  # 生成对比图表
)

使用示例：

from sslms_ffe_dfe_pam4 import sslms_ffe_dfe_pam4
import numpy as np

# 生成测试信号
signal = np.random.randn(10000)

# 1️⃣ NRZ模式，SSLMS算法
ffe, dfe, dlev, ffe_hist, dfe_hist, dlev_hist, eq_sig, dec = sslms_ffe_dfe_pam4(
    signal, 
    os=4,                          # 过采样4x
    modulation='NRZ',              # 二进制调制
    num_ffe_taps=3,                # 3级前馈均衡
    num_dfe_taps=5,                # 5级反馈均衡
    algorithm='SSLMS',             # 符号-符号LMS
    plot=True                      # 生成图表
)

# 2️⃣ PAM4模式，LMS算法
ffe, dfe, dlev, ffe_hist, dfe_hist, dlev_hist, eq_sig, dec = sslms_ffe_dfe_pam4(
    signal,
    os=4,
    modulation='PAM4',             # 四进制调制
    num_ffe_taps=4,
    num_dfe_taps=6,
    mu_ffe=0.005,                  # FFE学习率
    mu_dfe=0.005,                  # DFE学习率
    algorithm='LMS',               # 完整LMS
    pam4_levels=[-3, -1, 1, 3],    # 自定义决策电平
    plot=True
)

# 3️⃣ 访问结果
print(f"最终FFE权重: {ffe}")
print(f"最终DFE权重: {dfe}")
print(f"均衡后信噪比提升: {np.std(eq_sig):.4f}")
print(f"符号错误率: {np.mean(dec != signal):.6f}")




















# FFE + DFE Equalizer Implementation

"""
This file implements a Feedforward Equalizer (FFE) and a Decision Feedback Equalizer (DFE) supporting both NRZ and PAM4 modulations. 
The LMS (Least Mean Squares) and SSLMS (Steady State Least Mean Squares) adaptive algorithms are utilized for equalization.
"""

import numpy as np

class FFE:
    def __init__(self, taps):
        self.taps = taps
        self.weights = np.zeros(taps)

    def adapt(self, input_signal, desired_signal, mu):
        for n in range(len(input_signal) - self.taps):
            error = desired_signal[n] - self.filter(input_signal[n:n+self.taps])
            self.weights += mu * error * input_signal[n:n+self.taps]

    def filter(self, input_signal):
        return np.dot(self.weights, input_signal)

class DFE:
    def __init__(self, feedforward_taps, feedback_taps):
        self.ffe = FFE(feedforward_taps)
        self.feedback_weights = np.zeros(feedback_taps)

    def adapt(self, input_signal, desired_signal, mu):
        # Adapt using FFE first
        self.ffe.adapt(input_signal, desired_signal, mu)
        # DFE adaptation logic here

class AdaptiveEqualizer:
    def __init__(self, ffe_taps, dfe_feedforward_taps, dfe_feedback_taps):
        self.dfe = DFE(dfe_feedforward_taps, dfe_feedback_taps)

    def equalize(self, input_signal, desired_signal, mu):
        self.dfe.adapt(input_signal, desired_signal, mu)

# Sample usage
if __name__ == '__main__':
    # Example values
    mu = 0.01
    input_signal = np.random.randn(1000)
    desired_signal = np.random.randn(1000)
    equalizer = AdaptiveEqualizer(5, 5, 5)
    equalizer.equalize(input_signal, desired_signal, mu)
