"""
Phase Interpolator Integration Example

This example demonstrates how to integrate the phase interpolator with
the clock generator to create a practical CDR/sampling scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from clock_generator import generate_clock_signal
from phase_interpolator import phase_interpolate, generate_interpolated_bank


def example_1_single_phase():
    """Example 1: Generate a single interpolated clock at a specific phase."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Phase Interpolation")
    print("="*70)
    
    # Generate quadrature clocks
    t, clk_0, clk_90, clk_180, clk_270, _, _, ui = generate_clock_signal(
        clock_freq_hz=10e9,
        duration_ui=20,
        samples_per_ui=256,
        rj_rms_ui=0.005,
        dj_freq_hz=100e6,
        dj_peak_ui=0.01
    )
    
    # Generate a 45° phase clock
    num_bits = 8
    code_45deg = 128  # Middle of first quadrant
    
    clk_45, _, phase_45, _, ratio_45 = phase_interpolate(
        clk_0, clk_90, clk_180, clk_270, num_bits, code_45deg
    )
    
    print(f"\nGenerated {len(t)} samples ({t[-1]*1e9:.2f} ns)")
    print(f"Clock frequency: 10 GHz, UI: {ui*1e12:.2f} ps")
    print(f"\nInterpolated clock properties:")
    print(f"  Code: {code_45deg}")
    print(f"  Output phase: {phase_45:.2f}°")
    print(f"  Mixing ratio: {ratio_45:.4f}")
    print(f"  RMS amplitude: {np.sqrt(np.mean(clk_45**2)):.4f}")


def example_2_phase_sweep():
    """Example 2: Sweep through all phases and measure eye opening."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Phase Sweep Analysis")
    print("="*70)
    
    # Generate quadrature clocks
    t, clk_0, clk_90, clk_180, clk_270, _, _, _ = generate_clock_signal(
        clock_freq_hz=10e9,
        duration_ui=50,
        samples_per_ui=256,
        rj_rms_ui=0.005,
        dj_freq_hz=100e6,
        dj_peak_ui=0.01
    )
    
    # Simulate a simple data signal
    np.random.seed(42)
    data_bits = np.random.randint(0, 2, 50)
    data_signal = np.repeat(data_bits, 256)
    data_signal = data_signal + 0.1 * np.random.randn(len(data_signal))
    
    # Sweep through all phases in first quadrant
    num_bits = 6
    quadrant_size = 2 ** num_bits
    
    print(f"\nPhase sweep configuration:")
    print(f"  num_bits: {num_bits}")
    print(f"  Phases per quadrant: {quadrant_size}")
    print(f"  Phase step: {90.0/quadrant_size:.2f}°")
    
    # Sample data at each phase
    eye_heights = []
    phases = []
    
    for code in range(quadrant_size):
        clk_interp, _, phase, _, _ = phase_interpolate(
            clk_0, clk_90, clk_180, clk_270, num_bits, code
        )
        
        # Simple metric: average magnitude of sampled data
        sample_magnitude = np.mean(np.abs(data_signal))
        eye_heights.append(sample_magnitude)
        phases.append(phase)
    
    print(f"\nEye opening analysis:")
    print(f"  Phases analyzed: {len(phases)}")
    print(f"  Average sample magnitude: {np.mean(eye_heights):.3f}")
    print(f"  Sample magnitude range: {np.min(eye_heights):.3f} to {np.max(eye_heights):.3f}")


def example_3_phase_bank():
    """Example 3: Generate and use a full phase bank."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Phase Bank Generation and Storage")
    print("="*70)
    
    # Generate quadrature clocks
    t, clk_0, clk_90, clk_180, clk_270, _, _, _ = generate_clock_signal(
        clock_freq_hz=10e9,
        duration_ui=10,
        samples_per_ui=256,
        rj_rms_ui=0.005,
        dj_freq_hz=100e6,
        dj_peak_ui=0.01
    )
    
    # Generate phase bank
    num_bits = 6
    clk_bank, _, phases, _, codes = generate_interpolated_bank(
        clk_0, clk_90, clk_180, clk_270, num_bits
    )
    
    print(f"\nPhase bank configuration:")
    print(f"  num_bits: {num_bits}")
    print(f"  Total phases: {len(clk_bank)}")
    print(f"  Bank shape: {clk_bank.shape}")
    print(f"  Memory size: {clk_bank.nbytes / 1024:.1f} KB")
    print(f"\nPhase coverage:")
    print(f"  Min phase: {phases[0]:.2f}°")
    print(f"  Max phase: {phases[-1]:.2f}°")
    print(f"  Phase spacing: {np.mean(np.diff(phases)):.4f}°")
    
    # Access specific phases
    print(f"\nAccessing specific phases from bank:")
    for idx in [0, 64, 128, 192, 255]:
        print(f"  bank[{idx:3d}]: phase = {phases[idx]:6.2f}°")


def example_4_cdr_feedback():
    """Example 4: Simulate phase locked loop (PLL) feedback."""
    print("\n" + "="*70)
    print("EXAMPLE 4: CDR Phase Feedback Loop")
    print("="*70)
    
    # Generate quadrature clocks
    t, clk_0, clk_90, clk_180, clk_270, _, _, ui = generate_clock_signal(
        clock_freq_hz=10e9,
        duration_ui=100,
        samples_per_ui=256,
        rj_rms_ui=0.01,
        dj_freq_hz=100e6,
        dj_peak_ui=0.02
    )
    
    # Simulate data with phase offset
    np.random.seed(42)
    data_bits = np.random.randint(0, 2, 100)
    data_signal = np.repeat(data_bits, 256)
    phase_offset = 30.0  # 30° intentional offset
    
    # PLL feedback loop
    num_bits = 8
    code = 0  # Start at 0°
    Kp = 10   # Proportional gain
    
    print(f"\nCDR loop simulation:")
    print(f"  Target phase: {phase_offset:.2f}°")
    print(f"  Proportional gain: {Kp}")
    print(f"  Resolution: num_bits={num_bits}, step={360/4/2**num_bits:.3f}°")
    
    # Simulate feedback loop
    estimated_phases = []
    code_values = []
    
    for iteration in range(20):
        clk_interp, _, phase, _, ratio = phase_interpolate(
            clk_0, clk_90, clk_180, clk_270, num_bits, code
        )
        
        # Measure phase error (simplified)
        phase_error = phase_offset - phase
        
        # Update code based on phase error
        code_update = int(Kp * phase_error * (2**num_bits) / 90.0)
        code = max(0, min(1023, code + code_update))
        
        estimated_phases.append(phase)
        code_values.append(code)
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration:2d}: code={code:4d}, phase={phase:6.2f}°, error={phase_error:+6.2f}°")
    
    print(f"\nFinal convergence:")
    print(f"  Final code: {code_values[-1]}")
    print(f"  Final phase: {estimated_phases[-1]:.2f}°")
    print(f"  Final error: {phase_offset - estimated_phases[-1]:.2f}°")


def example_5_visualization():
    """Example 5: Create visualization of phase interpolator."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Visualization")
    print("="*70)
    
    # Generate clocks
    t, clk_0, clk_90, clk_180, clk_270, _, _, _ = generate_clock_signal(
        clock_freq_hz=10e9,
        duration_ui=5,
        samples_per_ui=256,
        rj_rms_ui=0.005,
        dj_freq_hz=100e6,
        dj_peak_ui=0.01
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Quadrature clocks
    ax = axes[0]
    ax.plot(t*1e12, clk_0, label='clk_0 (0°)', linewidth=2, color='blue')
    ax.plot(t*1e12, clk_90, label='clk_90 (90°)', linewidth=2, color='red')
    ax.plot(t*1e12, clk_180, label='clk_180 (180°)', linewidth=2, color='green')
    ax.plot(t*1e12, clk_270, label='clk_270 (270°)', linewidth=2, color='orange')
    ax.set_title("Four Quadrature Clock Signals", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Interpolated clocks at various phases
    ax = axes[1]
    num_bits = 5  # 32 phases per quadrant
    phase_codes = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    colors = plt.cm.hsv(np.linspace(0, 1, len(phase_codes)))
    
    for i, code in enumerate(phase_codes):
        clk_interp, _, phase, _, _ = phase_interpolate(
            clk_0, clk_90, clk_180, clk_270, num_bits, code
        )
        ax.plot(t*1e12, clk_interp, label=f'{phase:.1f}°', linewidth=1.5, color=colors[i], alpha=0.7)
    
    ax.set_title("Interpolated Clocks at Various Phases", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig("plots/phase_interpolator_example.png", dpi=150, bbox_inches='tight')
    print("✓ Plot saved to 'plots/phase_interpolator_example.png'")


if __name__ == '__main__':
    """Run all examples."""
    print("\n" + "="*70)
    print("PHASE INTERPOLATOR - INTEGRATION EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_single_phase()
    example_2_phase_sweep()
    example_3_phase_bank()
    example_4_cdr_feedback()
    example_5_visualization()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
