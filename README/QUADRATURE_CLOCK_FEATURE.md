# Quadrature Clock Generator Feature

## Overview
Enhanced `clock_generator.py` to generate 4 quadrature clock signals (0°, 90°, 180°, 270°) instead of just complementary pair (clk_p/clk_n).

## Changes Made

### 1. **Phase Generation (Lines 60-80)**
- Added quadrature clock generation with proper phase relationships
- `clk_0`: 0° phase (sin) - equivalent to original `clk_p`
- `clk_90`: 90° phase (cos) - newly generated and filtered
- `clk_180`: 180° phase (-sin) - equivalent to original `clk_n`
- `clk_270`: 270° phase (-cos) - newly generated and filtered
- All 4 clocks pass through the same LPF filter for consistent filtering

### 2. **Return Signature (Line 127)**
**Before:**
```python
return t, clk_p, clk_n, f_welch, phase_noise_dbchz, ui
```

**After:**
```python
return t, clk_0, clk_90, clk_180, clk_270, f_welch, phase_noise_dbchz, ui
```

Changed from 6 return values to 8 return values

### 3. **Function Signature (Docstring)**
Updated Returns section to document all 8 return values:
- `t` (1D ndarray): Time vector
- `clk_0` (1D ndarray): 0° quadrature clock signal
- `clk_90` (1D ndarray): 90° quadrature clock signal
- `clk_180` (1D ndarray): 180° quadrature clock signal
- `clk_270` (1D ndarray): 270° quadrature clock signal
- `f_welch` (1D ndarray): Frequency vector for phase noise
- `phase_noise_dbchz` (1D ndarray): Phase noise in dBc/Hz
- `ui` (float): Unit interval in seconds

### 4. **Example Usage (Line 148)**
Updated unpacking in `__main__` section:
```python
t, clk_0, clk_90, clk_180, clk_270, f_noise, pn_dbchz, ui = generate_clock_signal(...)
```

### 5. **Visualization (Lines 167-206)**
Enhanced plotting to show all 4 quadrature clocks:
- **Subplot 1**: Clocks 0° and 90°
- **Subplot 2**: Clocks 180° and 270°
- **Subplot 3**: Phase noise profile (unchanged)

Color scheme:
- 0° Clock: Blue
- 90° Clock: Red
- 180° Clock: Green
- 270° Clock: Orange

## Verification Results

### Return Structure
✓ Function returns 8-element tuple correctly
✓ All 4 clock signals have correct shape (matches time vector)
✓ Frequency and phase noise arrays included

### Clock Generation
✓ clk_0 and clk_180 have matching RMS values (opposite phase)
✓ clk_90 and clk_270 have matching RMS values (opposite phase)
✓ Phase relationships verified: 90° spacing between consecutive clocks
✓ All filtered uniformly through same LPF

### Phase Noise
✓ DJ (Deterministic Jitter) peak found at expected frequency
✓ RJ (Random Jitter) broadband floor measured correctly
✓ Phase noise profile generated and saved

## Usage Example

```python
from clock_generator import generate_clock_signal

# Generate 10 GHz clock with quadrature outputs
t, clk_0, clk_90, clk_180, clk_270, f_noise, pn_dbchz, ui = generate_clock_signal(
    clock_freq_hz=10e9,
    duration_ui=2000,
    samples_per_ui=256,
    rj_rms_ui=0.01,      # 1% RJ
    dj_freq_hz=100e6,    # 100 MHz DJ
    dj_peak_ui=0.025     # 2.5% DJ
)

print(f"Generated {len(t)} samples")
print(f"Unit Interval: {ui*1e12:.2f} ps")
# Use any/all of the 4 quadrature clocks in downstream processing
```

## Backward Compatibility

⚠️ **Breaking Change**: Return signature changed from 6 to 8 values

Scripts calling `generate_clock_signal()` expecting 6 return values will fail.

**Search Results**: No downstream scripts in the workspace currently use `generate_clock_signal()` - only used internally in `clock_generator.py` `__main__` section.

## Technical Details

### Phase Relationships
All clocks share the same frequency but are phase-shifted:
- clk_0: φ = 0°
- clk_90: φ = 90° (π/2 radians)
- clk_180: φ = 180° (π radians)
- clk_270: φ = 270° (3π/2 radians)

### Filtering
All 4 quadrature clocks pass through the same LPF filter:
```
Design: 5th order Butterworth
Cutoff: 3x clock frequency
Purpose: Remove high-frequency jitter artifacts
```

This ensures uniform frequency response across all quadrature outputs.

## Testing

Run `clock_generator.py` directly to test:
```bash
python3 clock_generator.py
```

Output:
- Console: Verification of DJ peak and RJ floor
- File: `plots/clock_generator_output.png` with all visualizations
