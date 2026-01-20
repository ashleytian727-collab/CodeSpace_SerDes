# Phase Interpolator - Quick Reference Guide (v2.0)

**NEW in v2.0:** Complementary clocks, DNL/INL error modeling, enhanced error analysis

## Function Signatures

### Single Phase Interpolation (with all features)
```python
clk_interp, clk_interp_n, phase, phase_error, ratio = phase_interpolate(
    clk_0, clk_90, clk_180, clk_270,  # 4 quadrature clocks
    num_bits,                         # phase resolution (2^num_bits phases per quadrant)
    code,                             # control code [0, 4*2^num_bits - 1]
    dnl_profile=None,                 # Optional DNL errors (deg)
    inl_profile=None,                 # Optional INL errors (deg)
    complementary=False               # Optional: return clk_interp_n
)
# Returns: (clk, clk_n or None, phase_deg, phase_error_deg, mixing_ratio)
```

### Generate All Phases (with error modeling)
```python
clk_bank, clk_bank_n, phases, phase_errors, codes = generate_interpolated_bank(
    clk_0, clk_90, clk_180, clk_270,  # 4 quadrature clocks
    num_bits,                          # phase resolution
    dnl_profile=None,                  # Optional
    inl_profile=None,                  # Optional
    complementary=False                # Optional
)
# clk_bank[code] gives the interpolated clock for that code
# clk_bank_n[code] gives complementary clock (if enabled)
```

### Create DNL/INL Error Profiles
```python
# DNL: Differential Non-Linearity (step-to-step errors)
dnl = create_dnl_profile(
    num_bits,
    dnl_magnitude=0.5,      # Peak error in degrees (typical: 0.1-1.0°)
    shape='random'          # 'random', 'sine', 'sawtooth', 'monotonic'
)

# INL: Integral Non-Linearity (cumulative errors)
inl = create_inl_profile(
    num_bits,
    inl_magnitude=1.0,      # Peak error in degrees (typical: 0.5-5.0°)
    shape='sine'            # 'random', 'sine', 'quadratic', 'linear'
)
```

## Code Mapping

| Code Range | Quadrant | Clock Mix | Phase Range |
|------------|----------|-----------|------------|
| 0 to 2^N-1 | 0 | clk_0 + clk_90 | 0° to 90° |
| 2^N to 2×2^N-1 | 1 | clk_90 + clk_180 | 90° to 180° |
| 2×2^N to 3×2^N-1 | 2 | clk_180 + clk_270 | 180° to 270° |
| 3×2^N to 4×2^N-1 | 3 | clk_270 + clk_0 | 270° to 360° |

*where N = num_bits*

## Common Configurations

### Fine Resolution (num_bits=8)
```python
# 1024 total phases, 0.35° spacing
clk, clk_n, phase, error, ratio = phase_interpolate(
    clk_0, clk_90, clk_180, clk_270, 8, code, complementary=True
)
# Valid codes: 0 to 1023
```

### Medium Resolution (num_bits=6)
```python
# 256 total phases, 1.41° spacing
clk, clk_n, phase, error, ratio = phase_interpolate(
    clk_0, clk_90, clk_180, clk_270, 6, code, complementary=True
)
# Valid codes: 0 to 255
```

### Coarse Resolution (num_bits=4)
```python
# 64 total phases, 5.63° spacing
clk, clk_n, phase, error, ratio = phase_interpolate(
    clk_0, clk_90, clk_180, clk_270, 4, code, complementary=True
)
# Valid codes: 0 to 63
```

## Phase Calculation

Given a code, output phase is calculated as:

```
quadrant = code // 2^num_bits
local_code = code % 2^num_bits
ratio = local_code / 2^num_bits
phase_ideal = (quadrant × 90°) + (ratio × 90°)
phase_actual = phase_ideal + dnl_profile[code] + inl_profile[code]
```

### Example
For num_bits=8:
- code=0 → phase = 0.00°
- code=128 → phase = 45.00° (0° + 0.5×90°)
- code=256 → phase = 90.00°
- code=384 → phase = 135.00°
- code=512 → phase = 180.00°
- code=768 → phase = 270.00°
- code=1023 → phase = 359.65°

## NEW in v2.0: Using Complementary Clocks

```python
# Generate complementary clock pair (180° out of phase)
clk_p, clk_n, phase, error, ratio = phase_interpolate(
    clk_0, clk_90, clk_180, clk_270, num_bits=8, code=200,
    complementary=True
)

# Verify complementarity
diff = clk_p + clk_n
assert np.max(np.abs(diff)) < 1e-10, "Not complementary!"
# For ideal complementary: clk_p = -clk_n (180° apart)
```

## NEW in v2.0: Using DNL/INL Profiles

```python
# Create realistic error profiles
dnl = create_dnl_profile(num_bits=8, dnl_magnitude=0.3, shape='random')
inl = create_inl_profile(num_bits=8, inl_magnitude=0.8, shape='sine')

# Apply errors to interpolation
clk, clk_n, phase_actual, total_error, ratio = phase_interpolate(
    clk_0, clk_90, clk_180, clk_270, 8, 256,
    dnl_profile=dnl,
    inl_profile=inl,
    complementary=True
)

print(f"Ideal phase: 90.00°")
print(f"Actual phase: {phase_actual:.2f}° (error: {total_error:+.3f}°)")
```

## Available DNL Shapes

| Shape | Characteristic | Use Case |
|-------|---|---|
| **random** | Random walk errors | Most realistic; typical hardware |
| **sine** | Sinusoidal pattern | Systematic phase-dependent errors |
| **sawtooth** | Ramps per quadrant | Monotonic drift per quadrant |
| **monotonic** | Linear drift | Systematic temperature-dependent drift |

## Available INL Shapes

| Shape | Characteristic | Use Case |
|-------|---|---|
| **random** | Random walk cumulative | Realistic accumulation of DNL |
| **sine** | Smooth sinusoid | Phase compression/expansion |
| **quadratic** | Parabolic curve | Non-linear interpolation effects |
| **linear** | Linear ramp | Monotonic gain/offset errors |

## Error Cases

```python
# ✗ Invalid code (out of range)
phase_interpolate(..., 8, 1024)  # Max valid = 1023
# ValueError: code must be an integer in range [0, 1023], got 1024

# ✗ Invalid num_bits
phase_interpolate(..., 0, 100)   # num_bits must be ≥ 1
# ValueError: num_bits must be a positive integer, got 0

# ✗ Mismatched clock lengths
phase_interpolate(clk_0[:-1], clk_90, clk_180, clk_270, 8, 500)
# ValueError: All clock signals must have the same length

# ✗ Negative code
phase_interpolate(..., 8, -1)
# ValueError: code must be an integer in range [0, 1023], got -1

# ✗ Wrong profile length
dnl_wrong = np.zeros(100)  # Should be 256 for num_bits=6
phase_interpolate(..., 6, 50, dnl_profile=dnl_wrong)
# ValueError: DNL profile length (100) must match total phases (256)
```

## Usage Pattern for CDR/Sampling

```python
from clock_generator import generate_clock_signal
from phase_interpolator import (
    phase_interpolate,
    create_dnl_profile,
    create_inl_profile
)

# 1. Generate quadrature reference clocks
t, clk_0, clk_90, clk_180, clk_270, f, pn, ui = generate_clock_signal(
    clock_freq_hz=10e9,
    duration_ui=1000,
    samples_per_ui=256
)

# 2. Create error profiles (optional, for realism)
dnl = create_dnl_profile(8, dnl_magnitude=0.3, shape='random')
inl = create_inl_profile(8, inl_magnitude=0.8, shape='sine')

# 3. Set phase resolution
num_bits = 8  # Fine control: 1024 phases

# 4. Sweep through phases or jump to specific phase
for code in range(0, 256):  # Sweep first quadrant (0° to 90°)
    clk_i, clk_i_n, phase_i, error_i, _ = phase_interpolate(
        clk_0, clk_90, clk_180, clk_270, num_bits, code,
        dnl_profile=dnl, inl_profile=inl, complementary=True
    )
    # Use clk_i to sample received signal
    sample = signal[...]  # Sample with clk_i
    sample_n = signal_n[...]  # Or with clk_i_n if needed
```

## Performance Notes

### Amplitude Variation
The interpolated clock amplitude varies slightly due to linear mixing:
- At quadrature boundaries (clk_0, clk_90, etc.): Full amplitude
- At 45° points (ratio=0.5): ~0.707× amplitude (RMS)

### Phase Accuracy
- Theoretical accuracy: Limited by quadrature clock phase relationship
- Practical accuracy: ±0.1° for well-matched quadrature clocks
- Best accuracy near quadrant boundaries (ratio near 0 or 1)
- With DNL/INL: Model realistic hardware errors up to ±5°

### Complementary Clock Quality
- Theoretical: Perfect 180° phase relationship (clk_p = -clk_n)
- Practical: Maintain relationship through all DNL/INL transforms
- Verification: max(clk_p + clk_n) should be near machine epsilon

## File Location
- Module: `/workspaces/CodeSpace_SerDes/phase_interpolator.py` (v2.0 - CONSOLIDATED)
- Documentation: `/workspaces/CodeSpace_SerDes/PHASE_INTERPOLATOR_DOCS.md`
- Test output: `/workspaces/CodeSpace_SerDes/plots/phase_interpolator_demo.png`

## Running Tests

```bash
cd /workspaces/CodeSpace_SerDes
python3 phase_interpolator.py
```

Test Coverage:
- ✓ Single phase interpolation with various codes
- ✓ Full phase bank generation
- ✓ Error handling validation  
- ✓ Complementary clock verification
- ✓ DNL/INL profile generation
- ✓ Visualization and waveform plots

## Integration Checklist - v2.0

### Basic Usage
- [ ] Import `phase_interpolate` from `phase_interpolator.py`
- [ ] Generate quadrature clocks from `clock_generator.py`
- [ ] Choose `num_bits` based on required phase resolution
- [ ] Select control code based on desired phase

### Enhanced Usage with v2.0 Features
- [ ] (Optional) Create DNL profile with `create_dnl_profile()`
- [ ] (Optional) Create INL profile with `create_inl_profile()`
- [ ] (Optional) Set `complementary=True` if differential clocks needed
- [ ] Check `phase_error` for error budget analysis
- [ ] Use `clk_interp_n` if differential sampling required
