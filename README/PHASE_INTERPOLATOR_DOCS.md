# Phase Interpolator Module

## Overview

The phase interpolator is a clock phase generation circuit that creates finely-controlled phase-shifted clocks from 4 quadrature clock inputs (0°, 90°, 180°, 270°). It uses linear interpolation between adjacent quadrature clocks to generate intermediate phases with high resolution.

## Key Features

- **4-Phase Input**: Takes 4 quadrature clocks as input (0°, 90°, 180°, 270°)
- **Configurable Resolution**: `num_bits` parameter controls phase resolution
- **Full 360° Coverage**: Generates phases across all 4 quadrants (360° range)
- **Linear Interpolation**: Adjacent clocks mixed with ratio determined by control code
- **Error Handling**: Validates inputs and provides clear error messages
- **Flexible Output**: Returns phase shift amount and mixing ratio for debugging

## Function API

### `phase_interpolate(clk_0, clk_90, clk_180, clk_270, num_bits, code)`

Generates a single interpolated clock signal based on the control code.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `clk_0` | ndarray | Clock signal at 0° phase |
| `clk_90` | ndarray | Clock signal at 90° phase |
| `clk_180` | ndarray | Clock signal at 180° phase |
| `clk_270` | ndarray | Clock signal at 270° phase |
| `num_bits` | int | Phase resolution (total phases = 4 × 2^num_bits) |
| `code` | int | Control code [0, 4×2^num_bits - 1] |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `clk_interp` | ndarray | Interpolated clock signal |
| `phase_degrees` | float | Output phase in degrees (0° to 360°) |
| `mixing_ratio` | float | Mixing ratio (0 = 100% clk_a, 1 = 100% clk_b) |

#### Raises

- `ValueError`: If code is out of valid range
- `ValueError`: If clock arrays have different lengths
- `ValueError`: If num_bits is not a positive integer

### `generate_interpolated_bank(clk_0, clk_90, clk_180, clk_270, num_bits)`

Generates all possible interpolated clock phases for a given resolution.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `clk_0` | ndarray | Clock signal at 0° phase |
| `clk_90` | ndarray | Clock signal at 90° phase |
| `clk_180` | ndarray | Clock signal at 180° phase |
| `clk_270` | ndarray | Clock signal at 270° phase |
| `num_bits` | int | Phase resolution |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `clk_bank` | 2D ndarray | Shape (4×2^num_bits, len(clk_0)) - all interpolated clocks |
| `phases` | 1D ndarray | Output phases in degrees for each index |
| `codes` | 1D ndarray | Code values [0, 4×2^num_bits - 1] |

## Architecture

### Phase Mapping

The interpolator divides the 360° range into 4 quadrants, each mixing a different pair of adjacent quadrature clocks:

```
Quadrant 0: code ∈ [0, 2^num_bits - 1]
  Mix: clk_0 + clk_90
  Phase range: 0° → 90°

Quadrant 1: code ∈ [2^num_bits, 2×2^num_bits - 1]
  Mix: clk_90 + clk_180
  Phase range: 90° → 180°

Quadrant 2: code ∈ [2×2^num_bits, 3×2^num_bits - 1]
  Mix: clk_180 + clk_270
  Phase range: 180° → 270°

Quadrant 3: code ∈ [3×2^num_bits, 4×2^num_bits - 1]
  Mix: clk_270 + clk_0
  Phase range: 270° → 360°
```

### Mixing Formula

For a given code, the interpolated clock is calculated as:

$$clk_{interp} = (1 - r) \cdot clk_a + r \cdot clk_b$$

where:
- $r$ = mixing ratio = (local_code) / 2^num_bits
- $clk_a$, $clk_b$ = adjacent quadrature clocks for the quadrant
- $r \in [0, 1)$ determines the interpolation between the two clocks

### Phase Calculation

The output phase is:

$$\phi_{out} = \phi_{start} + r \cdot 90°$$

where $\phi_{start}$ is the starting phase of the quadrant (0°, 90°, 180°, or 270°).

## Usage Examples

### Example 1: Basic Phase Interpolation

```python
from phase_interpolator import phase_interpolate
from clock_generator import generate_clock_signal

# Generate quadrature clocks
t, clk_0, clk_90, clk_180, clk_270, f, pn, ui = generate_clock_signal(
    clock_freq_hz=10e9,
    duration_ui=100,
    samples_per_ui=256
)

# Create clock at 45° (middle of first quadrant)
num_bits = 8  # 256 phases per quadrant
code = 128    # Half of 256 = 45° (0° + 0.5 × 90°)

clk_interp, phase, ratio = phase_interpolate(
    clk_0, clk_90, clk_180, clk_270, num_bits, code
)

print(f"Output phase: {phase:.1f}°")
print(f"Mixing ratio: {ratio:.3f}")
```

### Example 2: Generate Full Phase Bank

```python
from phase_interpolator import generate_interpolated_bank

# Generate all 256 interpolated phases
num_bits = 6
clk_bank, phases, codes = generate_interpolated_bank(
    clk_0, clk_90, clk_180, clk_270, num_bits
)

print(f"Generated {len(clk_bank)} clock phases")
print(f"Phase spacing: {phases[1] - phases[0]:.3f}°")
```

### Example 3: Sweep Through All Phases

```python
# Generate clocks with different phases
num_bits = 8
num_phases = 4 * (2 ** num_bits)

for code in range(0, num_phases, 10):  # Every 10th phase
    clk_interp, phase, ratio = phase_interpolate(
        clk_0, clk_90, clk_180, clk_270, num_bits, code
    )
    # Use clk_interp for downstream processing
```

## Performance Characteristics

### Phase Resolution

The achievable phase resolution depends on `num_bits`:

| num_bits | Phases/Quadrant | Total Phases | Phase Spacing |
|----------|-----------------|--------------|---------------|
| 1 | 2 | 8 | 45.0° |
| 2 | 4 | 16 | 22.5° |
| 3 | 8 | 32 | 11.25° |
| 4 | 16 | 64 | 5.625° |
| 5 | 32 | 128 | 2.8125° |
| 6 | 64 | 256 | 1.4063° |
| 7 | 128 | 512 | 0.7031° |
| 8 | 256 | 1024 | 0.3516° |

### Phase Accuracy

The phase accuracy of the interpolated output depends on:
1. **Quadrature Clock Quality**: Phase relationship between input clocks must be exactly 90°
2. **Linear Interpolation**: Assumes clocks are sinusoidal; works well for small phase ranges
3. **Amplitude Matching**: Input clocks should have equal amplitudes for accurate mixing

### Implementation Notes

- Linear interpolation works best for small phase steps (< 10°)
- For coarse phase steps (> 45°), non-linear interpolation may improve accuracy
- Recommended: Use num_bits ≥ 4 for practical CDR/clock applications

## Verification Tests

The module includes comprehensive tests:

1. **Single Phase Interpolation**: Verify codes map correctly to phases
2. **Full Phase Bank**: Confirm uniform phase spacing across all 360°
3. **Error Handling**: Validate out-of-range codes and invalid parameters
4. **Visualization**: Plot interpolated clocks and phase spacing uniformity

Run tests with:
```bash
python3 phase_interpolator.py
```

Output files created:
- `plots/phase_interpolator_demo.png`: Waveforms and interpolated clocks
- `plots/phase_spacing_uniformity.png`: Phase spacing verification

## Integration with Clock Generator

The phase interpolator works seamlessly with `clock_generator.py`:

```python
from clock_generator import generate_clock_signal
from phase_interpolator import phase_interpolate

# Generate high-quality quadrature clocks
t, clk_0, clk_90, clk_180, clk_270, f, pn, ui = generate_clock_signal(
    clock_freq_hz=10e9,
    duration_ui=1000,
    samples_per_ui=256,
    rj_rms_ui=0.005,
    dj_freq_hz=100e6,
    dj_peak_ui=0.01
)

# Create fine-resolution interpolated clocks
clk_bank, phases, codes = generate_interpolated_bank(
    clk_0, clk_90, clk_180, clk_270, num_bits=8
)

# Use clk_bank[code] for any phase
for code in range(len(clk_bank)):
    clk_i = clk_bank[code]
    # Perform receiver sampling with clk_i
```

## Design Considerations

### Advantages
- ✅ Smooth phase sweep from 0° to 360°
- ✅ Simple linear mixing formula
- ✅ No lookup tables required
- ✅ Scalable resolution via num_bits
- ✅ Works with any clock frequency

### Limitations
- ⚠️ Linear interpolation introduces phase error for large steps
- ⚠️ Requires precisely matched quadrature clock amplitudes
- ⚠️ Phase accuracy limited by quadrature clock phase relationship
- ⚠️ High jitter in input clocks propagates to output

### Typical Use Cases
1. **Clock Data Recovery (CDR)**: Phase adjustment in PLL feedback
2. **Receiver Sampler Timing**: Fine control of sampling instant
3. **Multi-phase Clock Generation**: Create arbitrary phases from 4-phase source
4. **Phase Sweep Testing**: Characterize receiver performance vs. phase

## Mathematical Background

### Linear Interpolation Theory

For two sinusoidal signals with 90° phase difference:
- clk_a(t) = sin(ωt)
- clk_b(t) = sin(ωt + π/2) = cos(ωt)

Linear mixing produces:
- clk_interp(t) = (1-r)·sin(ωt) + r·cos(ωt)
                = √[1 + 2r(1-r)] · sin(ωt + arctan(r/(1-r)))

The amplitude varies slightly (√[1 + 2r(1-r)]) as r changes, but is maximized at r=0.5 (when both clocks contribute equally).

For small phase steps, this error is negligible.

## Future Enhancements

Potential improvements:
1. **Higher-Order Interpolation**: Lagrange or Hermite interpolation for better phase accuracy
2. **Amplitude Correction**: Normalize output to maintain constant amplitude
3. **Non-Linear Mapping**: Apply correction to quadrature clock pair selection
4. **Jitter Filtering**: Optional LPF on output for jitter reduction
5. **Multi-Phase Bank**: Memory-optimized storage for large num_bits values

## References

- Phase interpolation theory: J. Buckwalter et al., "A Monolithic 6-GHz CMOS Integrated Everything CMOS PLL"
- Clock generation: IEEE 1149.1 Standard Test Access Port and Boundary-Scan Architecture
