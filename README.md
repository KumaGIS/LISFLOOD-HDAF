# LISFLOOD-HDAF
LISFLOOD-HDAF integrates LISFLOOD with EnKF to enable spatially distributed streamflow data assimilation and support OSSE-based evaluation of gauge-network design and assimilation frequency.

# LISFLOOD-EnKF Hydrological Data Assimilation Framework

### Overview
This repository contains the custom Ensemble Kalman Filter (EnKF) implementation developed for the LISFLOOD distributed hydrological model. The framework supports both **state augmentation** and **non-augmented (single-variable)** assimilation modes.

The data assimilation is performed in an **offline-coupled architecture**, where the LISFLOOD model runs forward in time, pauses at assimilation times, writes model states to disk, and calls the EnKF module to update them before resuming the simulation.

---

## Key Components

| File | Description |
|------|-------------|
| `Lisflood_EnKF_StateAugment.py` | EnKF implementation with augmented state vector (Q, UZ, SM1, SM2, SM3) |
| `Lisflood_EnKF_NoStateAugment.py` | EnKF implementation using discharge only |
| `stateVar.py` | Unified script controlling variable selection (toggle-based) |
| `readmeteo.py` | Unified script controlling perturbation of forcing data (toggle-based) |
| `Source Code Updates log.docx` | Full documentation of modifications |

---

## State Vector Configuration

### **Option 1 — State Augmentation**
Includes 5 hydrologic states:
- Channel discharge (`chanq`)
- Upper groundwater storage (`UZ`)
- Soil moisture layers (`SM1`, `SM2`, `SM3`)

### **Option 2 — No State Augmentation**
Includes only:
- Channel discharge (`chanq`)

---

## How to Switch Between Modes

Add the following toggle inside `stateVar.py`:

```python
USE_STATE_AUGMENTATION = True  # Set False for discharge-only mode

if USE_STATE_AUGMENTATION:
    state_var_list = ['chanq', 'uz', 'sm1', 'sm2', 'sm3']
else:
    state_var_list = ['chanq']
```

---

## Workflow Overview

```text
Run LISFLOOD → Pause at Assimilation Time → Extract State Vector
         ↓
   EnKF: setState() → setObservations() → Analysis Update
         ↓
 Write Updated States Back → Resume LISFLOOD Simulation
```

---

## Requirements

- Python 3.7
- PCRaster
- numpy, scipy, pandas, matplotlib
- netCDF4, xarray (if using forcing data stacks)

---

## Running the System

```
Similar to the deterministic LISFLOOD model run provided on the official website
```

---

## Citation
If you use this framework, please cite the upcoming publication:

*Kurugama, K. M., Kazama, S., & Hiraga, Y. (2025). Augmenting observation network design and assimilation frequency in distributed hydrological models: Insights from the 
LISFLOOD-based Hydrological Data Assimilation Framework. Journal of Hydrology, XXX, XXX–XXX. https://doi.org/10.1016/j.jhydrol.XXXX.XX.XXX*

---

## License
MIT License – free to modify and distribute with attribution.

