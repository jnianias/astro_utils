# S09 Stacking Comparison: Old vs New Implementation

## Summary of Differences

This document compares the stacking procedures between the **original S09 notebook** and the **revised S09 notebook** using `stack_spectra_across_sources()`.

---

## ✓ Verified Identical: Core Stacking Algorithm

The comparison test confirms that `stack_spectra_across_sources()` produces **numerically identical** results to the old manual stacking method when using the same inputs:
- Same weighting scheme: inverse median variance (`1/σ²`)
- Same velocity frame: systemic (shifted by `DELTAV_LYA`)
- Same interpolation to common velocity grid
- Same error propagation
- **Max difference: 0.0e+00** ✓

---

## ⚠️ Key Differences in Source Selection & Parameters

### 1. **NEW: Additional High-SNR Lyman Alpha Cut**

**OLD notebook:**
```python
# Only checks for systemic velocity
eml = row['DELTAV_LYA'] > -np.inf
if not eml:
    continue
```

**NEW notebook:**
```python
# Added requirement for high SNR red peak
has_sysvel = megatab['DELTAV_LYA'] > -np.inf
good_sysvel_err = megatab['DELTAV_LYA_ERR'] <= 50
high_snr = megatab['SNRR'] > 30.0  # ← NEW CUT

source_mask = has_sysvel & good_sysvel_err & high_snr
```

**Impact:** This additional `SNRR > 30.0` cut will **exclude sources** that have systemic velocity measurements but weak red Lyman alpha peaks. This could significantly reduce the sample size.

---

### 2. **Low-Ionization Line Selection**

**OLD notebook (adaptive):**
```python
# Choose lines based on detection
if liabs:
    linestoavg_lo = row['DV_LI_ABS_LINES'].split('+')  # Use detected lines
else:
    linestoavg_lo = ['SiII1260', 'SiII1527']  # Use default set
```

**NEW notebook (fixed):**
```python
# Always use the same two lines
li_abslines = ['SiII1260', 'CII1334']
```

**Key differences:**
- **OLD:** Adapts line selection per source based on `DV_LI_ABS_LINES` column
- **OLD:** Falls back to `['SiII1260', 'SiII1527']` if no absorption detected
- **NEW:** Always uses `['SiII1260', 'CII1334']` for all sources
- **Line change:** `SiII1527` → `CII1334`

**Impact:** 
- Different lines averaged for sources with detected absorption
- For sources without detected absorption, different fallback lines
- Could affect the spectral shape and S/N of the stacked result

---

### 3. **Single-Peaked Mask Definition**

**OLD notebook:**
```python
if ~(row['SNRB'] > 3.0) and row['Z'] < 4.0:
    # Add to SINGLE list
```

**NEW notebook:**
```python
single_peaked_mask = source_mask & ~(megatab['SNRB'] > 3.0) & (megatab['Z'] < 4.0)
```

**Note:** The logic is the same (`~(SNRB > 3.0)` handles both `SNRB <= 3.0` AND `SNRB = NaN`), but:
- OLD: Evaluated after loading spectrum
- NEW: Pre-evaluated mask before stacking

**Impact:** None (logically equivalent)

---

### 4. **Velocity Grid Bounds**

**OLD notebook:**
```python
newvel = np.arange(-2460, 2520, velstep)
```

**NEW notebook:**
```python
velbounds = [-2500, 2500]
common_vel = np.arange(velbounds[0], velbounds[1] + velstep, velstep)
```

**Impact:** Slightly different velocity grid:
- OLD: -2460 to +2520 km/s (69 bins)
- NEW: -2500 to +2500 km/s (68 bins)
- Difference is minor and at edges of velocity range

---

### 5. **Blacklist Not Applied**

**OLD notebook:**
```python
blacklist_abs = [
    '1592 from cluster A370',
    '1278888 from cluster SMACS2031', 
    '106 from cluster SMACS2131',
    '99 from cluster SMACS2131',
    '00074 from cluster RXJ1347',
    '67 from cluster SMACS2031',
    '2318888 from cluster MACS0940',
    '000748888 from cluster SMACS2031',
    '0001010 from cluster RXJ1347'
]
```
*Note: This blacklist is defined but **never actually used** in the old code!*

**NEW notebook:**
```python
# No blacklist defined
```

**Impact:** None (the old blacklist wasn't applied anyway)

---

### 6. **Absorption Parameter**

**OLD notebook:**
```python
avg_lines(row, linestoavg_lo, absorption=False, ...)
```

**NEW notebook:**
```python
auspec.stack_spectra_across_sources(..., absorption=False, ...)
```

**Impact:** Both use `absorption=False`, so weighting is by inverse variance within each source (not by SNR). This is correct and identical.

---

## 🔍 Most Critical Differences

### **Priority 1: High-SNR Cut (SNRR > 30.0)**
- **Biggest impact** on sample size
- Not present in old notebook
- Need to verify if this was intentionally added or is a mistake

### **Priority 2: Low-Ion Line Selection**
- OLD uses adaptive line selection per source
- NEW uses fixed `['SiII1260', 'CII1334']`
- Could affect spectral properties of the stack

### **Priority 3: Line Choice for Non-Detections**
- OLD fallback: `['SiII1260', 'SiII1527']`
- NEW always: `['SiII1260', 'CII1334']`

---

## 📊 Recommended Actions

1. **Remove or justify the `SNRR > 30.0` cut**
   - Was this intentional?
   - How many sources does it exclude?
   
2. **Decide on line selection strategy:**
   - Option A: Use adaptive selection (restore old logic)
   - Option B: Keep fixed lines (document why `CII1334` instead of `SiII1527`)
   
3. **Match velocity grid exactly** (if needed):
   - Use `[-2460, 2520]` to match old notebook
   - Or document why `[-2500, 2500]` is preferred

4. **Run quantitative comparison:**
   - Count sources in old vs new with all cuts
   - Compare final stacked spectra visually
   - Check if results are scientifically equivalent

---

## ✅ What's Confirmed Identical

- ✓ Core stacking algorithm (weighted averaging)
- ✓ Weighting scheme (inverse median variance)
- ✓ Velocity frame (systemic with DELTAV_LYA shift)
- ✓ Error propagation method
- ✓ Velocity step (75 km/s)
- ✓ Single vs double-peaked separation logic
- ✓ DELTAV_LYA_ERR <= 50 cut
- ✓ Z < 4.0 cut for single-peaked
