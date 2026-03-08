# Spanish Summary (Resumen para el equipo)
Este proyecto busca automatizar el "nowcasting" de nubosidad en el Valle de Aburrá.
Nuestra hipótesis es que la colorimetría del cielo predice la transparencia atmosférica.
---

# Project: UdeA Atmospheric Nowcasting (Applied Astronomy Class)

## Goal
Automated nowcasting of cloud cover for the Aburrá Valley to assess sky quality for astronomical observations.

## Hypothesis
Ground-based sky colorimetry (using the $B / (R+G)$ index) correlates with satellite-derived cloud-top temperature/transparency data.

## Workflow (WBS)
1. **Data Acquisition:** Fetch GLOBE API metadata and match with NOAA-20/GOES-19 satellite timestamps.
2. **Segmentation:** Use `Segment Anything` (SAM) on M3 Max to mask terrain and isolate the "sky window" for each observation.
3. **Metric Calculation:** Compute the Blue-to-Green+Red index on masked sky pixels.
4. **Validation:** Perform correlation analysis between calculated indices and satellite radiance.
5. **Report Generation:** Compile findings into `aastex631` format.

## Key Constraints
- All code must be modular (`.py` scripts).
- Use local M3 Max compute (MPS) for heavy image processing.
- All research outputs must adhere to AASTeX631 standards.