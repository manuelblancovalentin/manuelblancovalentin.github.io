---
title: "CMS28v2 – AI In-Pixel Readout Chip for HL-LHC"
collection: portfolio
category: portfolio
permalink: /portfolio/2023-01-01-cms28v2-ai-in-pixel-readout-chip
excerpt: 'Full digital logic and layout of a radiation-hard pixel ASIC in 28nm CMOS with integrated neural network classifiers'
date: 2023-01-01
technologies: [ASIC, 28nm CMOS, AI Accelerator, HLS, High-Energy Physics]
funding: 'CERN, Fermilab, Northwestern University, Columbia University'
github_url: ''
description: |
  Designed and implemented the full digital logic and layout for a radiation-hardened pixel ASIC in 28nm CMOS. Integrated neural network classifiers using hls4ml and Catapult HLS to achieve sub-10ns latency with minimal area footprint. This chip is designed for the HL-LHC detector upgrade at CERN to perform on-sensor AI-based filtering of pixel cluster data in real-time.
  
  **Key accomplishments:**
  - Full RTL-to-GDS II backend flow with timing closure at target clock frequencies
  - Sub-10ns neural network inference latency
  - Radiation-hardened design patterns for HL-LHC environment
  - Integration with detector front-end electronics
  - Production-ready design for fabrication

status: 'Completed / Taped Out'
---

## CMS28v2: In-Pixel AI Readout Chip

| Field | Value |
| --- | --- |
| Project | CMS28v2 – AI In-Pixel Readout Chip for HL-LHC |
| Year | 2023 |
| Technology | 28nm CMOS |
| Type | ASIC |
| Role | Lead Digital Designer |
| Funding | CERN, Fermilab, Northwestern University, Columbia University |
| Collaborators | Giuseppe Di Guglielmo, Farah Fahim, others |

### Project Overview

This project involved designing a complete radiation-hardened pixel ASIC for the HL-LHC CMS detector upgrade. The chip integrates on-sensor AI-based neural network classifiers to perform real-time data filtering and compression, reducing data bandwidth while maintaining physics performance.

### Technical Details

- **Neural Network Integration**: Implemented ML inference using hls4ml framework integrated with Catapult HLS for hardware synthesis
- **Latency Optimization**: Achieved sub-10ns neural network inference latency to meet detector trigger timing
- **Radiation Hardening**: Applied TMR (Triple Modular Redundancy) and other hardening techniques for HL-LHC radiation environment
- **Layout & Signoff**: Full physical design from synthesis through DRC/LVS and timing closure
- **Verification**: Comprehensive RTL and post-layout verification including radiation upset simulation

### Publications

- Parpillon et al. (2023). "Radiation-Hard Smart-Pixel Detector ASIC ReadOut with Digital AI in 28nm"

---

*GitHub repository: [Add link when available]*
