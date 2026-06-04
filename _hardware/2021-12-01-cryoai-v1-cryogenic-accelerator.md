---
title: "CryoAI v1"
collection: hardware
category: hardware
permalink: /hardware/cryoai-v1-cryogenic-accelerator
excerpt: 'Ultra-low-power ML accelerator with custom cryogenic memories and RISC-V'
subtitle: 'Cryogenic ML Accelerator'
date: 2021-12-01
year: 2021
technology: 'GF 22nm FD-SOI'
funding: 'Fermilab, Northwestern University, Columbia University'
github_url: ''
status: 'Completed'
card_badges: [GF, 22nm, AI accelerator]
taxonomy:
  discipline: [quantum]
  domain: [cryogenics]
  topic: [hardware, asic, ai-accelerator]
  technology: [gf22, fdsoi]
  foundry: [globalfoundries]
  process-node: [22nm]
  organization: [fermilab, northwestern, columbia]
  contribution: [architecture, rtl, asic-design, verification, tapeout]
card_image: '/images/hardware/cryoai/pnr.png'
card_image_alt: 'ECON-I 2021 V1 chip mounted on a printed circuit board'
gallery_images:
  - image: '/images/hardware/cryoai/pnr.png'
    alt: '...'
    caption: '...'
  - image: '/images/hardware/cryoai/actual_chip.png'
    alt: '...'
    caption: '...'
  - image: '/images/hardware/cryoai/cryostat.png'
    alt: '...'
    caption: '...'
  - image: '/images/hardware/cryoai/diagram.png'
    alt: '...'
    caption: '...'
---

## CryoAI v1 - Cryogenic ML Accelerator

| Field | Value |
| --- | --- |
| Project | CryoAI v1 - Cryogenic ML Accelerator |
| Year | 2021 |
| Technology | GF 22nm FD-SOI |
| Type | ASIC |
| Status | Completed |
| Funding | Fermilab, Northwestern University, Columbia University |

### Project Overview

On-edge ML accelerator optimized for cryogenic (4K) operation in GF 22nm. Features custom 22nm memory compilation, e-MRAM integration, and RISC-V processor.

### Design Flow

- **RTL Design**: Verilog HDL implementation and verification
- **Synthesis**: Cadence Genus with technology-specific optimization  
- **Place & Route**: Cadence Innovus with timing and power closure
- **Physical Verification**: DRC/LVS with Calibre and extraction
- **Verification**: Post-layout simulation and formal verification

---

*GitHub repository: [Add link when available]*

For related publications, see the [Publications](/publications/) page.
