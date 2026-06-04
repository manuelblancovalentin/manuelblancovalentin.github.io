---
title: "ECONAI-CMS v1"
collection: hardware
category: hardware
permalink: /hardware/econai-cms-v1-calorimeter
excerpt: 'Reprogrammable AI accelerator for CMS calorimeter with radiation hardening'
subtitle: 'CMS Calorimeter AI Accelerator'
date: 2021-09-01
year: 2021
technology: 'TSMC 65nm'
funding: 'CERN, Fermilab, Northwestern U, Columbia U, MIT, Brown U, Caltech, FIT, IPB'
github_url: ''
status: 'Deployed'
card_badges: [TSMC, 65nm, Radhard, TMR, AI accelerator, HEP]
taxonomy:
  discipline: [hep]
  topic: [hardware, asic, ai-accelerator, data-compression, radhard, tmr]
  technology: [tsmc65]
  foundry: [tsmc]
  process-node: [65nm]
  organization: [cern, fermilab, northwestern, columbia]
  contribution: [architecture, rtl, asic-design, verification, tapeout]
description: |
  On-chip reprogrammable AI accelerator for the CERN CMS calorimeter, implemented in TSMC 65nm. The design handles extreme detector data rates and includes automatic triple modular redundancy injection for radiation hardening.
card_image: '/images/hardware/econ/chip.png'
card_image_alt: 'ECON-I 2021 V1 chip mounted on a printed circuit board'
gallery_images:
  - image: '/images/hardware/econ/chip.png'
    alt: 'ECON-I 2021 V1 chip mounted on a printed circuit board'
    caption: 'ECON-I 2021 V1 chip mounted on a printed circuit board.'
  - image: '/images/hardware/econ/diagram.png'
    alt: 'Block diagram of the ECONAI-CMS converter, I2C configuration block, and encoder neural network'
    caption: 'Architecture overview: converter, distributed I2C configuration block, and encoder neural network.'
  - image: '/images/hardware/econ/layout.png'
    alt: 'Annotated physical layout of the ECONAI-CMS chip'
    caption: 'Annotated physical layout showing the converter, encoder neural network, and distributed I2C weights.'
  - image: '/images/hardware/econ/regtmr.png'
    alt: 'Register-level triple modular redundancy diagram'
    caption: 'Register-level triple modular redundancy with a majority voter.'
  - image: '/images/hardware/econ/fulltmr.png'
    alt: 'Full triple modular redundancy diagram'
    caption: 'Full triple modular redundancy with replicated combinational logic and majority voters.'
---

## ECONAI-CMS v1 - CMS Calorimeter AI Accelerator

| Field | Value |
| --- | --- |
| Project | ECONAI-CMS v1 - CMS Calorimeter AI Accelerator |
| Year | 2021 |
| Technology | TSMC 65nm |
| Type | ASIC |
| Status | Deployed |
| Funding | CERN, Fermilab, Northwestern U, Columbia U, MIT, Brown U, Caltech, FIT, IPB |

### Project Overview

On-chip reprogrammable AI accelerator for CERN CMS calorimeter in TSMC 65nm. Handles extreme data rates with radiation hardening via automatic TMR injection.

### Design Flow

- **RTL Design**: Verilog HDL implementation and verification
- **Synthesis**: Cadence Genus with technology-specific optimization  
- **Place & Route**: Cadence Innovus with timing and power closure
- **Physical Verification**: DRC/LVS with Calibre and extraction
- **Verification**: Post-layout simulation and formal verification

---

*GitHub repository: [Add link when available]*

For related publications, see the [Publications](/publications/) page.
