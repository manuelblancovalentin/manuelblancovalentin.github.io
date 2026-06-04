---
title: "CITC2 - Cryogenic Mixed-Signal Readout ASIC"
collection: hardware
category: hardware
permalink: /hardware/citc2-cryogenic-mixed-signal-readout-asic
excerpt: 'Digital controller and readout for mixed-signal cryogenic ASIC with 200MHz timing'
subtitle: 'Cryogenic mixed-signal controller and SPI readout ASIC'
date: 2023-01-01
year: 2023
technology: '22nm FD-SOI'
funding: 'Northwestern University, Fermilab'
github_url: ''
status: 'Taped Out'
card_badges: [GF, 22nm]
taxonomy:
  discipline: [quantum]
  domain: [cryogenics, control]
  topic: [hardware, asic, readout, mixed-signal]
  technology: [gf22, fdsoi]
  foundry: [globalfoundries]
  process-node: [22nm]
  organization: [northwestern, fermilab]
  contribution: [rtl, asic-design, verification, tapeout]
---

## CITC2 - Cryogenic Mixed-Signal Readout ASIC

| Field | Value |
| --- | --- |
| Project | CITC2 - Cryogenic Mixed-Signal Readout ASIC |
| Year | 2023 |
| Technology | 22nm FD-SOI |
| Type | ASIC |
| Status | Taped Out |
| Funding | Northwestern University, Fermilab |

### Project Overview

Developed digital controller, configuration interface, and SPI readout for mixed-signal cryogenic ASIC in 22nm FD-SOI. Led RTL-to-GDSII backend flow achieving 200MHz timing closure under cryogenic (4K) constraints.

### Design Flow

- **RTL Design**: Verilog HDL implementation and verification
- **Synthesis**: Cadence Genus with technology-specific optimization  
- **Place & Route**: Cadence Innovus with timing and power closure
- **Physical Verification**: DRC/LVS with Calibre and extraction
- **Verification**: Post-layout simulation and formal verification

---

*GitHub repository: [Add link when available]*

For related publications, see the [Publications](/publications/) page.
