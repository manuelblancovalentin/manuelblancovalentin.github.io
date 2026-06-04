---
title: "🤖 CryoAI – Prototyping cryogenic chips for machine learning at 22nm"
collection: talks
type: "Conference proceedings talk"
permalink: /talks/2022-10-04-fastml-cryoai
venue: "FASTML 2022"
date: 2022-10-04
location: "University Park, TX, US"
excerpt: "Talk on CryoAI, a 22nm cryogenic chip prototype for machine-learning workloads."
taxonomy:
  type: [talk]
  discipline: [computer-engineering, quantum]
  domain: [cryogenics]
  topic: [hardware, asic, ai-accelerator, machine-learning]
  technology: [gf22, fdsoi]
  foundry: [globalfoundries]
  process-node: [22nm]
  venue: [fastml]
card_image: '/images/talks/fastml_2022.png'
card_image_alt: 'CryoAI – Prototyping cryogenic chips for machine learning at 22nm.'
---

Talk given at the [FASTML 2022 Conference](https://indico.cern.ch/event/1156222/contributions/5062806/), presenting our design experience of a prototype System-on-Chip (SoC) for machine learning applications that run in a cryogenic environment to evaluate the performance of the digital backend flow. We combined two established open-source projects (ESP and HLS4ML) into a new system-level design flow to build and program the SoC. In the modular tile-based architecture, we integrated a low-power 32-bit RISC-V microcontroller (Ibex), 200KB SRAM-based scratchpad, and an 18K-parameter neural-network accelerator. The network is an autoencoder working on audio recordings and trained on industrial use cases for the early detection of failures in machines like slide rails, fans, or pumps. For the hls4ml translation, we optimized the reference architecture using quantization and model compression techniques with minimal AUC performance reduction. This project is also an early evaluation of Siemens Catapult as an HLS backend for hls4ml. Finally, we fabricated the SoC in a 22nm technology and are currently testing it.

<!-- For more information about the research, please visit [the ENABOL project page](https://manuelblancovalentin.github.io/kappa_budgeting/). -->

<embed src="/files/2022_fastml_cryoai.pdf" width="100%" height="1000px" 
 type="application/pdf">
