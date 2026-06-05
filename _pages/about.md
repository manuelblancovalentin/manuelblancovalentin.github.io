---
permalink: /
title: "About me"
author_profile: true
hide_title: true
redirect_from:
  - /about/
  - /about.html
---

<div class="landing-page">
  <section class="landing-hero" aria-labelledby="landing-hero-title">
    <p class="landing-eyebrow">Computer engineering · ML hardware · scientific instrumentation</p>
    <h1 id="landing-hero-title">
      Computer engineer working across
      <span class="landing-highlight">machine-learning hardware</span>,
      <span class="landing-highlight">ASIC/FPGA design</span>,
      scientific instrumentation, and adaptive on-chip learning.
    </h1>
    <p class="landing-hero__summary">
      I build hardware/software systems for machine learning under physical constraints:
      <span class="landing-inline-keyword">latency</span>,
      <span class="landing-inline-keyword">power</span>,
      <span class="landing-inline-keyword">numerical precision</span>,
      <span class="landing-inline-keyword">memory movement</span>,
      <span class="landing-inline-keyword">timing closure</span>,
      <span class="landing-inline-keyword">radiation</span>,
      and deployment on real devices. My work spans algorithms, software tooling, HLS/RTL
      implementation, verification, FPGA/ASIC workflows, and documentation for reproducible
      scientific use.
    </p>
  </section>

  <section class="landing-section" aria-labelledby="technical-focus-title">
    <div class="landing-section__header">
      <p class="landing-section__eyebrow">Technical focus</p>
      <h2 id="technical-focus-title">Areas I work across</h2>
    </div>
    <div class="landing-chip-cloud" aria-label="Technical focus areas">
      <span class="landing-chip">ML accelerators</span>
      <span class="landing-chip">ASIC design</span>
      <span class="landing-chip">FPGA/HLS</span>
      <span class="landing-chip">RTL implementation</span>
      <span class="landing-chip">hardware-aware ML</span>
      <span class="landing-chip">scientific instrumentation</span>
      <span class="landing-chip">radiation-tolerant systems</span>
      <span class="landing-chip">edge AI</span>
      <span class="landing-chip">neuromorphic learning</span>
      <span class="landing-chip">local plasticity</span>
      <span class="landing-chip">on-chip training</span>
      <span class="landing-chip">CPU reliability</span>
      <span class="landing-chip">timing closure</span>
      <span class="landing-chip">EDA</span>
      <span class="landing-chip">fixed-point arithmetic</span>
      <span class="landing-chip">verification</span>
      <span class="landing-chip">documentation</span>
    </div>
  </section>

  <section class="landing-section" aria-labelledby="what-i-build-title">
    <div class="landing-section__header">
      <p class="landing-section__eyebrow">What I build</p>
      <h2 id="what-i-build-title">Projects at the algorithm/hardware boundary</h2>
    </div>
    <div class="landing-card-grid landing-card-grid--build">
      <article class="landing-card">
        <h3>Hardware-aware ML systems</h3>
        <p>Tools and workflows that translate machine-learning models into efficient FPGA/ASIC implementations.</p>
        <ul class="landing-card__keywords">
          <li>hls4ml</li>
          <li>hls4ml-trainable</li>
          <li>ENABOL</li>
          <li>quantization</li>
          <li>fixed-point deployment</li>
        </ul>
      </article>

      <article class="landing-card">
        <h3>Scientific instrumentation hardware</h3>
        <p>Low-latency and reliable digital systems for experimental physics and detector readout.</p>
        <ul class="landing-card__keywords">
          <li>Fermilab</li>
          <li>CERN</li>
          <li>particle detectors</li>
          <li>radiation tolerance</li>
          <li>ASIC workflows</li>
        </ul>
      </article>

      <article class="landing-card">
        <h3>Adaptive and neuromorphic learning</h3>
        <p>Learning systems based on local adaptation, energy constraints, and on-device plasticity.</p>
        <ul class="landing-card__keywords">
          <li>NRCSTK</li>
          <li>local learning</li>
          <li>metabolic constraints</li>
          <li>edge adaptation</li>
          <li>non-backprop learning</li>
        </ul>
      </article>

      <article class="landing-card">
        <h3>Computer architecture and reliability</h3>
        <p>Reliability-aware architecture work across timing, memory systems, and resilient execution.</p>
        <ul class="landing-card__keywords">
          <li>AMD RAS</li>
          <li>CPU pipelines</li>
          <li>memory hierarchy</li>
          <li>ECC</li>
          <li>aging</li>
          <li>fault tolerance</li>
        </ul>
      </article>

      <article class="landing-card landing-card--wide">
        <h3>Full-stack research engineering</h3>
        <p>Bridging algorithms, software, hardware implementation, experiments, and documentation.</p>
        <ul class="landing-card__keywords">
          <li>Python</li>
          <li>C++</li>
          <li>HLS</li>
          <li>RTL</li>
          <li>verification</li>
          <li>reproducible workflows</li>
        </ul>
      </article>
    </div>
  </section>

  <section class="landing-section landing-narrative" aria-labelledby="trajectory-summary-title">
    <div class="landing-section__header">
      <p class="landing-section__eyebrow">TL;DR</p>
      <h2 id="trajectory-summary-title">How I got here</h2>
    </div>
    <div class="landing-narrative__body">
      <p>
        My path started in robotics engineering in Barcelona, where I learned to think across
        physical systems, electronics, control, and software. In Brazil, I moved into applied
        physics and industrial AI, working on deep-learning methods for imaging and reservoir
        characterization. During my Ph.D. at Northwestern, that background evolved into
        hardware-oriented machine learning: low-latency inference, ASIC and FPGA implementation,
        particle-detector instrumentation, and adaptive learning systems that can operate under
        tight physical constraints.
      </p>
      <p>
        Across these stages, the recurring theme has been the same: I like systems where algorithms
        have to survive contact with physics. That means numerical precision, latency, power, timing
        closure, radiation, memory movement, and the engineering details that decide whether an idea
        can actually run.
      </p>
    </div>
  </section>

  <section class="landing-section" aria-labelledby="trajectory-title">
    <div class="landing-section__header">
      <p class="landing-section__eyebrow">Trajectory</p>
      <h2 id="trajectory-title">Where the stack came from</h2>
    </div>
    {% include landing-trajectory.html %}
  </section>

  <section class="landing-section" aria-labelledby="evidence-graph-title">
    <div class="landing-section__header">
      <p class="landing-section__eyebrow">Evidence graph</p>
      <h2 id="evidence-graph-title">Skills tied to projects and environments</h2>
    </div>
    {% include landing-skill-graph.html %}
  </section>

  <section class="landing-section" aria-labelledby="contributions-title">
    <div class="landing-section__header">
      <p class="landing-section__eyebrow">Contributions</p>
      <h2 id="contributions-title">Concrete work behind the profile</h2>
    </div>
    <div class="landing-card-grid landing-card-grid--contributions">
      <article class="landing-contribution-card">
        <h3><a href="/software/hls4ml">hls4ml original author and contributor</a></h3>
        <p>Helped build the open-source workflow that translates machine-learning models into FPGA/ASIC implementations for low-latency scientific applications.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/software/hls4ml-trainable">hls4ml-trainable creator</a></h3>
        <p>Developing trainable extensions and hardware-aware learning workflows for hls4ml.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/software/enabol">ENABOL creator</a></h3>
        <p>Building adaptive training controllers and experimentation tools for on-chip learning and edge-AI systems.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/talks/2026-04-16-cosyne">NRCSTK neuromorphic research</a></h3>
        <p>Research on local, energy-constrained learning systems and adaptive neural dynamics for on-device plasticity.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/hardware/2023-01-01-cms28v2-ai-in-pixel-readout-chip">Radiation-hard in-pixel AI ASICs</a></h3>
        <p>Digital logic and layout for pixel readout chips with neural-network classifiers, radiation-hard design patterns, and sub-10 ns inference targets.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/hardware/econai-cms-v1-calorimeter">Reconfigurable neural-network ASICs</a></h3>
        <p>Architecture, RTL, verification, radiation-hardening, and tapeout work for TSMC 65 nm detector-data compression accelerators.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/hardware/cryoai-v1-cryogenic-accelerator">Cryogenic ML and readout ASICs</a></h3>
        <p>GF 22 nm FD-SOI cryogenic accelerators, quantum-control/readout systems, and mixed-signal digital control for low-temperature operation.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/software/wolf">Wolf EDA automation</a></h3>
        <p>Tooling for digital implementation flows, including synthesis, place-and-route, extraction, and signoff workflows.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/publications/">Scientific detector AI publications</a></h3>
        <p>Published work on in-pixel AI, smart pixels, cryogenic readout electronics, detector compression, and hls4ml-based acceleration.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/publications/">Industrial imaging and geoscience AI</a></h3>
        <p>Deep-learning methods for borehole image analysis, reservoir characterization, fracture/breakout detection, and uncertainty estimation.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/patents/">Patent contributions</a></h3>
        <p>Patent applications for ultrasonic image artifact removal, automatic breakout detection in reservoir well images, and hardware-measurement-based analysis methods.</p>
      </article>
      <article class="landing-contribution-card">
        <h3><a href="/software/">Personal research tooling</a></h3>
        <p>Reusable Python tools including Nodus for job/workflow orchestration and Pergamos for dynamic HTML reporting.</p>
      </article>
    </div>
  </section>
</div>
