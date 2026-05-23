---
layout: cv
title: "CV"
permalink: /cv/
author_profile: true
last_modified: 2026-05-23
skills:
  - ASIC
  - VLSI
  - RAS
  - FPGA
  - ML accelerators
  - HLS
  - Digital Design
  - Verification
  - PyTorch
  - SystemVerilog
  - Cadence EDA
redirect_from:
  - /resume
---

{% include base_path %}

{% include cv-header.html name="Manuel" surname="Blanco Valentin" title="PhD Computer Engineering — AI specialist" %}

{% capture profile_content %}
Passionate about designing efficient, fault-tolerant ML hardware systems and neuromorphic computing architectures. Expertise in radiation-hardened ASICs, high-level synthesis, and training-aware hardware accelerators. Enthusiastic about bridging the gap between machine learning and physical implementation challenges.
{% endcapture %}

{% include tbox.html type="profile" title="Profile" content=profile_content %}

{% include cv-skills.html skills=page.skills %}



Education
======
* <img class="cv-logo" src="{{ base_path }}/images/logos/nu.png" alt="Northwestern" /> **Ph.D. in Computer Engineering**, Northwestern University, Evanston, IL (2020–Present)
  * GPA: 3.9/4.0
  * Focus: Hardware-software co-designed ML systems, neuromorphic learning, radiation-hardened ASICs
  * Thesis: In progress

* <img class="cv-logo" src="{{ base_path }}/images/logos/nu.png" alt="Northwestern" /> **M.S. in Computer Engineering**, Northwestern University, Evanston, IL (2020–2022)
  * GPA: 3.9/4.0
  * Completed graduate coursework in ASIC design, ML accelerators, and digital systems

* <img class="cv-logo" src="{{ base_path }}/images/logos/cbpf.png" alt="CBPF" /> **M.S. in Physics & Scientific Instrumentation**, CBPF (Brazilian Center for Physics Research), Rio de Janeiro, Brazil (2015–2018)
  * GPA: 3.8/4.0
  * Thesis: [Deep learning methods on geological reservoir borehole log images and applications](https://www.researchgate.net/publication/336587891_Deep_learning_methods_on_geological_reservoir_borehole_log_images_and_applications)

* <img class="cv-logo" src="{{ base_path }}/images/logos/UPCtech.png" alt="UPC" /> **B.S. in Robotics & Electronics Engineering**, UPC BarcelonaTech (Polytechnical University of Barcelona), Barcelona, Spain (2010–2014)
  * GPA: 3.3/4.0
  * Thesis: [Design and implementation of the autonomous navigation control system of a model tugboat](https://hdl.handle.net/2099.1/24115) -in Spanish

Work experience
======

* <img class="cv-logo" src="{{ base_path }}/images/logos/amd.jpeg" alt="AMD" /> **Systems Engineer Junior – RAS** (June 2026–Sep 2026)
  * AMD, Austin, TX / USA
  * Developing techniques to evaluate resilience of high-performance processors under delay/fault scenarios
  * Building ML/data-analysis pipelines to extract trends from reliability datasets
  * Collaborating with system and hardware architects on mitigation strategies

* <img class="cv-logo" src="{{ base_path }}/images/logos/nu.png" alt="Northwestern" /> <img class="cv-logo" src="{{ base_path }}/images/logos/fermilab.jpg" alt="Fermilab" /> **ASIC + ML Accelerator Researcher** (2020–Present)
  * Northwestern University / Fermilab / CERN, Evanston, IL
  * Developing NRCST K: a neuromorphic learning system with local learning rules and metabolic optimization
  * Led multiple ASIC efforts across GF 22nm FD-SOI, TSMC 28nm, and TSMC 65nm processes (RTL/HLS→P&R→signoff)
  * Built reusable automation for synthesis, P&R, extraction, and signoff (Cadence Virtuoso/Innovus/Genus/Tempus/Voltus)
  * Implemented radiation-hardened logic with TMR; developed RTL tools for redundancy injection
  * Extended ENABOL: training-aware HLS templates and backward-pass wrappers for training-capable hardware

* <img class="cv-logo" src="{{ base_path }}/images/logos/Cadence.jpg" alt="Cadence" /> **Tempus SSV Graduate Intern – Timing/EDA** (Mar 2023–Sep 2023)
  * Cadence Design Systems, San Jose, CA / USA
  * Researched silicon aging/drift impacts in 5nm CMOS and their effects on timing closure
  * Developed and integrated drift estimation modules into Tempus SSV
  * Created tooling for drift-aware library characterization and algorithm improvements
  * Explored ML-assisted prediction techniques for drift-aware analysis

* <img class="cv-logo" src="{{ base_path }}/images/logos/petrobras.png" alt="Petrobras" /> **Deep Learning Specialist** (Sep 2015–Jun 2020)
  * Petrobras, Rio de Janeiro, Brazil
  * Built and deployed CNN/autoencoder/Bayesian deep learning models for industrial imaging and reservoir characterization
  * Delivered end-to-end pipelines: data ingestion, training/evaluation, uncertainty quantification, production reporting
  * Developed AI systems for oil & gas reservoir characterization using seismic and borehole image data
  * Applied Bayesian deep learning for uncertainty quantification and model robustness

* <img class="cv-logo" src="{{ base_path }}/images/logos/cern.png" alt="CERN" /> **ASIC Design & Verification Lecturer (Invited Instructor)** (Aug 2021–Sep 2021)
  * CERN INFIERI School, Universidad Autónoma de Madrid, Madrid, Spain
  * Delivered 5-day lecture/lab series on ASIC design & verification (synthesis, P&R, DRC/LVS, signoff) to international cohort of ~20+ students
  * Developed academic labs and automated flows covering full design methodology
  
Skills
======

**Programming**
* Python, C/C++, Verilog/VHDL, Tcl/Tk, Bash, Cadence SKILL, SPICE

**ML & AI**
* PyTorch, TensorFlow/Keras, NumPy/Pandas, quantization-aware workflows

**ASIC / EDA**
* Cadence: Virtuoso, Innovus, Genus, Tempus, Voltus, Quantus, Calibre DRC/LVS
* Mentor: Catapult HLS
* Signoff automation and verification

**HLS / FPGA**
* Catapult HLS, Vivado/Vitis, RTL/HLS verification & testbench generation

**Operating Systems**
* Linux (expert), macOS, Windows

**Languages**
* Spanish (native), Catalan (native), Portuguese, English (fluent), French (basic)

Publications
======

{% assign sorted_pubs = site.publications | sort: "sort_key" %}
<ul class="cv-publications">
{% for post in sorted_pubs %}
  <li>
    <a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a>
    — <i>{{ post.venue }}</i> ({{ post.date | default: "1900-01-01" | date: "%Y" }})
  </li>
{% endfor %}
</ul>

Patents
======

{% assign sorted_patents = site.patents | sort: "date" | reverse %}
<ul>
{% for post in sorted_patents %}
  <li>
    <a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a>
    — <i>{{ post.venue }}</i> ({{ post.date | default: "1900-01-01" | date: "%Y" }})
  </li>
{% endfor %}
</ul>

Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
Teaching
======

{% assign sorted_teaching = site.teaching | sort: "date" | reverse %}
<ul>
{% for post in sorted_teaching %}
  <li>
    <a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a>
    — <i>{{ post.venue }}</i> ({{ post.date | default: "1900-01-01" | date: "%Y" }})
  </li>
{% endfor %}
</ul>
  
