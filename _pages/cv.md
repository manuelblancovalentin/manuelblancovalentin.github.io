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

<div class="cv-page-shell">
{% include cv-section-nav.html %}
<div class="cv-page-content">

<section id="cv-profile" class="cv-scroll-section">

{% capture profile_content %}
Passionate about designing efficient, fault-tolerant ML hardware systems and neuromorphic computing architectures. Expertise in radiation-hardened ASICs, high-level synthesis, and training-aware hardware accelerators. Enthusiastic about bridging the gap between machine learning and physical implementation challenges.
{% endcapture %}

{% include tbox.html type="profile" title="Profile" content=profile_content %}

{% include cv-skills.html skills=page.skills %}

</section>

<section id="cv-education" class="cv-scroll-section">
{% include cv-section-heading.html title="Education" accent="#4e2a84" %}
{% capture phd_education_description %}
Hardware-software co-designed ML systems, neuromorphic learning, and radiation-hardened ASICs.

Thesis: In progress.
{% endcapture %}
{% assign nu_logo = base_path | append: "/images/logos/nu.png" %}
{% include cv-education-card.html
  logo=nu_logo
  logo_url="https://www.northwestern.edu"
  logo_alt="Northwestern University"
  institution="Northwestern University"
  department="Computer Engineering"
  accent="#4e2a84"
  title="Ph.D. in Computer Engineering"
  location="Evanston, IL"
  location_url="https://maps.app.goo.gl/Je4fPbAb5Hf2CuBA7"
  timeline="2020 - Present"
  gpa="3.9/4.0"
  description=phd_education_description
%}

{% capture ms_nu_education_description %}
Completed graduate coursework in ASIC design, ML accelerators, and digital systems.
{% endcapture %}
{% include cv-education-card.html
  logo=nu_logo
  logo_url="https://www.northwestern.edu"
  logo_alt="Northwestern University"
  institution="Northwestern University"
  department="Computer Engineering"
  accent="#4e2a84"
  title="M.S. in Computer Engineering"
  location="Evanston, IL"
  location_url="https://maps.app.goo.gl/Je4fPbAb5Hf2CuBA7"
  timeline="2020 - 2022"
  gpa="3.9/4.0"
  description=ms_nu_education_description
%}

{% capture cbpf_education_description %}
Thesis: [Deep learning methods on geological reservoir borehole log images and applications](https://www.researchgate.net/publication/336587891_Deep_learning_methods_on_geological_reservoir_borehole_log_images_and_applications).
{% endcapture %}
{% assign cbpf_logo = base_path | append: "/images/logos/cbpf.png" %}
{% include cv-education-card.html
  logo=cbpf_logo
  logo_url="https://cbpf.br"
  logo_alt="CBPF"
  institution="CBPF"
  department="Brazilian Center for Physics Research"
  accent="#2e8b57"
  title="M.S. in Physics & Scientific Instrumentation"
  location="Rio de Janeiro, Brazil"
  location_url="https://maps.app.goo.gl/DwAU89qUh75rskXK8"
  timeline="2015 - 2018"
  gpa="3.8/4.0"
  description=cbpf_education_description
%}

{% assign spain_flag = base_path | append: "/images/flags/spain.svg" %}
{% capture upc_education_description %}
Thesis: [Design and implementation of the autonomous navigation control system of a model tugboat](https://hdl.handle.net/2099.1/24115) <img class="cv-flag" src="{{ spain_flag }}" alt="Spanish language" title="Spanish">
{% endcapture %}
{% assign upc_logo = base_path | append: "/images/logos/UPCtech.png" %}
{% include cv-education-card.html
  logo=upc_logo
  logo_url="https://www.upc.edu/en"
  logo_alt="UPC BarcelonaTech"
  institution="UPC BarcelonaTech"
  department="Polytechnical University of Barcelona"
  accent="#0072bc"
  title="B.S. in Robotics & Electronics Engineering"
  location="Barcelona, Spain"
  location_url="https://www.google.com/maps/search/?api=1&query=Barcelona%2C%20Spain"
  timeline="2010 - 2014"
  gpa="3.3/4.0"
  description=upc_education_description
%}

</section>

<section id="cv-work-experience" class="cv-scroll-section">
{% include cv-section-heading.html title="Work experience" accent="#8a5a44" %}

{% assign amd_logo = base_path | append: "/images/logos/amd.jpeg" %}
{% capture amd_work_description %}
* Developing techniques to evaluate resilience of high-performance processors under delay/fault scenarios.
* Building ML/data-analysis pipelines to extract trends from reliability datasets.
* Collaborating with system and hardware architects on mitigation strategies.
{% endcapture %}
{% include cv-work-card.html
  logo=amd_logo
  logo_url="https://www.amd.com/en.html"
  logo_alt="AMD"
  company="AMD"
  group="Reliability, Availability, and Serviceability"
  accent="#111111"
  title="Systems Engineer Junior - RAS"
  location="Austin, TX, USA"
  location_url="https://maps.app.goo.gl/e4dMd5eMxYj9XBJW7"
  timeline="June 2026 - Sep 2026"
  description=amd_work_description
%}

{% assign fnal_logo = base_path | append: "/images/logos/fermilab.jpg" %}
{% assign cern_logo = base_path | append: "/images/logos/cern.png" %}
{% assign research_logos = nu_logo | append: "|" | append: fnal_logo | append: "|" | append: cern_logo %}
{% assign research_logo_urls = "https://www.northwestern.edu|https://www.fnal.gov|https://home.cern" %}
{% capture research_work_description %}
* Developing NRCST K: a neuromorphic learning system with local learning rules and metabolic optimization.
* Led multiple ASIC efforts across GF 22nm FD-SOI, TSMC 28nm, and TSMC 65nm processes (RTL/HLS to P&R to signoff).
* Built reusable automation for synthesis, P&R, extraction, and signoff (Cadence Virtuoso/Innovus/Genus/Tempus/Voltus).
* Implemented radiation-hardened logic with TMR; developed RTL tools for redundancy injection.
* Extended ENABOL: training-aware HLS templates and backward-pass wrappers for training-capable hardware.
{% endcapture %}
{% include cv-work-card.html
  logos=research_logos
  logo_urls=research_logo_urls
  logo_alts="Northwestern University|Fermilab|CERN"
  company="Northwestern University / Fermilab / CERN"
  group="ASIC + ML Accelerator Research"
  accent="#4e2a84"
  title="ASIC + ML Accelerator Researcher"
  location="Evanston, IL, USA"
  location_url="https://maps.app.goo.gl/Je4fPbAb5Hf2CuBA7"
  timeline="2020 - Present"
  description=research_work_description
%}

{% assign cadence_logo = base_path | append: "/images/logos/Cadence.jpg" %}
{% capture cadence_work_description %}
* Researched silicon aging/drift impacts in 5nm CMOS and their effects on timing closure.
* Developed and integrated drift estimation modules into Tempus SSV.
* Created tooling for drift-aware library characterization and algorithm improvements.
* Explored ML-assisted prediction techniques for drift-aware analysis.
{% endcapture %}
{% include cv-work-card.html
  logo=cadence_logo
  logo_url="https://www.cadence.com/en_US/home.html"
  logo_alt="Cadence"
  company="Cadence Design Systems"
  group="Timing / EDA"
  accent="#a6202d"
  title="Tempus SSV Graduate Intern"
  location="San Jose, CA, USA"
  location_url="https://maps.app.goo.gl/HtuYfiJuLLPEGVyE8"
  timeline="Mar 2023 - Sep 2023"
  description=cadence_work_description
%}

{% assign petrobras_logo = base_path | append: "/images/logos/petrobras.png" %}
{% capture petrobras_work_description %}
* Built and deployed CNN/autoencoder/Bayesian deep learning models for industrial imaging and reservoir characterization.
* Delivered end-to-end pipelines: data ingestion, training/evaluation, uncertainty quantification, production reporting.
* Developed AI systems for oil & gas reservoir characterization using seismic and borehole image data.
* Applied Bayesian deep learning for uncertainty quantification and model robustness.
{% endcapture %}
{% include cv-work-card.html
  logo=petrobras_logo
  logo_url="https://petrobras.com.br/en/"
  logo_alt="Petrobras"
  company="Petrobras"
  group="Deep Learning and Reservoir Characterization"
  accent="#008542"
  title="Deep Learning Specialist"
  location="Rio de Janeiro, Brazil"
  location_url="https://maps.app.goo.gl/DwAU89qUh75rskXK8"
  timeline="Sep 2015 - Jun 2020"
  description=petrobras_work_description
%}

{% capture cern_work_description %}
* Delivered 5-day lecture/lab series on ASIC design & verification (synthesis, P&R, DRC/LVS, signoff) to international cohort of ~20+ students.
* Developed academic labs and automated flows covering full design methodology.
{% endcapture %}
{% include cv-work-card.html
  logo=cern_logo
  logo_url="https://home.cern"
  logo_alt="CERN"
  company="CERN INFIERI School"
  group="Universidad Autonoma de Madrid"
  accent="#0053a1"
  title="ASIC Design & Verification Lecturer (Invited Instructor)"
  location="Madrid, Spain"
  location_url="https://maps.app.goo.gl/Joxur2y2kMfVo2du7"
  timeline="Aug 2021 - Sep 2021"
  description=cern_work_description
%}

</section>

<section id="cv-skills" class="cv-scroll-section">
{% include cv-section-heading.html title="Skills" accent="#3776ab" %}

<div class="cv-skill-card-grid">
  {% for category in site.data.cv_skills.skill_categories %}
    {% include cv-skill-card.html category=category %}
  {% endfor %}
</div>

</section>

<section id="cv-languages" class="cv-scroll-section">
{% include cv-section-heading.html title="Languages" accent="#2e8b57" %}

<div class="cv-skill-card-grid cv-skill-card-grid--languages">
  {% for category in site.data.cv_skills.language_categories %}
    {% include cv-skill-card.html category=category %}
  {% endfor %}
</div>

</section>

<section id="cv-publications" class="cv-scroll-section">
{% include cv-section-heading.html title="Publications" accent="#7b61ff" %}

{% assign sorted_pubs = site.publications | sort: "sort_key" %}
{% assign sorted_hardware = site.hardware | sort: "date" | reverse %}
{% assign sorted_talks = site.talks | sort: "date" | reverse %}
<div class="cv-publication-topic-grid">
{% for topic in site.data.cv_publication_topics.topics %}
  {% assign topic_count = 0 %}
  {% capture topic_items %}
    {% for publication_id in topic.publications %}
      {% assign publication_id_clean = publication_id | strip %}
      {% for post in sorted_pubs %}
        {% assign post_path = post.permalink | default: post.url %}
        {% assign post_id = post_path | remove: "/publication/" | remove: ".html" | remove: "/" %}
        {% if post_id == publication_id_clean %}
          {% assign topic_count = topic_count | plus: 1 %}
          <li class="cv-publication-topic-card__item">
            <a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a>
            <span class="cv-publication-topic-card__meta">
              {{ post.venue }} · {{ post.date | default: "1900-01-01" | date: "%Y" }}
            </span>
          </li>
          {% break %}
        {% endif %}
      {% endfor %}
    {% endfor %}
    {% for hardware_id in topic.hardware %}
      {% assign hardware_id_clean = hardware_id | strip %}
      {% for project in sorted_hardware %}
        {% assign project_path = project.permalink | default: project.url %}
        {% assign project_id = project_path | remove: "/hardware/" | remove: ".html" | remove: "/" %}
        {% if project_id == hardware_id_clean %}
          {% assign topic_count = topic_count | plus: 1 %}
          <li class="cv-publication-topic-card__item">
            <a href="{{ base_path }}{{ project.url }}">{{ project.title }}</a>
            <span class="cv-publication-topic-card__meta">
              ASIC · {{ project.date | default: "1900-01-01" | date: "%Y" }}
            </span>
          </li>
          {% break %}
        {% endif %}
      {% endfor %}
    {% endfor %}
    {% for talk_id in topic.talks %}
      {% assign talk_id_clean = talk_id | strip %}
      {% for talk in sorted_talks %}
        {% assign talk_path = talk.permalink | default: talk.url %}
        {% assign talk_id_current = talk_path | remove: "/talks/" | remove: "/publication/" | remove: ".html" | remove: "/" %}
        {% if talk_id_current == talk_id_clean %}
          {% assign topic_count = topic_count | plus: 1 %}
          <li class="cv-publication-topic-card__item">
            <a href="{{ base_path }}{{ talk.url }}">{{ talk.title }}</a>
            <span class="cv-publication-topic-card__meta">
              {{ talk.venue | default: "Talk" }} · {{ talk.date | default: "1900-01-01" | date: "%Y" }}
            </span>
          </li>
          {% break %}
        {% endif %}
      {% endfor %}
    {% endfor %}
  {% endcapture %}
  {% if topic_count > 0 %}
    <article class="cv-publication-topic-card" style="--cv-publication-topic-accent: {{ topic.accent | default: '#7b61ff' }};">
      <div class="cv-publication-topic-card__header">
        <h3>
          {% if topic.icon %}
            {% include lucide-icon.html name=topic.icon class="cv-publication-topic-card__icon" size=19 %}
          {% endif %}
          <span>{{ topic.title }}</span>
        </h3>
        <span>{{ topic_count }}</span>
      </div>
      <ul class="cv-publication-topic-card__list">
        {{ topic_items }}
      </ul>
    </article>
  {% endif %}
{% endfor %}
</div>

</section>

<section id="cv-peer-review" class="cv-scroll-section">
{% include cv-section-heading.html title="Peer review" accent="#0f766e" %}

<article class="cv-peer-review-card" style="--cv-peer-review-accent: #0f766e;">
  <div class="cv-peer-review-card__main">
    <div class="cv-peer-review-card__header">
      {% include lucide-icon.html name="clipboard-check" class="cv-peer-review-card__icon" size=22 %}
      <div>
        <p class="cv-peer-review-card__eyebrow">Verified peer reviews</p>
        <h3>IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems</h3>
      </div>
    </div>
    <a class="cv-peer-review-card__link" href="https://www.webofscience.com/wos/author/record/ODJ-9773-2025" target="_blank" rel="noopener noreferrer">
      Verified through Web of Science / Clarivate
      {% include lucide-icon.html name="arrow-up-right" class="cv-peer-review-card__external-icon" size=14 %}
    </a>
  </div>
  <div class="cv-peer-review-card__count" aria-label="3 verified reviews">
    <span>3</span>
    <small>reviews</small>
  </div>
</article>

</section>

<section id="cv-patents" class="cv-scroll-section">
{% include cv-section-heading.html title="Patents" accent="#b8860b" %}

{% assign sorted_patents = site.patents | sort: "date" | reverse %}
<div class="cv-compact-card-grid">
{% for post in sorted_patents %}
  <article class="cv-compact-card cv-compact-card--patent" style="--cv-compact-card-accent: #b8860b;">
    <div class="cv-compact-card__header">
      {% include lucide-icon.html name="scroll-text" class="cv-compact-card__icon" size=21 %}
      <div class="cv-compact-card__title-block">
        <p class="cv-compact-card__eyebrow">Patent</p>
        <h3><a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a></h3>
      </div>
      {% if post.patent_number or post.publication_number %}
        <span class="cv-compact-card__badge">{{ post.patent_number | default: post.publication_number }}</span>
      {% endif %}
    </div>
    <p class="cv-compact-card__meta">
      {{ post.venue | default: post.patent_office }}
      {% if post.date %} · {{ post.date | date: "%Y" }}{% endif %}
    </p>
    {% if post.excerpt %}
      <p class="cv-compact-card__description">{{ post.excerpt }}</p>
    {% endif %}
    <div class="cv-compact-card__footer">
      {% if post.status %}
        <span class="cv-compact-card__pill">{{ post.status }}</span>
      {% endif %}
      <a class="cv-compact-card__link" href="{{ base_path }}{{ post.url }}">
        Patent details {% include lucide-icon.html name="arrow-up-right" class="cv-compact-card__external-icon" size=14 %}
      </a>
      {% if post.patent_url %}
        <a class="cv-compact-card__link" href="{{ post.patent_url }}" target="_blank" rel="noopener noreferrer">
          Patent record {% include lucide-icon.html name="arrow-up-right" class="cv-compact-card__external-icon" size=14 %}
        </a>
      {% endif %}
    </div>
  </article>
{% endfor %}
</div>

</section>

<section id="cv-talks" class="cv-scroll-section">
{% include cv-section-heading.html title="Talks" accent="#0072ce" %}
{% assign sorted_talk_cards = site.talks | sort: "date" | reverse %}
<div class="cv-compact-card-grid">
{% for post in sorted_talk_cards %}
  {% assign talk_kind = "Talk" %}
  {% assign talk_details_label = "Talk details" %}
  {% assign talk_icon = "mic" %}
  {% if post.taxonomy.type contains "poster" or post.type contains "Poster" or post.type contains "poster" %}
    {% assign talk_kind = "Poster" %}
    {% assign talk_details_label = "Poster details" %}
    {% assign talk_icon = "presentation" %}
  {% elsif post.type contains "slides" or post.type contains "Slides" or post.type contains "presentation" %}
    {% assign talk_icon = "presentation" %}
  {% endif %}
  <article class="cv-compact-card cv-compact-card--talk" style="--cv-compact-card-accent: #0072ce;">
    <div class="cv-compact-card__header">
      {% include lucide-icon.html name=talk_icon class="cv-compact-card__icon" size=21 %}
      <div class="cv-compact-card__title-block">
        <p class="cv-compact-card__eyebrow">{{ talk_kind }}</p>
        <h3><a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a></h3>
      </div>
      {% if post.date %}
        <span class="cv-compact-card__badge">{{ post.date | date: "%Y" }}</span>
      {% endif %}
    </div>
    <p class="cv-compact-card__meta">
      {{ post.venue | default: post.location }}
      {% if post.date %} · {{ post.date | date: "%B %-d, %Y" }}{% endif %}
    </p>
    {% if post.excerpt %}
      <p class="cv-compact-card__description">{{ post.excerpt }}</p>
    {% endif %}
    <div class="cv-compact-card__footer">
      {% if post.location %}
        <span class="cv-compact-card__pill">{{ post.location }}</span>
      {% endif %}
      <a class="cv-compact-card__link" href="{{ base_path }}{{ post.url }}">
        {{ talk_details_label }} {% include lucide-icon.html name="arrow-up-right" class="cv-compact-card__external-icon" size=14 %}
      </a>
      {% if post.paperurl %}
        <a class="cv-compact-card__link" href="{{ post.paperurl }}" target="_blank" rel="noopener noreferrer">
          Related record {% include lucide-icon.html name="arrow-up-right" class="cv-compact-card__external-icon" size=14 %}
        </a>
      {% endif %}
    </div>
  </article>
{% endfor %}
</div>

</section>

<section id="cv-teaching" class="cv-scroll-section">
{% include cv-section-heading.html title="Teaching" accent="#555" %}

{% assign sorted_teaching = site.teaching | sort: "date" | reverse %}
<div class="cv-compact-card-grid">
{% for post in sorted_teaching %}
  <article class="cv-compact-card cv-compact-card--teaching" style="--cv-compact-card-accent: #555;">
    <div class="cv-compact-card__header">
      {% include lucide-icon.html name="graduation-cap" class="cv-compact-card__icon" size=21 %}
      <div class="cv-compact-card__title-block">
        <p class="cv-compact-card__eyebrow">{{ post.type | default: "Teaching" }}</p>
        <h3><a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a></h3>
      </div>
      {% if post.date %}
        <span class="cv-compact-card__badge">{{ post.date | date: "%Y" }}</span>
      {% endif %}
    </div>
    <p class="cv-compact-card__meta">
      {{ post.venue | default: post.location }}
      {% if post.date %} · {{ post.date | date: "%B %-d, %Y" }}{% endif %}
    </p>
    {% if post.excerpt %}
      <p class="cv-compact-card__description">{{ post.excerpt }}</p>
    {% endif %}
    <div class="cv-compact-card__footer">
      {% if post.location %}
        <span class="cv-compact-card__pill">{{ post.location }}</span>
      {% endif %}
      <a class="cv-compact-card__link" href="{{ base_path }}{{ post.url }}">
        Teaching details {% include lucide-icon.html name="arrow-up-right" class="cv-compact-card__external-icon" size=14 %}
      </a>
    </div>
  </article>
{% endfor %}
</div>

</section>
</div>
</div>

<script src="{{ base_path }}/assets/js/cv-section-nav.js" defer></script>
  
