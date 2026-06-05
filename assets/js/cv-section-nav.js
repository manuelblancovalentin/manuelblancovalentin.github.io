(function () {
  const links = Array.from(document.querySelectorAll('[data-cv-section-link]'));
  const sections = links
    .map((link) => document.getElementById(link.dataset.cvSectionLink))
    .filter(Boolean);

  if (!links.length || !sections.length) {
    return;
  }

  function setActiveSection(sectionId) {
    links.forEach((link) => {
      link.classList.toggle('is-active', link.dataset.cvSectionLink === sectionId);
    });
  }

  function updateActiveSection() {
    const candidates = sections
      .map((section) => ({
        section,
        top: section.getBoundingClientRect().top,
      }));

    const passed = candidates
      .filter((candidate) => candidate.top <= 150)
      .sort((first, second) => second.top - first.top);
    const nearest = passed[0] || candidates.sort((first, second) => first.top - second.top)[0];

    if (nearest) {
      setActiveSection(nearest.section.id);
    }
  }

  links.forEach((link) => {
    link.addEventListener('click', (event) => {
      const target = document.getElementById(link.dataset.cvSectionLink);
      if (!target) {
        return;
      }

      event.preventDefault();
      setActiveSection(target.id);
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      window.setTimeout(updateActiveSection, 280);
    });
  });

  updateActiveSection();
  window.addEventListener('scroll', updateActiveSection, { passive: true });
  window.addEventListener('resize', updateActiveSection);
}());
