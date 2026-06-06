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
    const nav = document.querySelector('.cv-section-nav');
    const activationOffset = nav
      ? nav.getBoundingClientRect().bottom + 48
      : 150;
    const candidates = sections
      .map((section) => ({
        section,
        top: section.getBoundingClientRect().top,
      }));

    const passed = candidates
      .filter((candidate) => candidate.top <= activationOffset)
      .sort((first, second) => second.top - first.top);
    const nearest = passed[0] || candidates.sort((first, second) => first.top - second.top)[0];

    if (nearest) {
      setActiveSection(nearest.section.id);
    }
  }

  function getScrollOffset() {
    const masthead = document.querySelector('.masthead');
    const nav = document.querySelector('.cv-section-nav');
    const mastheadBottom = masthead
      ? masthead.getBoundingClientRect().bottom
      : 0;

    if (!nav) {
      return mastheadBottom + 16;
    }

    const navStyle = window.getComputedStyle(nav);
    const navRect = nav.getBoundingClientRect();

    if (navStyle.position === 'fixed') {
      return mastheadBottom + 16;
    }

    return navRect.bottom + 36;
  }

  function scrollToSection(target) {
    const visibleTarget = target.querySelector('.cv-section-heading') || target;
    const targetTop = visibleTarget.getBoundingClientRect().top + window.pageYOffset;
    const top = Math.max(0, targetTop - getScrollOffset());

    window.scrollTo({
      top,
      behavior: 'smooth',
    });
  }

  links.forEach((link) => {
    link.addEventListener('click', (event) => {
      const target = document.getElementById(link.dataset.cvSectionLink);
      if (!target) {
        return;
      }

      event.preventDefault();
      setActiveSection(target.id);
      scrollToSection(target);
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', `#${target.id}`);
      }
      window.setTimeout(updateActiveSection, 280);
      window.setTimeout(updateActiveSection, 620);
    });
  });

  updateActiveSection();
  window.addEventListener('scroll', updateActiveSection, { passive: true });
  window.addEventListener('resize', updateActiveSection);
}());
