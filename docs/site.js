const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.dataset.panel).hidden = !selected;
  }
}
for (const tab of tabs) {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
      (tabs.indexOf(tab) + direction + tabs.length) % tabs.length;
    selectTab(tabs[index]);
    tabs[index].focus();
  });
}
const dialog = document.getElementById('figure-dialog');
for (const link of document.querySelectorAll('[data-figure]')) {
  link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const image = document.getElementById('expanded-figure');
    image.src = link.href;
    image.alt = link.querySelector('img').alt;
    dialog.showModal();
  });
}
document.getElementById('close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
document.getElementById('copy-citation').addEventListener('click', async () => {
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(document.getElementById('bibtex').textContent);
    status.textContent = 'Citation copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('bibtex'));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    status.textContent = 'Citation selected for copying.';
  }
});

const comparison = document.getElementById('drawer-comparison');
if (comparison) {
  const real = document.getElementById('comparison-real');
  const sim = document.getElementById('comparison-sim');
  const slider = document.getElementById('comparison-reveal');
  const status = document.getElementById('comparison-status');
  const stage = comparison.querySelector('.comparison-stage');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let visible = false;
  let ready = false;
  let failed = false;
  let draggedPointer = null;
  let playRequest = 0;

  function reveal(value) {
    const scan = Math.max(0, Math.min(100, value));
    slider.value = String(scan);
    slider.setAttribute('aria-valuetext', `Real ${Math.round(100 - scan)}%, SIM ${Math.round(scan)}%`);
    comparison.style.setProperty('--reveal', `${scan}%`);
  }

  async function start() {
    if (!ready || reducedMotion.matches || !visible || failed) return;
    const request = ++playRequest;
    try {
      sim.currentTime = real.currentTime;
      await Promise.all([real.play(), sim.play()]);
      if (request !== playRequest) return;
      status.hidden = true;
    } catch {
      if (request !== playRequest) return;
      real.pause();
      sim.pause();
      status.textContent = 'Animation could not play.';
      status.hidden = false;
    }
  }

  function pause() {
    playRequest += 1;
    real.pause();
    sim.pause();
  }

  slider.addEventListener('input', () => reveal(Number(slider.value)));
  function drag(event) {
    const bounds = stage.getBoundingClientRect();
    reveal((event.clientX - bounds.left) / bounds.width * 100);
  }
  stage.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    draggedPointer = event.pointerId;
    stage.setPointerCapture(event.pointerId);
    drag(event);
  });
  stage.addEventListener('pointermove', event => { if (event.pointerId === draggedPointer) drag(event); });
  stage.addEventListener('pointerup', () => { draggedPointer = null; });
  stage.addEventListener('pointercancel', () => { draggedPointer = null; });

  function onReady() {
    if (ready || failed || real.readyState < 3 || sim.readyState < 3) return;
    ready = true;
    real.playbackRate = sim.playbackRate = 1;
    start();
  }
  function onVideoError() {
    failed = true;
    pause();
    status.textContent = 'Video unavailable.';
    status.hidden = false;
  }
  for (const video of [real, sim]) {
    video.addEventListener('canplay', onReady);
    video.addEventListener('error', onVideoError);
  }

  // Both files contain the same forward/reverse cycle; keep their clocks aligned.
  real.addEventListener('timeupdate', () => {
    if (!failed && !real.paused && !sim.paused && !real.seeking && !sim.seeking &&
        Math.abs(sim.currentTime - real.currentTime) > 0.06) sim.currentTime = real.currentTime;
  });

  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting && !document.hidden;
    if (visible) start(); else pause();
  }, { threshold: 0.05 }).observe(comparison);
  document.addEventListener('visibilitychange', () => {
    const bounds = comparison.getBoundingClientRect();
    visible = !document.hidden && bounds.bottom > 0 && bounds.top < window.innerHeight;
    if (visible) start(); else pause();
  });
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) pause(); else start();
  });
  pause();
  reveal(Number(slider.value));
  if (real.error || sim.error) onVideoError();
  onReady();
}
