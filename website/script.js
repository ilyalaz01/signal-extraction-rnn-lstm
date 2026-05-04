// Signal Source Extraction — interactive bits.
// Vanilla JS, no dependencies. Respects prefers-reduced-motion.

(function () {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- Hero canvas: four drifting sines + cyan composite ----------
  function setupHeroCanvas() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = 0, h = 0;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      w = rect.width; h = rect.height;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener('resize', resize, { passive: true });

    // four "frequencies" for visual rhythm — not literal Hz
    // periods chosen so they look like 2/10/50/200Hz at relative scales
    const components = [
      { period: 1.0, amp: 0.45, phaseSpeed: 0.07, color: 'rgba(6,182,212,0.10)', lw: 1.2 },
      { period: 0.20, amp: 0.30, phaseSpeed: 0.11, color: 'rgba(245,158,11,0.10)', lw: 1.0 },
      { period: 0.045, amp: 0.18, phaseSpeed: 0.18, color: 'rgba(134,239,172,0.10)', lw: 0.8 },
      { period: 0.012, amp: 0.10, phaseSpeed: 0.25, color: 'rgba(255,255,255,0.06)', lw: 0.6 },
    ];

    function drawFrame(t) {
      ctx.clearRect(0, 0, w, h);
      const cy = h * 0.55;
      const ampPx = h * 0.18;

      // individual components
      for (const c of components) {
        ctx.strokeStyle = c.color;
        ctx.lineWidth = c.lw;
        ctx.beginPath();
        for (let x = 0; x <= w; x += 2) {
          const u = x / w;
          const y = cy + ampPx * c.amp * Math.sin((u / c.period) * Math.PI * 2 + t * c.phaseSpeed);
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // cyan composite (sum)
      ctx.strokeStyle = 'rgba(6,182,212,0.32)';
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let x = 0; x <= w; x += 2) {
        const u = x / w;
        let s = 0;
        for (const c of components) {
          s += c.amp * Math.sin((u / c.period) * Math.PI * 2 + t * c.phaseSpeed);
        }
        const y = cy + ampPx * s * 0.55;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    if (reduceMotion) {
      drawFrame(0);
      return;
    }
    let last = performance.now();
    function loop(now) {
      const t = now / 1000;
      drawFrame(t);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  // ---------- Signal explorer widget ----------
  function setupExplorer() {
    const canvas = document.getElementById('explorer-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = 0, h = 0;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      w = rect.width; h = rect.height;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    // visualization params: 0..0.2 seconds at fs=1000Hz = 200 samples wide
    const N = 800;
    const T = 0.2; // seconds
    const freqs = [2, 10, 50, 200];
    // legend palette: 2 Hz purple, 10 Hz amber, 50 Hz green, 200 Hz red
    const colors = {
      2:   'rgba(167,139,250,0.65)',
      10:  'rgba(245,158,11,0.65)',
      50:  'rgba(134,239,172,0.65)',
      200: 'rgba(251,113,133,0.65)',
    };

    const initialSlider = Number((document.getElementById('beta-slider') || { value: 13 }).value);
    const state = {
      checks: { 2: true, 10: true, 50: true, 200: true },
      beta: (initialSlider / 100) * 2 * Math.PI,
    };

    // deterministic pseudo-random for stable visualization across re-renders
    function mulberry32(seed) {
      return function () {
        let t = (seed += 0x6D2B79F5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }
    function gauss(rng) {
      const u = 1 - rng(), v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    function draw() {
      ctx.clearRect(0, 0, w, h);

      // axis line
      const cy = h / 2;
      ctx.strokeStyle = 'rgba(148,163,184,0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, cy); ctx.lineTo(w, cy);
      ctx.stroke();

      // amplitude scale: max possible sum is 4, so amp / 4.5
      const yScale = (h * 0.42) / 4.5;
      const beta = state.beta;
      const rng = mulberry32(42);

      // precompute per-sample phase noise (one draw per i, shared across freqs to make beta visibly degrade)
      const phaseNoise = new Float32Array(N);
      for (let i = 0; i < N; i++) phaseNoise[i] = gauss(rng) * beta;

      const ampNoise = new Float32Array(N * 4);
      for (let i = 0; i < N * 4; i++) ampNoise[i] = gauss(rng) * 0.05;

      // draw individual sines, each in its own assigned color
      ctx.lineWidth = 1.5;
      for (let fi = 0; fi < freqs.length; fi++) {
        const f = freqs[fi];
        if (!state.checks[f]) continue;
        ctx.strokeStyle = colors[f];
        ctx.beginPath();
        for (let i = 0; i < N; i++) {
          const t = (i / N) * T;
          const v = Math.sin(2 * Math.PI * f * t);
          const x = (i / N) * w;
          const y = cy - v * yScale;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // composite (cyan, drawn on top with thicker line)
      ctx.strokeStyle = 'rgba(6,182,212,0.95)';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const t = (i / N) * T;
        let s = 0;
        for (let fi = 0; fi < freqs.length; fi++) {
          const f = freqs[fi];
          if (!state.checks[f]) continue;
          const a = 1 + ampNoise[i * 4 + fi];
          const phi = phaseNoise[i] * 0.5;
          s += a * Math.sin(2 * Math.PI * f * t + phi);
        }
        const x = (i / N) * w;
        const y = cy - s * yScale;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // wire up controls
    document.querySelectorAll('.freq-toggles .chip').forEach((chip) => {
      const f = Number(chip.dataset.freq);
      const cb = chip.querySelector('input');
      cb.addEventListener('change', () => {
        state.checks[f] = cb.checked;
        draw();
      });
    });

    const slider = document.getElementById('beta-slider');
    const readout = document.getElementById('beta-readout');
    const foot = document.getElementById('explorer-readout');
    const marks = document.querySelectorAll('.slider-marks span');

    function betaFromSlider(v) {
      // 0..100 → 0..2π, with the marks at: 0, 12.5(π/8), 25(π/4), 50(π/2), 75(π wait), 100(2π) — let's set: 0, 6.25, 12.5, 25, 50, 100
      const t = v / 100;
      return t * 2 * Math.PI;
    }
    function labelForBeta(b) {
      if (b < 0.01) return '0';
      const ratio = b / Math.PI;
      if (Math.abs(ratio - 0.125) < 0.04) return 'π/8';
      if (Math.abs(ratio - 0.25) < 0.04) return 'π/4';
      if (Math.abs(ratio - 0.5) < 0.04) return 'π/2';
      if (Math.abs(ratio - 1.0) < 0.04) return 'π';
      if (Math.abs(ratio - 2.0) < 0.04) return '2π';
      return ratio.toFixed(2) + 'π';
    }
    function commentForBeta(b) {
      // Bucket boundaries match the experiment narrative; they fire on the
      // actual slider value (no off-by-one in the labelling).
      if (b < Math.PI / 8)  return 'β < π/8 — very mild phase noise; task is easy.';
      if (b < Math.PI / 4)  return 'π/8 ≤ β < π/4 — small phase perturbation.';
      if (b < Math.PI / 2)  return 'π/4 ≤ β < π/2 — the operating point we chose for our experiments. Task is hard but solvable.';
      if (b < Math.PI)      return 'π/2 ≤ β < π — past the knee; difficulty jumps.';
      return 'β ≥ π — near the noise floor; no architecture can recover.';
    }
    function updateMarks(b) {
      const ratio = b / Math.PI;
      const targets = [0, 0.125, 0.25, 0.5, 1.0, 2.0];
      let active = 0; let bestDist = Infinity;
      targets.forEach((t, i) => {
        const d = Math.abs(t - ratio);
        if (d < bestDist) { bestDist = d; active = i; }
      });
      marks.forEach((m, i) => m.classList.toggle('mark-active', i === active));
    }

    function syncFromSlider() {
      state.beta = betaFromSlider(Number(slider.value));
      readout.textContent = labelForBeta(state.beta);
      foot.textContent = commentForBeta(state.beta);
      updateMarks(state.beta);
      draw();
    }
    if (slider) {
      slider.addEventListener('input', syncFromSlider);
      syncFromSlider();
    }

    window.addEventListener('resize', resize, { passive: true });
    resize();
  }

  // ---------- Reveal on scroll ----------
  function setupReveals() {
    if (reduceMotion) return;
    const targets = document.querySelectorAll('.section, .t-item, .card, .stat, .explore-card');
    targets.forEach((el) => el.classList.add('reveal'));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -60px 0px' });
    targets.forEach((el) => io.observe(el));
  }

  // ---------- Init ----------
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { setupHeroCanvas(); setupExplorer(); setupReveals(); });
  } else {
    setupHeroCanvas();
    setupExplorer();
    setupReveals();
  }
})();
