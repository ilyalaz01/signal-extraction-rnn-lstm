# Website — Signal Source Extraction

Static, single-page explainer for the parent project. No build step.

## What this is

A softer entry point to the research project for a general audience: teammates, curious students,
anyone who lands on the GitHub repo. The full research narrative lives in the parent repository's
`README.md` (698 lines); this page links there.

## Local preview

Just open `index.html` in a browser. Two options:

```bash
# Option 1: file:// URL
xdg-open "file:///mnt/c/ilya/code projects/signal-extraction-rnn-lstm/website/index.html"

# Option 2: tiny local server (recommended — fonts/CDN behave better)
cd website && python3 -m http.server 8000
# then visit http://localhost:8000
```

## Deploy to Vercel

This subdirectory is self-contained. From the repo root:

```bash
cd website
vercel deploy            # preview deployment
vercel deploy --prod     # production
```

Vercel will detect a static site, run no build step, and serve the files as-is.
`vercel.json` adds long-lived cache headers for figures and basic security headers.

## Files

| File          | Purpose                                                          |
|---------------|------------------------------------------------------------------|
| `index.html`  | The page. Semantic HTML, KaTeX from CDN.                         |
| `styles.css`  | All styles. Single file. No preprocessor.                        |
| `script.js`   | Hero canvas + interactive signal explorer + scroll reveals.      |
| `assets/*.png`| Six figures copied from `../assets/figures/`.                    |
| `vercel.json` | Static-site config: cache headers + security headers.            |
