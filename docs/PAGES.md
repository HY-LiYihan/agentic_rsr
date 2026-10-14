# GitHub Pages

The project page is a static site: `docs/index.html`, `docs/site.css`,
`docs/site.js` and the paper media in `docs/assets/`. It has no build dependencies
or inference services. `docs/.nojekyll` disables Jekyll processing.

Publish using GitHub repository Settings → Pages → Deploy from a branch,
selecting `master` and `/docs`. This does not restore the removed test CI workflow.
GitHub still runs its own Pages deployment job when publishing changes.

Published URL: <https://hy-liyihan.github.io/agentic_rsr/>.

On 2026-10-14, the Pages API reported `built` for source commit
`6ae6cafc22126be45fbd82e95c254bb6a2dde159`; deployment run `38436840592`
completed successfully. The published site was checked with Playwright at
1440x1000, 768x1024 and 390x844: affiliation marks and figures loaded, no page
overflow was detected, result tabs and keyboard navigation, enlargement/Escape,
citation copying and the no-JavaScript fallback passed. Asset links responded to
HEAD requests. The live PDF was not downloaded in full for a checksum comparison;
the checked-in PDF matches the supplied original with SHA256
`0f60edefef3f67bbe02bb3c8435e755d695ca70277bd3b21858757a25edeaabe`.

Follow-up on 2026-10-14: the Pages deployment for `08cd909` completed
successfully (run `38445293903`). The public PDF was downloaded in full and
its SHA256 matched the checked-in PDF and the author's supplied `main.pdf`:
`0f60edefef3f67bbe02bb3c8435e755d695ca70277bd3b21858757a25edeaabe`.
The published Linux-validation and FR3-provenance JSON files also matched their
checked-in SHA256 values. These checks complete the earlier PDF transfer gap;
they do not change the paper's reported results or the code's inference limits.

For local inspection, open `docs/index.html` directly or serve the repository:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8765/docs/>. Image enlargement, result tabs and the
citation control are progressive enhancements; the figure links and citation text
remain available without JavaScript. The public source and documentation links
point to this repository's `master` branch.

Before publication, review authors, affiliations, paper PDF, reported results,
media attribution and current reproduction limits. The paper link and BibTeX
now point to [arXiv:2610.10479](https://arxiv.org/abs/2610.10479), verified
against the official arXiv API on 2026-10-14. The numeric result blocks have
been removed at the author's request; paper visualizations remain available.
