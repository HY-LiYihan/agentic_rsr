# GitHub Pages

The project page is a static site: `docs/index.html`, `docs/site.css`,
`docs/site.js` and the paper media in `docs/assets/`. It has no build dependencies
or inference services. `docs/.nojekyll` disables Jekyll processing.

Publish the separate website snapshot using Settings → Pages → Deploy from a
branch, selecting `main` and `/`. This does not restore the removed test CI
workflow. GitHub runs its managed Pages deployment job when publishing changes.

The canonical website is <https://agentic-rsr.github.io/>, hosted separately in
`Agentic-RSR/agentic-rsr.github.io`. The current publication snapshot contains
the static site and its display media; Pages serves the repository `main` branch
root.

## Public Repository Page Policy

The organization site above is the only maintained project page. The public
source repository `HY-LiYihan/agentic_rsr` is a source snapshot and no longer
maintains its own GitHub Pages deployment. Its Pages setting is disabled; the
former project-page URL is not a supported entry point. Future public source
snapshots retain the repository's two-commit history and do not add page-only
publication commits.

On 2026-10-15, website commit `6d436cb1588ea00f8bda7078f0380bba2fcf1351`
deployed successfully (run `38615682138`). Live checks passed at 1440x1000,
768x1024 and 390x844, including image loading, result tabs, figure enlargement,
citation copying and no-JavaScript links. Both comparison animations loaded and
played with matching 13.146467-second loops; the divider starts at 50%.

## Private Development Preview

Development defaults to `HY-LiYihan/agentic_rsr_private` (`origin/master`).
Only an explicit request to summarize and publish changes publicly authorizes
updating `HY-LiYihan/agentic_rsr` (`public`). See [AGENTS.md](../AGENTS.md).

On 2026-10-14, enabling Pages on the private repository from `master:/docs`
returned HTTP 422: `Your current plan does not support GitHub Pages for this
repository.` Private Pages is not enabled. A supported GitHub plan is required;
after it is available, select `master` and `/docs` in the private repository's
Pages settings. The expected URL is
`https://hy-liyihan.github.io/agentic_rsr_private/`, but it is not a deployed
preview. Pages from a private personal repository is normally publicly
accessible; repository privacy alone does not restrict the website's audience.

If that private Pages URL is enabled later, `docs/index.html` redirects hosted
requests to the canonical organization site at <https://agentic-rsr.github.io/>;
local `file://` previews remain local.

Use `docs/index.html` locally for development. New edits remain local/private
until explicitly published as a website snapshot.

## Previous Public Verification

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
point to `HY-LiYihan/agentic_rsr` on its `master` branch.

Before publication, review authors, affiliations, paper PDF, reported results,
media attribution and current reproduction limits. The paper link and BibTeX
now point to [arXiv:2610.10479](https://arxiv.org/abs/2610.10479), verified
against the official arXiv API on 2026-10-14. The numeric result blocks have
been removed at the author's request; paper visualizations remain available.
