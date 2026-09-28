# Release checklist

This folder contains the 13 Python scripts and their direct local companion modules. Complete the article-specific checks below before publishing a tagged release.

- [x] Include `iscat_darkfield_mie_comparison_gui.py`, `nv_odmr_gui.py`, `sfg_two_plots.py`, `SPRM_Reflectivity_GUI.py`, and `evanescent_depth_gui.py` alongside the entry scripts.
- [x] Confirm the copyright holder and author names in `LICENSE` and `CITATION.cff` with all code contributors.
- [x] Record the author-provided manuscript panel associations for all eight entry scripts in `README.md`.
- [ ] Record the exact parameter settings and exported data for each panel. Include any input files needed to regenerate the panels and confirm the final manuscript panel labels.
- [ ] Run the available checks, then verify the complete package on a fresh Python environment and inspect representative exported figures.
- [x] Remove local absolute paths, machine identifiers, and edit-history notes from the publishable scripts. Keep the original-script backup outside this GitHub folder.
- [x] Check the 12 graphical interfaces for English window titles, controls, tabs, status text, and plot labels; keep user-facing error and export messages in English.
- [ ] Review each script's scientific assumptions and third-party data or licensing obligations for the final article figures.
- [ ] Add the final GitHub repository URL and a version-specific Zenodo DOI to `CITATION.cff` and the manuscript's Code availability statement.

Suggested release version: `v1.0.0` for the exact code used in the article. Later code changes should receive new versioned releases so that the article's DOI continues to identify the original code.
