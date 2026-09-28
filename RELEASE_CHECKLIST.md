# Release checklist

This public repository contains the 13 Python scripts and their direct local companion modules. Unchecked items track follow-up reproducibility and archival work; they are not represented as completed.

- [x] Include `iscat_darkfield_mie_comparison_gui.py`, `nv_odmr_gui.py`, `sfg_two_plots.py`, `SPRM_Reflectivity_GUI.py`, and `evanescent_depth_gui.py` alongside the entry scripts.
- [x] Confirm the copyright holder and author names in `LICENSE` and `CITATION.cff` with all code contributors.
- [x] Record the author-provided manuscript panel associations for all eight entry scripts in `README.md`.
- [ ] Document the exact parameter settings and exported data for each panel when confirmed. Until then, readers seeking exact reproduction can contact the authors using the corresponding-author details in the article.
- [ ] Run the available checks, then verify the complete package on a fresh Python environment and inspect representative exported figures.
- [x] Remove local absolute paths, machine identifiers, and edit-history notes from the publishable scripts. Keep the original-script backup outside this GitHub folder.
- [x] Check the 12 graphical interfaces for English window titles, controls, tabs, status text, and plot labels; keep user-facing error and export messages in English.
- [x] Audit embedded third-party numerical inputs, add source and license notes in `THIRD_PARTY_DATA.md`, and correct the two optional Ordal Pt/Pd constants. This does not replace final figure-parameter review or legal advice.
- [x] Add the GitHub repository URL to `CITATION.cff` and the suggested Code availability statement.
- [ ] If a tagged Zenodo archive is created, add its version-specific DOI to `CITATION.cff` and the article.

When an archival version is ready, tag the exact code used in the article. Later changes should receive new versioned releases so that a future DOI continues to identify the archived code.
