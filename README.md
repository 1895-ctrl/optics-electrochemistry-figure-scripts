# Optics and electrochemistry figure scripts

Python scripts used to generate illustrative curves and figures for a review article. The models, parameters, and literature sources are described in each script. These simulations should not be presented as experimental measurements.

## Contents

| Script | Purpose | Manuscript panel | Command-line check or export | Included companion module |
| --- | --- | --- | --- | --- |
| `ATRIR_seiras_material_overlay.py` | Au/Ag/Cu ATR-SEIRAS material comparison | Box 3, Fig. 2a | `python ATRIR_seiras_material_overlay.py --selftest` | None |
| `gaussian_signal_gui.py` | Two Gaussian components and their sum | Box 1, Fig. 4b | `python gaussian_signal_gui.py --smoke-test --output gaussian_test.png` | None |
| `iscat_simplified_gui.py` | iSCAT/Mie comparison with separate camera benchmark | Box 2, Fig. 2b | `python iscat_simplified_gui.py --batch` | `iscat_darkfield_mie_comparison_gui.py` |
| `mie_scattering_efficiency_gui.py` | Ideal-sphere Mie scattering cross sections | Box 2, Fig. 1b | `python mie_scattering_efficiency_gui.py --selftest` | None |
| `nv_center_gui.py` | NV-center ODMR splitting curves | Box 2, Fig. 4b | GUI only | `nv_odmr_gui.py` |
| `SFG_two_plot_gui.py` | Polarization-dependent SFG curves | Box 3, Figs. 4b and 4c | `python SFG_two_plot_gui.py --selftest` | `sfg_two_plots.py` |
| `SPRM_Nano_Perturbation_GUI.py` | Effective-layer SPR perturbations | Box 2, Fig. 3b | `python SPRM_Nano_Perturbation_GUI.py --export-default output_dir` | `SPRM_Reflectivity_GUI.py` |
| `TIRF_mechanism_gui.py` | TIRF penetration-depth illustration | Box 1, Fig. 2b | `python TIRF_mechanism_gui.py --selftest` | `evanescent_depth_gui.py` |

All five companion modules named above are included alongside the eight entry scripts, for a total of 13 Python files. The companion modules can also be run directly for their more detailed interfaces. The code uses modeled or illustrative data unless a script explicitly identifies a literature-derived parameter; it does not contain experimental raw data.

## Installation and use

Use Python 3.10 or newer with Tk support for the graphical interfaces. From this directory:

```bash
python -m venv .venv
# Activate .venv before the following commands.
python -m pip install -r requirements.txt
python ATRIR_seiras_material_overlay.py --selftest
python mie_scattering_efficiency_gui.py --selftest
python gaussian_signal_gui.py --smoke-test --output gaussian_test.png
```

Activate the virtual environment before installing or running these commands. In Windows PowerShell, run `.\.venv\Scripts\Activate.ps1`; on macOS and Linux, run `source .venv/bin/activate`. Tk is supplied with many Python distributions but may require a separate operating-system package on Linux.

Run an entry script without its check or export flag to open its GUI. Exports are written to the path selected in the GUI unless a command-line output path is supplied. The release checks were run with Python 3.14.7, NumPy 2.5.3, Matplotlib 3.11.2, Pillow 12.3.0, miepython 3.3.0, openpyxl 3.1.5, pandas 2.3.3, and SciPy 1.18.1. The `requirements.txt` file lists the direct Python dependencies. Some scripts use optional fonts; if they are unavailable, the output uses a fallback font.

## Reproducing the article figures

The table above records the author-provided script-to-panel associations. The GUI defaults may not match the final published panels. Numerical parameter settings needed to reproduce the article results exactly are available from the authors upon request; contact the authors using the corresponding-author details in the article. The code and its self-tests are provided here, but the repository alone does not establish exact reproduction of each final panel. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for follow-up checks.

## Citation and license

The code is licensed under the MIT License; see [LICENSE](LICENSE). Third-party numerical sources and their separate terms are documented in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). Citation metadata is in [CITATION.cff](CITATION.cff). The public code repository is [optics-electrochemistry-figure-scripts](https://github.com/1895-ctrl/optics-electrochemistry-figure-scripts). If a versioned Zenodo archive is created later, its DOI should be added to the citation metadata and article.

Suggested manuscript wording:

> **Code availability.** The custom Python code used to generate the illustrative figures in this article is publicly available at https://github.com/1895-ctrl/optics-electrochemistry-figure-scripts. Numerical parameter settings required for exact reproduction of the article results are available from the authors upon request.

The statement should be adjusted to reflect the exact figures and data covered by the archived release.
