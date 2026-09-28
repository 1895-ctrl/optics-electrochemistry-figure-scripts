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

Run an entry script without its check or export flag to open its GUI. Exports are written to the path selected in the GUI unless a command-line output path is supplied. The scripts were tested with Python 3.14.7, NumPy 2.5.3, Matplotlib 3.11.2, Pillow 12.3.0, miepython 3.3.0, openpyxl 3.1.5, pandas 2.3.3, and SciPy 1.18.1. The `requirements.txt` file lists the direct Python dependencies. Some scripts use optional fonts; if they are unavailable, the output uses a fallback font.

## Reproducibility

The table above maps the scripts to panels in the associated review article. Default settings are illustrative and may differ from those used for the final article panels. Numerical settings required for exact reproduction are available from the corresponding authors upon request. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the scope of the code and its validation.

## Citation and license

This code is released under the [MIT License](LICENSE). Citation details for Gong Zhang and Sikai Lei are provided in [CITATION.cff](CITATION.cff). Third-party numerical sources and their separate terms are documented in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). Please also cite the relevant original publications when using their data.
