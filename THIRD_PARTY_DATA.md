# Third-party data and literature provenance

This repository contains original Python modeling code and small, cited sets of
published numerical constants. It does not bundle experimental raw data,
third-party article PDFs, supplementary-information PDFs, screenshots, or
reproductions of published figures. The MIT license in `LICENSE` covers the
authors' code; it does not relicense the cited publications, databases,
manufacturer documents, or third-party Python dependencies.

## Optical-constant tables

`mie_scattering_efficiency_gui.py` and `SPRM_Reflectivity_GUI.py` contain
tabulated refractive indices and extinction coefficients. Their embedded
values were checked against the CC0-licensed
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database)
(database commit `c5c2f188e848453def5970e347399d653df2ffc2`, accessed
2026-09-28). All 952 rows checked matched the corresponding database rows
within the scripts' rounded precision (258 Mie rows and 694 SPRM rows;
maximum comparison tolerances: 0.55 nm in wavelength and 0.006 in each of
`n` and `k`). This is a provenance and transcription check, not a claim that
the optical constants are universal for every specimen or that their original
measurements were independently reproduced.

The Mie tables correspond to database datasets for Au/Ag/Cu (Johnson and
Christy), Pt (Werner), Ni (Johnson and Christy), TiO2 (Jolivet anatase), ITO
(Konig), and Si (Green 2008). The SPRM tables correspond to Au/Ag/Cu
(Johnson and Christy; Olmon, Babar and Weaver, or McPeak as individually
named in the script), Pt (Rakic Brendel-Bormann and Tselin), and Pd (Johnson
and Christy; Rakic Brendel-Bormann; Palm). The selected original-paper
citations and wavelength ranges are recorded in the scripts and the linked
database records. Please cite both the relevant original measurement and
[Polyanskiy, *Scientific Data* 11, 94 (2024)](https://doi.org/10.1038/s41597-023-02898-2)
when using those optical constants in a publication. The database's CC0
notice applies to its compilation; original source papers retain their own
publication terms.

## Other literature-derived numerical inputs

| Scripts | Numerical source and scope |
| --- | --- |
| `ATRIR_seiras_material_overlay.py` | Drude constants for Au, Ag, Cu, Pt, and Pd from [Ordal et al., *Applied Optics* 24, 4493–4499 (1985), Table I](https://doi.org/10.1364/AO.24.004493). The optional Pt and Pd parameter pairs are `(41500, 558)` and `(44000, 124)` cm^-1, respectively; the article-linked material comparison uses Au, Ag, and Cu. |
| `sfg_two_plots.py`, `SFG_two_plot_gui.py` | Small sets of CO/Pt(111) optical parameters, geometry, and comparison values from [Li et al., *Topics in Catalysis* 61, 751–762 (2018)](https://doi.org/10.1007/s11244-018-0949-7), especially Table 1 and Fig. 2. The publication is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the scripts do not include its figure artwork or text. |
| `nv_odmr_gui.py` | Nuclear-spin constants attributed in the script to [Felton et al., *Physical Review B* 79, 075203 (2009)](https://doi.org/10.1103/PhysRevB.79.075203). Model-B lifetimes and branching probabilities use [Robledo et al., *New Journal of Physics* 13, 025013 (2011)](https://doi.org/10.1088/1367-2630/13/2/025013), Table 1; the approximate 178 ns singlet lifetime is estimated from that paper's Fig. 3(d) temperature fit, not Table 1. User-adjustable pump and microwave settings are illustrative model choices. |
| `iscat_darkfield_mie_comparison_gui.py` | The 0.1% shot-noise/SNR-10 benchmark follows [Cole et al., *ACS Photonics* 4, 211–216 (2017)](https://doi.org/10.1021/acsphotonics.6b00912). The approximately 0.03% BSA contrast follows [Piliarik and Sandoghdar, *Nature Communications* 5, 4495 (2014)](https://doi.org/10.1038/ncomms5495). These are separate literature benchmarks, not a shared camera calibration. |
| `iscat_simplified_gui.py` | The camera and noise comparison references [Wu et al., *Journal of Physical Chemistry C* (2025)](https://doi.org/10.1021/acs.jpcc.4c07989) and [its supporting information, Fig. S4](https://doi.org/10.1021/acs.jpcc.4c07989.s001). The supporting-information PDF is not included here; its separate CC BY-NC 4.0 terms are not applied to the code. |
| `evanescent_depth_gui.py`, `mie_scattering_efficiency_gui.py`, `SPRM_Reflectivity_GUI.py` | Glass and medium dispersion equations or limited numerical specifications are attributed within the scripts to the original papers and manufacturers, including [SCHOTT optical-glass datasheets](https://www.schott.com/en-us/products/optical-glass-p1000267/downloads), [SCHOTT D 263 M](https://www.schott.com/en-us/products/d-263-p1000318/downloads), and [Cargille immersion-oil specifications](https://www.cargille.com/wp-content/uploads/2020/01/2020-Immersion-Oil-Catalog.pdf). Derived Cauchy approximations are labeled as such in the scripts, not presented as measured curves. |
| `mie_scattering_efficiency_gui.py` | Camera specification examples are attributed to [Hamamatsu](https://www.hamamatsu.com/us/en/product/cameras/cmos-cameras/C13440-20CU.html) and [Teledyne Photometrics](https://www.teledynevisionsolutions.com/en-in/products/prime-bsi/). No vendor software or manual is redistributed. |

All other plotted curves are generated by the scripts' equations and chosen
parameters. Literature values may be idealized, rounded, interpolated, or
specimen-dependent; consult the cited source and the script's assumptions
before treating any output as a quantitative experimental prediction. Exact
settings for the final article panels can be requested from the authors, as
explained in `README.md`.
