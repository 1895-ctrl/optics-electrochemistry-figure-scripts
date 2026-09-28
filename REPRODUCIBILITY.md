# Reproducibility

This repository contains eight figure-entry scripts and five companion modules.
Their manuscript-panel associations are listed in [README.md](README.md).
Outputs are modeled or illustrative curves, not experimental raw measurements.

The GUI defaults are not a record of the settings used for each final article
panel. Numerical settings required for exact reproduction of those panels are
available from the corresponding authors upon request. Consequently, running
a script with its defaults should not be taken as verification of a published
panel.

Built-in numerical checks and representative exports were run on Windows with
the Python and package versions listed in [README.md](README.md). These checks
test code behavior and selected model invariants; they do not establish
panel-by-panel agreement with the final article. GUI operation in a fresh
environment on other operating systems has not been verified.

Material constants, literature benchmarks, and other third-party numerical
inputs are identified in the scripts and summarized in
[THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). For quantitative use, select
parameters appropriate to the actual material and instrument. When citing a
particular state of this evolving repository, identify its Git commit.
