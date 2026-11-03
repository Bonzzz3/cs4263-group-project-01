# Report analysis scripts

Scripts behind the report's Appendix A. Run them from the repository root, for example `python analysis/paired_tests.py`.

| Script | Produces |
|---|---|
| `paired_tests.py` | McNemar paired tests against B0 (Table A2), escape miss rates and unsafe rate by size band (Table A3), cost per exactly correct instance (Table A1) |
| `solver_scaling.py` | Solver states and solve time for 1–4 guards and up to 80 rooms (Table A4) |
| `solver_timing_repeats.py` | Repeat timings used for the medians in Table A4 |
| `figure1.py` | Fig. A1, saved as `analysis/figure1_tradeoff.png` |

`solver_scaling.py` and `solver_timing_repeats.py` take a few minutes because the 4-guard runs take about 50 s each. Timings depend on the machine.
