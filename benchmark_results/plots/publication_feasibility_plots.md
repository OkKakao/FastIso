# Publication Feasibility Plots

## Chart Contract

- Question: does FastIso occupy a publishable speed/fidelity niche for instrument-broadened fine-isotopic profiles?
- Takeaway: FastIso is not a faster fine peak enumerator; it is a faster dense-profile simulator once fine structure and peak broadening are required.
- Surface: static PNG figures for quick manuscript feasibility review.
- Palette: explicit blue/gold/orange/olive/pink roots plus neutral reference marks.

## Key Readout

- Large 12 kDa FastIso profile: 0.008841 s with L2 1.06e-03 vs IsoSpecPy.
- Large 12 kDa IsoSpecPy dense profile: 0.204 s.
- Large 12 kDa pyOpenMS dense profile: 0.3617 s with L2 7.25e-03.
- Large 12 kDa enviPat dense profile through the current wrapper: 0.7309 s with L2 1.10e-02.
- Large 12 kDa Brainpy dense profile: 0.002647 s with L2 4.57e-02; it is an aggregated baseline, not a fine-structure baseline.
- Large 12 kDa cluster +10 peak counts: brainpy=1, pyopenms=767, envipat=744.
- Auto-windowed largest case: 236.7 kDa with residual generation 0.003376 s.
- Best CZT window speedup: 7.22x for huge_240k at 0.2 Da and output dm=0.001.
- Best grid-matched CZT speedup: 7.01x for huge_240k at 0.2 Da.

## Plots

- fine_structure_speed_accuracy.png
- fine_structure_time_breakdown_large12k.png
- fine_cluster_preservation_large12k.png
- large_scale_fastiso_pruned_runtime.png
- czt_window_speedup_crossover.png
- czt_window_grid_matched_heatmap.png
- external_profile_batch10.png
- internal_direct_vs_pruned_speedup.png

## Interpretation

The current evidence supports a plausible methods/software paper if the claim is framed around dense, instrument-broadened profile simulation rather than peak-list enumeration.
The strongest figure is the speed-accuracy plot: FastIso remains near IsoSpecPy-level profile fidelity while avoiding the per-peak convolution cost that dominates IsoSpecPy, pyOpenMS, and enviPat dense-profile workflows.
Brainpy is useful as an aggregated baseline, but the cluster preservation plot shows why it should not be presented as a fine-structure competitor.
