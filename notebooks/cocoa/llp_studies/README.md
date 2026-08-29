# LLP displaced-jet studies

Analysis scripts backing the numbered sections of [`../LLP_jet_findings.md`](../LLP_jet_findings.md).

All of these run on the **raw** COCOA generation file (`Out_Tree`), not on model output — they
are statements about the sample and the detector, not about HGPflow. None requires a trained
checkpoint. Defaults point at `cocoa_hss_val_15k.root`; pass `--raw` for another file and
`--events N` to change statistics.

```bash
cd /global/u1/a/agolub/HGPflow
PYTHONPATH=$PWD .venv/bin/python notebooks/cocoa/llp_studies/<script>.py
```

| script | doc section | headline result |
|---|---|---|
| `llp_decay_radius.py` | §7.1, header | where the decays are, in **full 3D** (the sample is forward): A⁰ decays 76.7% tracker / 15.2% calorimeter / 8.1% beyond. Splits the ceiling into a simulation limit and a genuine physics limit |
| `truth_composition.py` | §5 | truth composition is flat in `Lxy` (~68% charged everywhere); the reco "neutral hadron" excess is the trackless-charged relabelling |
| `track_availability.py` | §7.1, §6.1 | COCOA creates **zero** track objects past `Lxy` = 200 mm; per-particle deposit-vs-momentum offset split neutral/charged |
| `bending_charge_signed.py` | §6.1 | charge-signed `q·Δφ` isolates bending: −0.350 prompt → **0.0000** past 1 m |
| `decay_geometry.py` | §6.2, §6.3 | the `delta·L/R` mechanism — shift grows ~5× with per-particle 3D `L/R`, but `delta` is not constant, so treat it as scaling not calibration; decays are wide (56% too wide for one R=0.4 jet) |
| `jet_axis_track_coverage.py` | §6.4 | jet-axis accuracy is monotonic in track coverage: 65.7%/35.2%/9.5% coverage → 95.8%/57.7%/31.3% within ΔR<0.1 |
| `llp_axis_vs_decay_radius.py` | §6.5 | **the decisive test**: displaced jets that still have tracks (3D displacement 10–50 mm) reconstruct at **96.7%**, indistinguishable from prompt — HGPflow is not the bottleneck, the track collection is. Bins by 3D distance *and* Lxy to show what the transverse-only view hid |
| `ctau_and_boost.py` | §10.3 | generated **cτ = 1.00 m**, `m_a` = 55 GeV, βγ ~ 1.6; full closure of the region fractions, and proof that no cτ puts more than ~24% of decays in the calorimeter |
| `calo_geometry.py` | methodology | barrel ρ ≈ 1496.5 mm, endcap \|z\| ≈ 3220 mm |
| `unprojection_oracle.py` | §7.2 | un-projection ceiling: 28% → 59% axis recovery; one jet vertex suffices; needs ≤25 mm. `--straight-line` removes bending and gives 94.9% |
| `shower_axis_quality.py` | §7.3 | lever arm exists (83% of showers ≥2 topoclusters, 619 mm span) but axes are only good to ~41° from barycentres |
| `fake_rate_by_sample.py` | background (a) | reco-jet-centric **fake rate**, dR and overlap definitions — the dR one over-counts on displaced signal, so both are shown. Needs `hgpflow_v2` + eval files |
| `feature_migration.py` | background (b) | do *real* QCD jets get reconstructed with more **signal-like** (trackless, neutral-heavy) features under one model than another? Probably the larger of the two background effects. Needs `hgpflow_v2` + eval files |
| `overlap_match.py` | §8 pt 5, **§14–§15** | **reference-free** matching by energy overlap — no axis chosen anywhere, so one number is comparable across all decay regions; prints the dR-based efficiency alongside on the identical selection. Needs `hgpflow_v2` + eval files |
| `energy_response_by_region.py` | **§13** | closes §9.3: energy **is** recovered for unmatched tracker-region jets (0.92 at R<0.8) but **not** for unmatched calorimeter ones (0.36 — new negative result); calibration bias bounded at ≲6%. Needs `hgpflow_v2` + eval files |
| `tracking_reach_payoff.py` | **§12** | payoff vs assumed tracking reach: LRT-precedented 300 mm buys **+2.8 pts**, full tracker volume **+27.3 pts** (24.5% → 51.8%). Corrects §8's optimistic estimate |
| `region_efficiency_reco.py` | **§11** | **the bridge measurement** — real model output, full 30k: region split confirmed, region C dead (0.1%), calorimeter-decay reference change worth **0.9% → 26.9%**. Needs `hgpflow_v2` + eval files |
| `calo_depth_split.py` | **§17** | splits region B by decay depth — confirms the late-shower mechanism (dep_E/E **0.71 → 0.27 → 0.00**) and shows early-calo decays deposit *better* than prompt. Raw file only |
| `region_split_axis.py` | §10 | splits LLP jets by decay region; **calorimeter decays measure the FLIGHT direction, not the momentum axis** (28.4% → 47.5%, reference change only) |
| `calo_jet_baseline.py` | §9.1 | the track-free alternative is **worse**: calo-only jets give 14.3% on LLP jets vs 37.4% prompt, collapsing the same way — the geometric failure is not particle-flow-specific |
| `vertex_resolution.py` | §7.3 | calorimeter-only vertex fit gives ~1700 mm against a ~25 mm requirement — short by ~70× |

**One exception to "raw file only":** `region_efficiency_reco.py` is the bridge measurement — it
runs the real evaluation pipeline (`PerformanceCOCOA` + the model's merged predictions, ~10–30 min)
and reports the section-2 matching efficiency split by decay region, against both the momentum axis
and the A⁰ flight direction. It needs the `hgpflow_v2` package and the eval outputs.

## Reading these numbers

**Everything here is a best case.** `shower_axis_quality.py` and `vertex_resolution.py` use truth
`cell_parent_idx` links to group showers, which a real algorithm would have to do itself.
`unprojection_oracle.py` uses truth particles with the same constituent set for both jets, so
energy and clustering are perfect and only *position* varies — it bounds axis recovery, it does
not predict a matching efficiency. Do not compare its 28% baseline against the 0.44 plateau in
§2; different selections, different quantities. The trustworthy figure is the ratio (~2×).

**Decay grouping is approximate.** The raw file has no `particle_parent_idx`, so decay chains
cannot be reconstructed. Scripts that need "the particles from one `a` decay" group final-state
particles by production vertex rounded to 5 mm, which secondary decays (e.g. `K_S → ππ`
downstream) contaminate. This is why §6.2 reads the *slope* of column A rather than its absolute
values.

**One trap is deliberately left visible.** Column C of `decay_geometry.py` compares a vector-sum
axis against a coordinate centroid. That is an estimator mismatch, not physics, and it is worth
ΔR ≈ 1.19 on prompt groups — large enough to swamp the effect being measured. An earlier version
of this study made exactly that comparison and reached the wrong conclusion. Column A compares two
pT-weighted centroids and is the one to read.
