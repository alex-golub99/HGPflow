# HGPflow on displaced (LLP) jets — HSS test set

**Sample:** HSS (`H → aa`, pseudoscalar `a` decaying at a displaced vertex), 30k test events
(`cocoa_hss_pflow_30k.root`, with a PPflow baseline).
**Models compared:** HGPflow trained on **qqbar** (out-of-domain) vs trained on **hss** (in-domain),
plus the parametric **PPflow** baseline.
**LLP tag:** a truth particle is *LLP-origin* if its production displacement
`Lxy = √(prod_x² + prod_y²) > 10 mm`; a jet's **LLP fraction** is the pT-weighted fraction of its
constituents that are LLP-origin. Unless noted, "LLP jets" means jets with LLP fraction ≥ 0.9.
**Matching:** truth↔reco jets, anti-kT R=0.4, matched if `ΔR < 0.1` (tight, deliberately).

> ### Where these decays actually happen — read this first
>
> Taken from the truth decay graph (the A⁰ is pdg 36; `node_decx/y/z` is its decay vertex), and
> classified in **full 3D** — these decays are strongly **forward**, so a transverse-only test is
> misleading. Tracker volume = ρ < 1496.5 mm **and** |z| < 3220 mm.
>
> | | inside tracker | in calorimeter | beyond calorimeter |
> |---|---|---|---|
> | **A⁰ decays** (all) | 76.7% | **15.2%** | **8.1%** |
> | **LLP jets analysed here** | 84.8% | **11.2%** | 4.0% |
>
> A⁰ 3D flight distance: median **1103 mm**, 90% **6307 mm**, 99% 14864 mm. Median Lxy 291 mm but
> median |z| **928 mm** — and of the decays that leave the tracker, **89.2% exit through the
> endcap**, only 22.7% through the barrel. The generated lifetime is **cτ = 1.00 m** with
> `m_a` = 55 GeV (βγ ~ 1.6): the lifetime is long, but the heavy slow A⁰ and the exponential's
> front-loading put ~77% of decays in the tracker regardless — no cτ exceeds ~24% in the
> calorimeter (§10.3).
>
> Three consequences:
>
> 1. **The sample spans all three regions.** ~15% of A⁰ decays are at or past the calorimeter, and
>    genuine calorimeter-decay jets are ~11% of the LLP jets analysed — not a negligible tail.
> 2. **A tracking extension addresses the ~77–85% decaying inside the tracker volume** (§7.1, §8),
>    but the ~15% decaying *in* the calorimeter can never be tracked by anything, and need
>    CalRatio / timing / shower-shape methods instead. The ~8% decaying beyond the calorimeter are
>    a genuine acceptance loss, invisible to any jet reconstruction.
> 3. **`Lxy` is a transverse proxy for a sample that is substantially forward**, and the sections
>    where that mattered have been corrected: §6.5 now bins by **3D production distance** and §6.2
>    uses the per-particle 3D `L/R`. §6.1 deliberately keeps `Lxy` (bending depends on the
>    *transverse* lever arm — see the note there); §7.3 remains `Lxy`-binned, which is cosmetic for
>    its order-of-magnitude conclusion; and the `llp_lxy_mm = 10` *tag* is safe (methodology
>    notes).

---

## Executive summary

1. **In-domain training sharpens jet energy resolution dramatically** — the hss-trained model has
   ~28% tighter jet-pT resolution than the qqbar model on all jets and **~42% tighter on LLP jets**,
   with comparable bias and far better than PPflow. §15 adds the bigger half of the story: it also
   buys **+11 points of energy containment** on displaced jets (overlap metric, all jets, no
   survivor selection) — a gain the ΔR metric was structurally blind to.
2. **LLP matching efficiency is low (~0.44 ceiling) for *all* models, including the in-domain one.**
   This is **not** an energy loss, a fake-rate artifact, or a model deficiency. §6.5 makes the last
   point directly: displaced jets that still have tracks (3D displacement 10–50 mm) reconstruct at
   **96.7%** axis accuracy — statistically indistinguishable from the 95.8% of prompt jets. HGPflow handles displacement; it is the absence of
   tracks that it cannot handle, and that is upstream of the network.
   Note also that the strong LLP *resolution* in §1 is measured on matched jets only — the
   well-tracked subset — so efficiency, not resolution, is the honest headline (§6.5).
3. **Root cause: geometric displacement, and the operative variable is track coverage.** Without
   tracks (the LLP decays beyond the radius COCOA tracks — though still *inside* the tracker volume,
   see the callout above), the charged decay products are reconstructed from calorimeter energy alone — as *neutrals* — and that energy lands at its **deposit position**, not
   the particle's **momentum direction**. The energy is captured (~85%) but angularly displaced, so it
   fails the momentum-axis match. Jet-axis accuracy tracks the pT fraction carrying a track almost
   perfectly — 65.7% / 35.2% / 9.5% coverage gives 95.8% / 57.7% / 31.3% of jet axes within ΔR<0.1
   (§6.4).
4. **The displacement grows with the decay radius `Lxy`** — demonstrated directly (§3.4) —
   confirming this is a limit of the information *reaching* a prompt-designed reconstruction, *not* a
   training gap (and not an irreducible one — see point 7).
   §6 derives the precise mechanism: the shift is not the raw displacement but the decay's angular
   asymmetry `delta` rescaled by `L/R`, which is why it grows with the decay radius (monotonic,
   approximately linear at small `L/R` — §6.2).
5. **The reconstructed "neutral hadrons" in these jets are displaced *charged* hadrons.** The truth
   composition of HSS particles is ~68% charged at *every* `Lxy`; the reco label flips because charge
   in HGPflow is track-seeded, and the training target itself relabels trackless charged hadrons as
   neutral. §5.
6. **Both candidate fixes were tested, and both fail on the existing samples.** Displaced-vertex-aware
   tracking is not implementable: COCOA creates *zero* track objects beyond ~200 mm production
   radius (and, for forward production, beyond |z| ≈ 600 mm even at small `Lxy` — §7.1). Vertex-based
   un-projection works in principle (28% → 59% axis recovery) but needs the decay vertex to ≤ 25 mm,
   and calorimeter-only vertex finding delivers ~1700 mm. §7.
7. **The decays span all three detector regions, and the sample is strongly forward.** From the
   truth decay graph: 76.7% of A⁰ decays are inside the tracker volume, **15.2% in the
   calorimeter**, 8.1% beyond it; of those leaving the tracker, 89.2% exit through the *endcap*
   (§7.1). So the ceiling has two distinct causes — a *simulation* tracking limit for the ~77%
   inside the tracker, and a genuine *physics* limit for the ~15% decaying in the calorimeter,
   which nothing can track. Note `Lxy` is a transverse proxy for a forward sample, so the
   `Lxy`-binned tables mix the two.
8. **What is actionable now is a metric change, not a reconstruction change — and the right
   reference is region-dependent** (§10). For tracker-volume decays the momentum axis is the correct
   reference; for calorimeter decays the reconstructed axis measures the A⁰ **flight direction**
   instead (median ΔR 0.123 vs 0.738), and scoring against it helps with no algorithm change
   (those are truth-level figures; on real model output the gain is 0.9% → 26.9% — point 9).
   A single global reference cannot serve both populations.
9. **Measured on real model output, not just truth-level** (§11): the region split reproduces
   exactly (84.7 / 11.5 / 3.8%), region C is confirmed dead (0.1% on both references), and the
   decomposition sums to §1's independently measured 37.1%. The calorimeter-decay reference change
   is worth **0.9% → 26.9%** — a ~30× gain, larger than the truth-level 1.7× because those numbers
   carried a momentum-space clustering bias. That bias is confined to calorimeter decays: the
   truth-level machinery transfers to real output within a few points for prompt (95.8% → 92.4%),
   mixed (57.7% → 52.0%) and tracker (~31% → 25.3%) jets, and collapses only in region B.
10. **The ceiling is largely a metric artifact (§14).** A reference-free energy-overlap match —
   calibrated so that it is slightly *stricter* than ΔR<0.1 on prompt jets — takes tracker-region
   LLP jets from **25.3% to ~74%** (range 64–80% over the association-radius scan, §14.4), and all
   LLP jets from **24.5% to ~66%** (range 57–71%). Three quarters of
   displaced jets are being reconstructed with most of their energy in one reco jet; only the
   *pointing* fails. This is worth more than the entire tracking extension (§12: +27.3 pts), costs
   nothing, and is available today. Region B is capped by genuinely missing energy (§13.2) and
   region C stays dead.
11. **Background side measured (§16): in-domain training costs nothing, and the reconstruction
   choice is worth ~9× in S/B.** §1's fake-rate column is an artifact — measured by energy overlap
   the real rates are 7.0% (hss) / 6.7% (qqbar) / 8.6% (PPflow), so HGPflow *beats* PPflow rather
   than losing to it, and the hss model's extra jets are real (~10 real per extra fake). On QCD
   benchmarks HGPflow leaks **1.0–1.2%** of ordinary jets into trackless signal-like territory
   against PPflow's **7.7–11.2%**, at equal signal efficiency — ≈9× better S/B on that variable.
   In-domain training does *not* shift prompt jets toward signal-like features (2.2% vs 2.0%).
12. **Region B is three populations, not one (§17).** Deposited energy collapses
   **0.71 → 0.27 → 0.00** as remaining calorimeter depth falls below ~2 m, confirming §13.2's
   late-shower mechanism by direct measurement — but decays near the ECAL face deposit *better*
   than prompt jets (0.66 vs 0.543). The CalRatio-relevant band (HCAL, ~2200–2800 mm) deposits 27%
   with 48.7% flight-axis accuracy, and is only **~4% of all decays** — so for that physics the
   binding constraint is statistics, not reconstruction.
13. **The track-free alternative is worse, and a measured ceiling now exists.** Calorimeter-only
   jets are worse than HGPflow everywhere and collapse identically with decay distance (§9.1) — the
   failure is not particle-flow-specific. On the ΔR metric, the measured achievable ceiling is
   **~52%** (§12.3; an earlier ~85% estimate in §10.2 is superseded): full-volume tracking for the
   ~85% of jets decaying in the tracker volume, the flight-direction reference for the ~11% in the
   calorimeter, and a hard floor of ~4% that deposit nothing and are permanently lost. On the
   energy-overlap metric, ~66% is already achieved today (point 10).

---

## 1. Model comparison: in-domain training buys resolution

Jet-pT residuals `(pT_reco − pT_truth)/pT_truth` for the three collections, all jets and LLP-only
(LLP fraction ≥ 0.9):

![All-jet residuals: PPflow vs qqbar-trained vs hss-trained](hss_test_qqbar_vs_hss_jet_residuals.png)

![LLP-jet residuals: PPflow vs qqbar-trained vs hss-trained](hss_test_qqbar_vs_hss_jet_residuals_llp90.png)

| resolution (IQR of pT residual) | PPflow | HGPflow (qqbar) | HGPflow (hss) |
|---|---|---|---|
| all jets | 0.293 | 0.232 | **0.167** (~28% tighter than qqbar) |
| LLP jets | 0.366 | 0.332 | **0.194** (~42% tighter than qqbar) |

Bias (median residual) is comparable between the two models and much better than PPflow on LLP jets
(PPflow systematically under-measures displaced-jet pT by ~18%).

> **The LLP row is survivor-selected.** These residuals are computed on *matched* jets only, which
> for LLP jets means predominantly the well-tracked low-`Lxy` population (§6.5). Read 0.194 as "the
> resolution on displaced jets HGPflow successfully finds", not as the resolution on displaced jets
> in general. §6.5 shows why the two differ so much.

> **Caveat on "matching efficiency" as a metric.** The `f_matched` printed alongside these
> distributions is *reco-jet purity* (fraction of matched pairs passing the cut), whose denominator
> tracks jet **multiplicity** — so the hss model's slightly lower `f` reflects it producing *more*
> jet candidates, not finding fewer true jets. Measuring true jet-finding efficiency (fraction of
> **truth** jets recovered), the hss model is marginally ahead:
>
> | | reco jets/ev | fake rate (ΔR<0.1) | true-jet efficiency |
> |---|---|---|---|
> | truth | 2.816 | — | — |
> | PPflow | 1.894 | 0.467 | 35.9% |
> | HGPflow (qqbar) | 1.926 | 0.478 | 35.7% |
> | HGPflow (hss) | 2.165 | 0.517 | **37.1%** |
>
> **The fake-rate column is superseded by §16.1 and ranks the models backwards.** Those values are
> ΔR-based, and on displaced signal ΔR counts a correctly reconstructed jet as a fake because no
> truth jet points at it (§14). Measured by energy overlap, the real fake rates are **7.0% (hss) /
> 6.7% (qqbar) / 8.6% (PPflow)** — HGPflow is *better* than PPflow, not worse, and the hss model's
> extra jets are real ones (~10 real per extra fake, §16.1).

---

## 2. The puzzle: why is LLP matching efficiency only ~0.44?

Truth-jet-centric matching efficiency vs pT for LLP jets (fraction of truth LLP jets with a reco jet
within ΔR), for the three collections and, in the right panel, LLP vs prompt for the hss model:

![LLP matching efficiency vs pT (ΔR<0.1)](hss_test_llp_eff_vs_pt_dr10.png)

Key observations:

- Efficiency **rises monotonically and plateaus at ~0.44** by 65 GeV — it does **not** turn over at
  high pT (an apparent drop at ~150 GeV was a 25-jet statistical fluctuation).
- The low *integrated* efficiency (~36%) is a **soft-spectrum effect**: ~25k LLP jets sit at 15 GeV
  vs ~25 at 160 GeV, so the low-efficiency threshold bins dominate the average.
- **All three collections plateau at the same ~0.44** above 40 GeV. In-domain training helps only in
  the soft regime (15 GeV: hss 0.165 vs qqbar 0.147, a >10σ gain) and washes out at high pT.
  *(§15 later shows this "washing out" is a property of the ΔR metric, not of the training: the
  geometric ceiling saturates identically for every reconstruction, hiding an +11-point containment
  gain that the overlap metric reveals.)*
- The plateau **ceiling far below 1** is the real signal — over half of even high-pT LLP jets fail
  the tight match. That both models hit the *same* ceiling means the limit is **upstream of the
  network**, not a training gap.

---

## 3. Diagnosing the ceiling

### 3.1 Are the jets lost, or just displaced? — efficiency vs ΔR cut

![Integrated LLP efficiency vs ΔR match cut](hss_llp_eff_vs_dr.png)

Loosening the match cone recovers efficiency substantially but plateaus around **~0.6 even at
ΔR<0.5** (larger than the R=0.4 jet). So the jets are *partly* displaced-but-present and *partly*
something else — motivating a direct energy measurement.

### 3.2 Is the energy there? — reco energy in a cone

Sum of reconstructed particle pT within a cone of each truth LLP jet axis, over the truth jet pT,
split by matched vs unmatched:

![Cone energy ratio, matched vs unmatched LLP jets](hss_llp_cone_energy_ratio.png)

| | Eratio (ΔR<0.4 cone) | Eratio (ΔR<0.8 cone) | ⟨charged⟩ | ⟨neut. had⟩ | ⟨photon⟩ |
|---|---|---|---|---|---|
| matched | 0.95 | 1.08 | 0.35 | 0.45 | 0.20 |
| unmatched | 0.43 | **0.85** | 0.07 → 0.13 | 0.74 → 0.70 | ~0.19 |

- **Matched jets** capture ~95% of the energy and retain a real **charged** component (0.35) — i.e.
  they kept tracks, and they reconstruct cleanly.
- **Unmatched jets** capture only 43% within a 0.4 cone but **85% within a 0.8 cone** → the energy is
  **present but displaced** beyond 0.4, not lost. What's recovered is ~90% **neutral** (⟨charged⟩ only
  0.07–0.13): the trackless-charged → neutral flip, as expected.
- Only ~15% is genuinely missing at cone 0.8 (calorimeter under-measurement + any invisible muon/
  neutrino energy).

**Takeaway:** the failure is *position*, not energy. The neutral pathway is the correct *energy*
fallback with no track; it costs resolution, not energy.

> **Read the class columns as reco labels, not physics.** `<charged>`, `<neut. had>` and `<photon>`
> above come from `hgpflow_dict['hgpflow_class']` — the *predicted* class. They do **not** say these
> jets are physically neutral. See §5: at truth level these jets are ~69% charged, and the apparent
> neutral-hadron excess is the trackless-charged relabelling, not a property of the LLP decay.

### 3.3 How far is the energy displaced? — energy centroid shift

ΔR between the truth jet momentum axis and the pT-weighted centroid of the reconstructed energy:

![Reco energy centroid shift from truth axis](hss_llp_energy_centroid_shift.png)

- **Matched:** sharp peak at **0.06** — energy sits on the momentum axis.
- **Unmatched:** **median 0.28, broad and flat out to ~0.8** — displaced well past the 0.1 match cut,
  by a *geometry-dependent* amount with no single scale (the flatness reflects the range of decay
  radii, boosts and opening angles in the sample). A low-shift tail corresponds to a secondary
  failure mode: energy near the axis but **fragmented** into sub-threshold sub-jets.

### 3.4 The causal link — displacement grows with decay radius `Lxy`

Per-jet energy centroid shift vs the jet's LLP decay radius (pT-weighted mean Lxy of its LLP
constituents; alignment self-validated against the stored LLP flag):

![Energy centroid shift vs jet Lxy](hss_llp_shift_vs_lxy.png)

The median shift **rises with `Lxy`** and crosses the ΔR=0.1 match threshold around the tracker
radius: the farther out the LLP decays, the farther the reconstructed energy drifts from the momentum
direction, and the jet stops matching. This is the direct, causal demonstration that the efficiency
ceiling is a **displacement effect**.

Supporting context (LLP-origin particle displacement vs pT):

![LLP-origin particle Lxy vs pT](hss_llp_lxy_vs_pt.png)

---

## 4. Conclusions from the initial analysis

- **HGPflow reconstructs the LLP energy** — the calorimeter records it and the algorithm recovers
  ~85% of it, correctly as neutral particles since the tracks are absent.
- **The LLP jet-matching ceiling (~0.44 at ΔR<0.1) is a geometric/positional limit**, not an energy
  loss, a fake artifact, or a model shortfall. Without tracks, energy is placed at its deposit
  position rather than its momentum direction, and that offset grows with the decay radius.
- **In-domain (hss) training delivers a large resolution improvement** (~28–42% tighter jet-pT IQR)
  and a marginal jet-finding gain, but **cannot lift the matching ceiling** — the missing information
  (original momentum direction) is not in the inputs.

The original closing bullet proposed two possible fixes — displaced-vertex-aware tracking, or a
displacement-aware clustering/matching scheme. **Sections 5–8 test both. Both fail on the existing
samples**, for different and independently measured reasons. §8 states what is actually actionable.

---

## 5. The reconstructed "neutral hadrons" are displaced *charged* hadrons

`H → aa` with `a → qq̄` produces ordinary quark jets, so the displaced jets should have ordinary
composition. They do. Truth composition of HSS particles by production radius (3k val events,
pT > 0.5 GeV, |η| < 2.5, pT-weighted):

| Lxy [mm] | charged | neut. had | photon | ch. **with** track | ch. **without** track |
|---|---|---|---|---|---|
| 0–1 | 67.6% | 13.3% | 19.1% | 67.6% | 0.0% |
| 1–10 | 65.9% | 13.6% | 20.5% | 65.7% | 0.2% |
| 10–50 | 69.3% | 13.0% | 17.7% | 63.3% | 6.0% |
| 50–200 | 68.9% | 10.8% | 20.3% | 11.7% | 57.2% |
| 200–1000 | 69.1% | 10.8% | 20.1% | 0.0% | 69.1% |
| >1000 | 62.4% | 18.1% | 19.5% | 0.0% | 62.4% |

The composition is **flat in `Lxy`** — displaced jets are not neutral-rich in any physical sense.
Against the §3.2 reco numbers for unmatched jets:

| | charged | neut. had | photon |
|---|---|---|---|
| truth (Lxy 200–1000) | 69% | 11% | 20% |
| reco (unmatched jets) | 7% | 74% | 19% |

**The photon fraction is the control and it is preserved (20% → 19%)** — photons are genuinely
neutral and are labelled correctly. Charged goes 69% → 7% while neutral hadron goes 11% → 74%: a
near-exact 1:1 transfer of ~62% of the jet energy from one class into the other.

This is not only an inference artifact. At `hgpflow_v2/dataset/dataset_mini.py:179-184` the *training
target* is redefined — `flat_trackless_chhad_and_e_mask` shifts class 0 (charged hadron) to class 3
(neutral hadron) — so the network is explicitly taught that a trackless charged pion is a neutral
hadron. Combined with `proxy_is_charged = track_eye.sum(dim=2).bool()`
(`hgpflow_v2/models/hgpflow_model.py:108`), charge is structurally track-seeded and there is no path
to a "charged" output without a track. Energy fraction HGPflow can even in principle call charged:

| Lxy [mm] | truly charged | can be called charged | forced to neutral |
|---|---|---|---|
| 0–10 | 67.5% | 67.4% | 0.0% |
| 10–200 | 69.0% | 25.8% | 43.2% |
| >200 | 67.9% | 0.0% | 67.9% |

**Consequence.** For jet *energy* this is benign — the calorimeter does not care about the label. It
matters for hadronic calibration (charged and neutral hadron response differ) and for any downstream
observable built on charged multiplicity or vertex structure.

---

## 6. Why the `a`'s straight flight does not protect the jet axis

The `a` is electrically neutral, so it does not bend: it flies straight from the primary vertex and
its decay vertex sits exactly on the `n̂` ray at radius `L`. Momentum conservation makes `Σp⃗` point
along `n̂`. Neither fact rescues the match, and the reason is worth stating precisely.

### 6.1 The decay products *are* charged, but bending switches off with `Lxy`

The field fills the tracker volume regardless of what entered it, so the charged decay products bend
from the decay point outward — but with a shortened lever arm. Bending is charge-coherent while
geometry is not, so signing Δφ by charge isolates it (π±/K±/p, pT > 1 GeV):

| Lxy [mm] | N | median q·Δφ (bending) | median \|Δφ\| (total) |
|---|---|---|---|
| 0–1 | 42878 | **−0.350** | 0.352 |
| 1–10 | 2061 | −0.226 | 0.226 |
| 10–50 | 4368 | −0.275 | 0.276 |
| 50–200 | 10939 | −0.245 | 0.247 |
| 200–1000 | 18399 | −0.093 | 0.170 |
| >1000 | 3567 | **0.0000** | 0.166 |

At prompt radii the offset is *entirely* bending (signed and unsigned medians agree to three
decimals) — which is the method's own sanity check, since a particle born at the origin has zero
geometric displacement by construction. Past 1 m the coherent part is exactly zero while ~0.17 of
incoherent offset remains: bending has switched itself off and what survives is pure geometry.

> **Why `Lxy` and not 3D distance here.** Bending is driven by the **transverse** lever arm, so
> `Lxy` is the correct ordering variable for *this* table even though the sample is forward — a
> particle at 3D distance 500 mm that is mostly forward still has nearly the full transverse path
> to the calorimeter and bends like a prompt one. Re-binning by 3D distance actually *mixes*
> bending behaviour (the 200–1000 mm bin then reads −0.217 instead of −0.093). Track availability
> and the jet-axis result are the opposite case and need 3D distance (§6.5, §7.1).
>
> **How to read this table.** It identifies *what kind* of offset dominates at each radius — that
> is what licenses the straight-line un-projection of §7.2, which is only well-posed once bending
> is gone. It does **not** predict matching efficiency: these are per-particle medians in one
> coordinate, computed as though deposit positions were used for every particle. Prompt jets never
> live in that hypothetical — they have tracks. For the jet-level consequence see §6.4, and note
> that the per-particle offset actually *shrinks* with `Lxy` (0.35 → 0.17) while matching gets
> worse, so offset size is not the operative variable. The two effects are conveniently
> anti-correlated in `Lxy` — where bending is large, tracks still exist (§7.1); where tracks
> vanish, bending vanishes too.

Per-particle deposit-vs-momentum ΔR, split by charge, shows the same trade-off:

| Lxy [mm] | neutral | charged |
|---|---|---|
| 0–1 | 0.004 | 0.307 |
| 10–50 | 0.027 | 0.269 |
| 50–200 | 0.095 | 0.307 |
| 200–1000 | 0.292 | 0.350 |
| >1000 | 0.370 | 0.447 |

### 6.2 The mechanism: `delta · L/R`

For a product emitted at angle `θ` from `n̂`, decaying at radius `L`, landing on a calorimeter at
radius `R`:

```
momentum angle from n̂ =  θ
deposit  angle from n̂ =  θ · (1 − L/R)      <- deposits collimate toward n̂
```

Both distributions stay *centred* on `n̂`, so a perfectly balanced decay would suffer no centroid
shift at all. But a jet axis is an energy-weighted **angular position**, not a vector sum. The
centroid of an asymmetric decay sits off-axis at some `delta`; the rescaling drags the deposit
centroid to `delta(1 − L/R)` while the momentum centroid stays at `delta`. The residual is

```
jet-axis shift  =  delta · L/R          (linear in decay radius, zero only for a balanced decay)
```

Measured, comparing like with like (both pT-weighted centroids, so no estimator mismatch):

| Lxy [mm] | N | ΔR(momentum centroid, deposit centroid) |
|---|---|---|
| 0–10 | 2374 | 0.089 |
| 10–200 | 870 | 0.093 |
| 200–500 | 704 | 0.200 |
| 500–1000 | 581 | 0.327 |
| >1000 | 243 | 0.356 |

Subtracting the prompt baseline, the shift rises steadily with displacement, as the mechanism
requires.

**A correction to an earlier version of this section.** That version divided `Lxy` by the *barrel*
radius (1496.5 mm) to form `L/R` and obtained `delta` = 0.497 rad, against a measured angular
spread of 0.554 rad — an apparently striking ~10% agreement. That agreement was an artifact of
using a barrel radius for a strongly forward sample. Computing `L/R` correctly — per particle, as
the ratio of the true 3D flight distance to the true 3D distance to that particle's deposit
(barrel or endcap) — gives:

| ⟨L/R⟩ bin | N | median shift | implied `delta` = shift/⟨L/R⟩ |
|---|---|---|---|
| <0.05 (baseline) | 2575 | 0.083 | — |
| 0.05–0.15 | 421 | 0.074 | −0.139 |
| 0.15–0.30 | 426 | 0.145 | 0.263 |
| 0.30–0.50 | 436 | 0.234 | 0.374 |
| 0.50–1.01 | 914 | **0.407** | 0.397 |

**What survives:** the shift grows monotonically with `L/R`, by roughly a factor of 5 across the
range, which is the mechanism's central prediction and is solid.

**What does not:** `delta` is *not* constant across bins (0.26 → 0.40), so the simple
`shift = delta · L/R` form is an approximation that degrades at large `L/R` — unsurprising, since
its derivation assumed small emission angles and these decays have a 0.554 rad spread. The
quantitative closure with the measured angular spread claimed earlier does not survive the fix.
Treat §6.2 as establishing the *mechanism and its scaling*, not a calibrated relation.

### 6.3 These decays are wide, which is why `delta` is large

| quantity | value |
|---|---|
| true `m_a` (fixed at generation; §10.3) | **55 GeV** |
| true A⁰ boost `E/m` (median; §10.3) | **1.86** (βγ median 1.57) |
| visible vertex-group invariant mass (median) | 18.3 GeV |
| visible vertex-group `E/m` (median) | 3.1 |
| pT-weighted angular spread about the `a` axis (measured) | 0.554 rad |
| decays too wide to form one R=0.4 jet | **55.6%** |

> An earlier version labelled the 18.3 GeV / 3.1 rows as the decay mass and boost; they are
> *visible* vertex-group quantities (neutrinos, acceptance and the 5 mm grouping all lose mass).
> The true values from the decay graph — `m_a` = 55 GeV, boost 1.86 — make the conclusion
> **stronger**: the A⁰ is heavy and *slow*, so its decays are even wider than the visible numbers
> suggested, and its lab decay length is compressed (§10.3).

So the jets being matched are frequently *sub-parts* of a decay, and a sub-part inherits no
protection from momentum conservation. The `(1 − L/R)` collimation additionally makes deposit space
narrower than momentum space, so anti-kT merges differently in the two and fragments fall below the
8 GeV / 2-constituent thresholds in `hgpflow_v2/performance/jet_helper.py:66` — this is the §3.3
"energy near the axis but fragmented into sub-threshold sub-jets" tail, now with a mechanism.


### 6.4 The net effect at jet level: track coverage is the variable

Building each truth jet's axis the way HGPflow effectively does — direction from the **track**
where a track exists, from the **deposit position** where it does not — and comparing to the truth
momentum axis (2000 events; same constituent set for both jets, so only *position* varies):

| jet type | N | pT frac with a track | **realistic** med ΔR | <0.1 | <0.2 | deposits for **all**, med ΔR | <0.1 |
|---|---|---|---|---|---|---|---|
| prompt (llp_frac < 0.1) | 1346 | **65.7%** | **0.0015** | **95.8%** | 99.3% | 0.0588 | 75.0% |
| mixed (0.1–0.9) | 1218 | 35.2% | 0.0658 | 57.7% | 72.7% | 0.1168 | 44.8% |
| LLP (llp_frac ≥ 0.9) | 4067 | **9.5%** | **0.2101** | **31.3%** | 48.5% | 0.2130 | 27.6% |

The result is monotonic in **track coverage** and in nothing else:

```
pT fraction carrying a track   65.7%  ->  35.2%  ->   9.5%
jet axis within dR < 0.1       95.8%  ->  57.7%  ->  31.3%
```

- **Prompt jets are accurate because two thirds of their pT carries a track** (exact direction), and
  their remaining neutrals have deposit ≈ momentum anyway (0.004, §6.1). Both components are right,
  so the axis is right — median ΔR = 0.0015. Bending never enters, because deposit positions are
  barely used.
- **The right-hand block is the hypothetical where the tracks are discarded.** Prompt jets fall to
  75.0%, so bending *would* cost something — but only mildly, and it is not what makes prompt jets
  match. It is a correction to a situation that does not arise.
- **LLP jets barely move between the two blocks** (31.3% → 27.6%): at 9.5% coverage they are already
  in the trackless regime, so removing the remaining tracks changes almost nothing. This is the same
  statement as §7.1 — past 200 mm there is nothing left to lose.

The realistic LLP figure of **31.3%** is a truth-level calculation with perfect energy and perfect
clustering, and it already sits close to the ~36% integrated efficiency measured on reconstructed
jets in §2. That agreement is the strongest single indication that the ceiling is set by geometry
upstream of the network, not by the network.

> This supersedes an intermediate azimuthal-only measurement of the jet-level shift (prompt 0.046,
> LLP 0.088). That quantity is real but assumed deposit positions for *every* constituent, which is
> the wrong hypothetical for prompt jets, and two medians in one coordinate cannot support a claim
> about pass rates in any case.


### 6.5 Is HGPflow the limitation, or are the inputs? — the decisive test

Take LLP jets **only** and bin them by displacement, building the axis as in §6.4. The ordering
variable must be **3D production distance**, not `Lxy`: COCOA's track availability depends on both
coordinates (§7.1), and this sample is strongly forward, so a transverse-only binning mixes
well-tracked central decays with untracked forward ones.

| 3D distance [mm] | N | pT frac with a track | median ΔR | axis <0.1 | <0.2 |
|---|---|---|---|---|---|
| 10–50 | 150 | **69.4%** | **0.0045** | **96.7%** | 100.0% |
| 50–150 | 597 | 49.2% | 0.0195 | **93.3%** | 99.3% |
| 150–300 | 829 | 22.5% | 0.0684 | 68.3% | 95.7% |
| 300–600 | 1309 | 7.7% | 0.1414 | 33.6% | 72.8% |
| 600–1200 | 1790 | 1.6% | 0.2576 | 10.7% | 35.0% |
| >1200 | 3524 | 0.8% | 0.4753 | 19.2% | 25.4% |

**Jets whose decay is 10–50 mm from the origin reconstruct at 96.7% — statistically
indistinguishable from prompt jets (95.8%, §6.4) — and the 50–150 mm bin is still 93.3%.** These
are genuinely displaced decays. Displacement *per se* does not degrade the reconstruction at all.
Accuracy collapses in lockstep with track coverage, 69.4% → ~1%, and with nothing else.

> **This supersedes an `Lxy`-binned version** which reported 86.4% in its leading bin. That binning
> diluted the result by mixing forward decays — large 3D flight distance, small transverse radius,
> and no tracks — into the low-`Lxy` bins. The corrected result is *stronger*: on the right axis,
> tracked displaced jets are as good as prompt jets, not merely close.

**So HGPflow is not the bottleneck; the track collection is.** The algorithm handles displaced jets
well when the directional information is present, and cannot invent it when it is not — as no
particle-flow algorithm consuming these inputs could. That is a materially different conclusion
from "HGPflow reconstructs displaced jets poorly", and it is the one the data supports.

90.9% of LLP jets sit above 150 mm in 3D distance, where coverage is ≤22%. That is the population
any tracking extension would have to reach, and the 96.7% / 93.3% rows are the quantified estimate
of what reaching it buys.

> The `<0.1` fraction ticks up in the last two rows while the median keeps degrading (0.29 → 0.41 →
> 0.43). That is not performance recovering: §10 quantifies the main cause — the >1200 mm bin
> contains the region-C degenerate-extrapolation artifact (~8% of that bin at ΔR ≈ 0, matching the
> uptick), plus calorimeter decays whose axis has collimated onto the flight direction.

#### This dissolves the "strong resolution, low efficiency" tension

The two metrics measure **different populations**, not conflicting things:

- The **IQR of 0.194** (§1) is computed on *matched* jets. §3.2 shows matched LLP jets retain a real
  charged component (⟨charged⟩ = 0.35), i.e. they are predominantly the well-tracked
  small-displacement population — the 96.7% / 93.3% rows above.
- The **efficiency** is computed over *all* LLP truth jets, including the untracked bulk.

So "excellent resolution but low efficiency" is one fact seen twice: the tracked subset reconstructs
excellently, the untracked subset not at all, and the resolution metric only ever sees the
survivors. This is the survivor bias `llp_helper.py` warns about in its docstring, and it is why the
**efficiency is the more honest headline of the two** — a resolution quoted on a 30%-selected subset
is not a statement about displaced-jet reconstruction in general.

---

## 7. Can it be fixed? Both routes measured

### 7.1 Displaced-vertex-aware tracking — there are no track objects to work with

**First, where the decays are.** From the truth decay graph (A⁰ = pdg 36), classified in full 3D —
tracker volume = ρ < 1496.5 mm **and** |z| < 3220 mm:

| | inside tracker | in calorimeter | beyond calorimeter | median Lxy | median \|z\| |
|---|---|---|---|---|---|
| A⁰ decays (all), N=7713 | 76.7% | **15.2%** | **8.1%** | 291 mm | 928 mm |
| LLP jets analysed, N=7384 | 84.8% | **11.2%** | 4.0% | 319 mm | 802 mm |

A⁰ 3D flight distance: median 1103 mm, 90% 6307 mm. Of the decays leaving the tracker, **89.2% exit
through the endcap** — this is a forward sample, which is why a transverse-only classification
misleads.

So the population splits three ways, and the three parts have genuinely different prospects: the
~77% decaying inside the tracker volume are trackable in principle, the ~15% decaying inside the
calorimeter are not trackable by anything, and the ~8% decaying beyond it are lost entirely.

**Second, track availability** (2k val events, charged particles, pT > 0.5 GeV):

| Lxy [mm] | N charged | track entry exists | in acceptance | reconstructed | linked to particle |
|---|---|---|---|---|---|
| 0–1 | 33683 | 99.9% | 99.9% | 99.9% | 99.9% |
| 1–10 | 1745 | 98.5% | 98.5% | 98.5% | 98.5% |
| 10–50 | 3473 | 88.5% | 88.5% | 88.5% | 88.5% |
| 50–200 | 9121 | 16.0% | 16.0% | 16.0% | 16.0% |
| 200–1000 | **16141** | **0.0%** | 0.0% | 0.0% | 0.0% |
| >1000 | 2738 | **0.0%** | 0.0% | 0.0% | 0.0% |

All four columns are identical — this is not an efficiency or acceptance loss that could be loosened.
COCOA's parametric tracker never *creates* a track object past ~200 mm: no entry in
`track_parent_idx`, no hits, nothing for a large-radius algorithm to run on. The modal displaced
population (16k of ~31k charged LLP particles) sits exactly where availability is identically zero.

**The cutoff is not on `Lxy` alone.** Linked-track fraction in 2D (charged, pT > 0.5 GeV):

| Lxy \\ \|z\| | 0–200 | 200–600 | 600–1500 | 1500–3220 |
|---|---|---|---|---|
| 0–10 | 100% | 82% | — | — |
| 10–50 | 100% | 79% | **0%** | **0%** |
| 50–200 | 31% | 15% | 0% | 0% |
| 200+ | 0% | 0% | 0% | 0% |

A particle at `Lxy` = 30 mm with |z| = 800 mm has **zero** track availability, while one at
`Lxy` = 30 mm, |z| = 100 mm has 100%. Binned by **3D distance** the availability is clean and
monotonic — 100%, 100%, 60.6%, 12.6%, 0%, 0% across 0–10, 10–50, 50–200, 200–600, 600–1500,
>1500 mm — which is why 3D distance is the correct ordering variable throughout §6.5.

**Verdict: not implementable on these samples — and for most, though not all, of the population this
is a simulation limit rather than a physics one.** About 77% of A⁰ decays are inside the tracker
volume, i.e. in a region a real detector instruments and where large-radius tracking is designed to
work. The remaining ~23% decay in or past the calorimeter and are beyond any tracking fix.

Recovering the trackable ~77% requires regenerating all 295k events with a modified COCOA tracking
configuration, then re-splitting, re-deriving input scales, and retraining both stages. That is
weeks of work against a physically sensible target — not an attempt to beat an information limit
(contrast §7.3, which is blocked on information and would not be rescued by regeneration). One
honesty note on "real experiments already do this": *demonstrated* large-radius tracking (e.g.
ATLAS LRT) reconstructs tracks from production radii out to roughly **300 mm** — the silicon
volume — not the full 1.4 m to the calorimeter face. Decays at 3D ≤ 300 mm (~19% of LLP jets) are
the directly precedented band; the 300–1500 mm range would need outer-layer / standalone-style
tracking that real detectors achieve only with degraded resolution. The resimulation payoff should
therefore be quoted as a **function of tracking reach**, not a single number (§8, point 6).

### 7.2 Displacement-aware un-projection — works, but needs a 25 mm vertex

For a straight-line particle, knowing the vertex **v** and the deposit position **x** gives the
momentum direction exactly as `m̂ = (x − v)/|x − v|` — no regression, closed form. Oracle test on
6128 LLP truth jets (llp_frac ≥ 0.9, anti-kT R=0.4, pT > 8 GeV), same constituent set for both jets
so that only *position* varies:

| jet axis built from | median ΔR to truth axis | frac < 0.1 | frac < 0.2 |
|---|---|---|---|
| deposit positions (what reco sees) | 0.208 | **28.0%** | 48.6% |
| un-projected, exact per-particle vertex | 0.074 | **59.9%** | 74.5% |
| un-projected, ONE jet vertex (perfect) | 0.076 | **58.9%** | 74.2% |
| ONE jet vertex, smeared 10 mm | 0.080 | 57.7% | 73.9% |
| ONE jet vertex, smeared 25 mm | 0.092 | 53.1% | 72.3% |
| ONE jet vertex, smeared 50 mm | 0.118 | 43.0% | 67.7% |
| ONE jet vertex, smeared 100 mm | 0.173 | 25.4% | 56.0% |
| ONE jet vertex, smeared 200 mm | 0.293 | 10.2% | 31.9% |

Three readings:

- **One vertex per jet is enough.** Exact per-particle vertices give 59.9%, a single shared jet vertex
  58.9%. The shared-vertex assumption costs one percentage point.
- **Bending sets the ceiling.** Re-running with straight-line propagation for everything (bending
  absent) recovers **94.9%**. With real bending it is 58.9%. The missing ~40% is irreducible magnetic
  deflection of the charged constituents, unrecoverable without tracks.
- **The vertex must be known to ≲25 mm.** At 50 mm the gain is halved; at 100 mm the result is no
  better than doing nothing.

### 7.3 Calorimeter vertex resolution — short by a factor of ~70

The depth lever arm exists: **82.5%** of displaced particles produce ≥2 topoclusters, with a median
radial span between barycentres of **619 mm** (median cell-level radial extent 747 mm). But the axes
those clusters define are poor:

| line source | N | impact param to true vertex | angle error vs momentum |
|---|---|---|---|
| cells (all) | 7672 | 632 mm | 24.5° |
| cells (neutral) | 2766 | 314 mm | 12.4° |
| cells (charged) | 4906 | 857 mm | 33.1° |
| **topo barycentres (all)** | 6443 | **1072 mm** | **40.8°** |
| topo barycentres (neutral) | 2204 | 1046 mm | 40.2° |

A single shower axis is good to ~12° at absolute best (neutral, cell-level) and ~41° from the
topocluster barycentres the network actually sees — *worse* than cells, because 2–3 barycentres are
dominated by lateral shower fluctuation. Fitting a common vertex from these lines:

| source | N | median \|Δv\| (3D) | median ΔLxy |
|---|---|---|---|
| topo barycentres | 1927 | **1748 mm** | 758 mm |
| cells (upper bound) | 2219 | 1542 mm | 662 mm |
| topo, neutrals only | 777 | 1750 mm | 799 mm |

More constituents barely help (2 lines → 1826 mm; 8+ lines → 1621 mm) because the lines are nearly
parallel and the fit is ill-conditioned along the flight direction.

**Required ≤25 mm; achieved ~1700 mm — short by a factor of ~70.** Calorimeter-only displaced vertex
finding at COCOA granularity cannot support the un-projection. This is an information limit, not an
effort limit.

---

## 8. Revised conclusions and recommendation

1. **The §1 result stands and is the real deliverable.** In-domain training buys ~28–42% tighter
   jet-pT resolution. That is a genuine, usable improvement.
2. **On the *existing samples*, the ceiling is not fixable by reconstruction.** Both routes named in
   the original §4 have been measured and both are blocked *here*: no track objects exist past
   200 mm (§7.1), and the vertex needed for un-projection is ~70× beyond calorimeter reach (§7.3).
   This is a statement about these inputs, not about the algorithm or about displaced jets in
   general — see point 5, which is the constructive counterpart.
3. **Do not build the calorimeter-pointing / vertex-finding head.** §7.3 is the gate and it fails,
   by a factor of ~70. Unlike the tracking route, this one is blocked on *information*, so
   regenerating the samples would not rescue it either.
4. **Split the efficiency by decay region first — it costs nothing and makes everything else
   tractable** (§10). The single number averages three populations whose remedies and ceilings
   differ completely; ~4% of LLP jets (region C) can never be matched at all, and reporting them
   inside one average obscures both the real ceiling and the real progress.
   Then, for calorimeter decays specifically, **score against the flight direction rather than the
   momentum axis** (§10.1). Measured on real model output (§11): **0.9% → 26.9%**, a ~30× gain on
   11.5% of LLP jets, worth +3 points on the whole LLP sample — with no retraining, and it measures
   the quantity an LLP analysis actually wants. This is implemented and ready:
   `llp_helper.tag_truth_jets_decay_region` + `llp_efficiency_by_region`.
5. **Do change the matching metric more generally — this is now the single highest-value action
   in this document (§14).** Measured: a reference-free energy-overlap match takes all-LLP
   efficiency from **24.5% → ~66%**, and tracker-region jets from **25.3% → ~74%**. That is
   +41 points, larger than the entire tracking-extension payoff (§12: +27.3), at zero cost and no
   retraining. Robust to the association radius: even at the most conservative setting the metric
   change (57.1%) still exceeds the tracking extension's optimistic ceiling (51.8%) — §14.4. The metric is calibrated against prompt jets, where overlap>0.5 is slightly
   *stricter* than ΔR<0.1, so the gain is real. Implemented in `llp_studies/overlap_match.py`.

   Scoring a calorimeter-built object against a momentum axis defined by an unobservable vertex
   compares incommensurable quantities. One further option remains unimplemented: a **truth
   calo-axis match**, clustering the truth reference jet from
   `particle_eta_extrap_calo` / `particle_phi_extrap_calo`. The reader already loads those branches
   index-aligned through the fiducial cut (added for §14's overlap match); what is left is a
   `truth_axis='calo'` option in `compute_jets`. Likely largely redundant with the overlap match,
   so low priority.
6. **HGPflow itself is sound on displaced jets — and §14 shows the *energy* use case already
   works today (74% on tracker-region jets). The tracking extension is now justified only for the
   *directional* use case.** §6.5 is the key evidence: LLP jets whose decay is
   10–50 mm from the origin reconstruct at **96.7%** axis accuracy — indistinguishable from the
   95.8% of prompt jets — because 69.4% of their pT still carries a track, and the 50–150 mm bin is
   still 93.3%. Displacement does not break the algorithm; absent tracks do. 90.9% of LLP jets sit
   above 150 mm in 3D distance where coverage is ≤22%, and that is the population a tracking
   extension would recover.

   **The target is well-defined but partial.** 76.7% of A⁰ decays are inside the tracker volume
   (§7.1), i.e. where real large-radius tracking already operates — extending COCOA's tracking there
   is ordinary work, not an attempt at the impossible. But **~15% decay inside the calorimeter and
   ~8% beyond it**, and no tracking extension touches those. For the calorimeter-decay component the
   field's own answer applies instead — CalRatio (E_HCAL/E_ECAL), jet timing, and longitudinal
   shower profile, which treat tracklessness as the *signature* rather than a defect. HGPflow already
   ingests `topo_ecal_e` / `topo_hcal_e` (as `em_frac`), so that information is in its inputs.

   That makes resimulation a *quantified* proposal rather than a speculative one: modify the COCOA
   tracking parametrisation to reconstruct tracks from displaced production vertices, regenerate
   250k/15k/30k, re-split, re-derive scales, retrain both stages. The physical headroom is real —
   the calorimeter face is at ~1496 mm, so roughly 1.3 m of tracking volume exists, and COCOA's
   sharp cutoff at ~200 mm production distance is a property of its parametric tracking model, not
   an inherent limit of that volume. **But be honest about precedent** (§7.1): demonstrated LRT in
   real experiments reaches production radii of roughly 300 mm, which covers only ~19% of these LLP
   jets; the full-volume extension goes beyond demonstrated real-detector performance. The payoff
   curve (§12) quantifies exactly this, and any proposal should quote its conservative (300 mm) and
   optimistic (full-volume) scenarios rather than one number. Regeneration is also the *only*
   route: the raw file stores fitted track
   parameters plus extrapolations to the six calorimeter layers (ρ 1547 → 3825 mm) and **no tracker
   hits at all**, so pattern recognition cannot be rerun offline.

   **Expected payoff — measured, and lower than first estimated (§12).** The tracking-reach curve
   gives: LRT-precedented 300 mm reach **+2.8 points** (24.5% → 27.3%); full tracker volume
   **+27.3 points** (24.5% → 51.8%). The payoff is concentrated entirely in the 300–1500 mm range
   that real experiments have not demonstrated. An earlier version of this point quoted "toward
   93–97%", taken from §6.5's favourable 10–50 mm bin; averaged over all tracker-region jets the
   optimistic endpoint is ~57.5%, because track coverage caps at the charged fraction (~68%, §5)
   and full-volume tracking reaches only 52%. The experiment is still the one that would answer
   "can HGPflow reliably reconstruct displaced jets?", but it should be proposed with the ~52%
   number, not the ~85% one.
7. **A separate, smaller question**: whether HGPflow should be able to *label* displaced particles as
   charged (§5). The prior is favourable — at Lxy > 200 mm, 69/(69+11) ≈ 86% of non-photon energy is
   charged — but a trackless π± shower and a K_L shower are both trackless hadronic showers, so
   per-object discrimination is weak and the prior does most of the work. It affects calibration and
   charged-multiplicity observables, not jet energy.

---

## 9. What HGPflow is, and is not, appropriate for

The natural challenge to §8 is: *if tracks are what matter and LLP jets have none, is particle flow
simply the wrong tool here? Shouldn't one fall back to calorimeter jets, which never depended on
tracks?*

### 9.1 The track-free alternative is worse, not better

Truth jets matched against COCOA's calorimeter-only `topo_jet_*` collection — topoclusters
clustered with anti-kT, no tracks anywhere in the chain (ΔR < 0.1):

| truth jet type | topo jet pT cut | N | efficiency |
|---|---|---|---|
| prompt | >5 GeV | 1986 | 54.3% |
| LLP | >5 GeV | 7384 | **17.7%** |
| prompt | >10 GeV | 1986 | 37.4% |
| LLP | >10 GeV | 7384 | **14.3%** |

And the collapse with decay distance has the same shape (calo jets, pT > 10 GeV):

| 3D decay distance [mm] | N | efficiency |
|---|---|---|
| 10–150 | 686 | 44.8% |
| 150–300 | 750 | 33.5% |
| 300–600 | 1181 | 24.2% |
| 600–1200 | 1587 | 8.8% |
| >1200 | 3180 | **2.3%** |

**The geometric failure is not particle-flow-specific.** It belongs to *any* reconstruction that
takes direction from calorimeter deposit positions, because those positions are interpreted as
pointing back to the origin. Dropping to calorimeter jets to avoid the track dependence makes
displaced-jet reconstruction worse, not better — and §1 already showed the parametric PPflow
baseline is worse than HGPflow on LLP jets too (IQR 0.366 vs 0.194; true-jet efficiency 35.9% vs
37.1%).

> Two caveats on this comparison: `topo_jet_*` is **uncalibrated**, so a pT cut bites harder on it
> than on a calibrated collection (hence both thresholds are shown); and this is a full matching
> efficiency including clustering and threshold effects, whereas §6.5 isolates position only. Do
> not read 14.3% against §6.5's numbers as a clean ratio — the *shape* and the *direction* of the
> comparison are what is robust.

### 9.2 So: appropriate for what?

"Is HGPflow inappropriate for displaced jets?" would generalise to "all calorimeter-based
reconstruction is inappropriate for displaced jets" — true, but not a statement about HGPflow. The
defensible version is narrower:

> HGPflow cannot recover the **momentum direction** or the **charge structure** of jets from decays
> beyond the tracked region, and no calorimeter-based method can. It does recover their **energy** —
> ~85% capture (§3.2), with the best jet-pT resolution of the three collections measured (§1).
>
> **Region restriction added after §13.** The energy claim holds for decays inside the *tracker
> volume* (unmatched jets there recover 0.92 at R<0.8). For unmatched **calorimeter-region** decays
> the energy is largely *missing*, not displaced — 0.36 at R<0.8 — most likely from late shower
> development. Do not extend the energy claim to region B.

Which makes appropriateness **observable-dependent**:

| use case | verdict |
|---|---|
| **Displaced-jet tagging** (low track multiplicity, calorimeter ratio, shower shape) | Fine, arguably advantaged — needs energy and shower structure, not a precise axis. HGPflow already ingests `topo_ecal_e`/`topo_hcal_e` as `em_frac`, so CalRatio-type information is in its inputs |
| **Resonance mass from the jet four-vector** | Degraded, and *biased*: the `(1 − L/R)` collimation (§6.2) shrinks the reconstructed opening angle, so mass comes out systematically **low**. Affects every calorimeter-based method equally |
| **Matching a jet to a displaced vertex found elsewhere** (muon spectrometer, timing, LRT) | Degraded — needs the axis |
| **MET** | Moderately affected; directional errors partially cancel in the vector sum |

Also worth separating: a matching efficiency is an **evaluation metric, not an analysis
efficiency**. A real LLP search never matches reco jets to truth momentum axes — it finds jet-like
energy deposits and computes observables on them. HGPflow does find that energy.

### 9.3 Two validation gaps

> **Both now measured — see §13.** Gap 1 is closed with a split verdict: energy *is* recovered for
> unmatched tracker-region jets (0.92 at R<0.8) and is *not* for unmatched calorimeter-region ones
> (0.36). Gap 2 is bounded: the calibration bias is ≲6% on matched jets across all regions.

Neither is a defect, but both bound what the energy claims above can support:

1. **The energy response on *unmatched* LLP jets has never been measured at jet level.** By
   construction there is no matched jet to form a residual against. §3.2's cone study (85% within
   ΔR<0.8) is supporting evidence, not a response measurement. Anyone proposing to use HGPflow for
   displaced-jet *energy* should close this first — the §3.2 cone machinery already does most of it.
2. **The neutral relabelling may carry a calibration bias.** These objects are charged pion showers
   being reconstructed and calibrated as neutral hadrons (§5). Whether that induces a systematic
   energy shift in the untracked regime is unmeasured.

Neither is a reason to reject HGPflow for displaced jets. Both are reasons not to claim its
displaced-jet *energy* performance is established.

---

## 10. Splitting the problem by decay region

The single ~0.44 efficiency averages three populations with different physics, different remedies,
and different achievable ceilings. That averaging is part of why the number has been hard to act on.
Scoring each region against the reference that is physically appropriate for it:

| LLP jets by decay region | N | vs **momentum** axis | | | vs **flight direction** | |
|---|---|---|---|---|---|---|
| | | median | <0.1 | <0.2 | median | <0.1 |
| decay in tracker volume | 6263 | 0.2004 | 29.5% | 49.9% | 0.7195 | 5.2% |
| **decay in calorimeter** | 827 | 0.7379 | 28.4% | 31.2% | **0.1230** | **47.5%** |
| decay beyond calorimeter | 294 | *0.0000* | *97.6%* | *99.3%* | 1.4725 | 0.0% |

> **The bottom row is an artifact, not a result.** Particles produced past the calorimeter are never
> propagated, so their stored `particle_*_extrap_calo` degenerates to the momentum direction and this
> truth-level idealisation returns ΔR = 0. In reality they deposit nothing and form no reco jet.
> Read region C as *permanently lost*, never as *reconstructed perfectly*.

### 10.1 Calorimeter decays measure the flight direction, not the momentum axis

For a decay inside the calorimeter, `L/R → 1`, so the `(1 − L/R)` collimation of §6.2 drives every
deposit onto the A⁰'s **line of flight**. The measurement confirms it: the reconstructed axis sits a
median ΔR of **0.123** from the flight direction versus **0.738** from the decay products' momentum
axis — a factor of six.

**The reconstruction is not failing for these jets. It is measuring a different quantity**, and
arguably the more useful one: the flight direction points back at the decay vertex, which is what an
LLP analysis wants for vertex association. Scored against it, efficiency goes **28.4% → 47.5% with no
algorithm change and no retraining** — purely a change of reference.

Note the inversion: for tracker-volume decays the momentum axis is clearly the right reference
(29.5% vs 5.2%), and for calorimeter decays it is clearly the wrong one. A single global reference
cannot serve both.

### 10.2 The three populations and their remedies

| | share of A⁰ / of analysed jets | remedy | ceiling |
|---|---|---|---|
| **A.** decay in tracker volume | 77% / 85% | **extend tracking** — §6.5 gives 93–97% | high |
| **B.** decay in calorimeter | 15% / 11% | **change the reference** to flight direction (§10.1); CalRatio / timing / shower profile for tagging | moderate |
| **C.** decay beyond calorimeter | 8% / 4% | none — no energy deposited | **zero** |

Region C sets a hard floor: ~4% of the current LLP jet sample can never be matched by any
reconstruction, however good.

Taking A to ~95% with tracking, B to ~50% with the right reference, and C to 0 gives a weighted
ceiling of roughly **85%**, against ~31% today.

> **Superseded by §12.3.** Both inputs to that arithmetic were too optimistic. The ~95% for region
> A came from a favourable low-displacement subpopulation; the payoff curve shows full-volume
> tracking reaches ~57.5%. The measured ceiling is **~52%**, against 24.5% today.

> **§11 measures all of this on real model output.** The region split, the dead region C and the
> reference inversion all survive; the calorimeter-decay gain turns out to be **~30× rather than
> 1.7×** (0.9% → 26.9%), because the truth-level numbers in this section carry a momentum-space
> clustering bias. Read §10 for the mechanism and §11 for the magnitudes.

### 10.3 The cτ arithmetic: why "most decays in the calorimeter" was never reachable

The sample **was** generated with a long lifetime — extracted from the truth graph,
`ℓ = d₃D/βγ` per A⁰ is a clean exponential (mean/median = 1.450 vs the exponential's 1.443) with

| | |
|---|---|
| **generated cτ** | **1.00 m** |
| true `m_a` | 55 GeV (fixed) |
| A⁰ βγ | median **1.57**, [25, 75] = 0.82, 3.34 |
| lab flight distance | median 1103 mm, mean 2309 mm |

Two effects then decide where the decays land:

1. **The A⁰ is heavy and slow.** From a 125 GeV Higgs, a 55 GeV `a` carries only ~30 GeV in the
   H frame, so βγ ~ 1.6 — the lab decay length `βγ·cτ` is compressed relative to cτ.
2. **The exponential's mode is at zero.** The decay-position density is monotonically *decreasing*:
   63% of decays always come before the mean, no matter where the mean is put.

Propagating cτ = 1 m through each A⁰'s true direction and boost predicts tracker/calo/beyond =
**76.7% / 14.3% / 9.1%**, against the measured **76.7% / 15.2% / 8.1%** — full closure: geometry
plus one exponential lifetime explains the sample completely.

Scanning cτ with this boost spectrum: the calorimeter fraction is **maximised at cτ ≈ 3.6 m,
reaching only ~24%**. No lifetime choice makes calorimeter decays the majority — the exponential
always front-loads the tracker volume, and pushing cτ up mainly feeds the beyond-calorimeter
(invisible) tail. The current sample, at 15.2%, is already ~⅔ of the theoretical maximum.

**Consequences for sample design.** If calorimeter-decay LLP jets are the physics target: (a)
raising cτ toward ~3.6 m buys at most 15% → 24%; (b) a *lighter* `a` boosts harder (longer lab
lengths per cτ, and more collimated decays — a different phenomenology); (c) the efficient route
is truth-level selection — generate more events and select the calorimeter-decay subset, since no
generation parameter can concentrate the exponential there.

---

## 11. The bridge measurement: region-split efficiency on real model output

Everything in §6.4–§10 is truth-level idealisation. This section is the first decomposition of the
**actual** §2 efficiency, run through the real evaluation pipeline (`PerformanceCOCOA` on the
hss-trained model's merged predictions) over the full 30k test set — 29,976 common events, 84,424
truth jets. Script: `llp_studies/region_efficiency_reco.py`.

**HGPflow (hss-trained), ΔR < 0.1:**

| region | N | eff (momentum axis) | eff (flight direction) |
|---|---|---|---|
| prompt (llp_frac < 0.1) | 14923 | **92.4%** | — |
| mixed (0.1–0.9) | 14383 | 52.0% | 21.9% |
| tracker | 46659 | **25.3%** | 4.2% |
| **calo** | 6359 | **0.9%** | **26.9%** |
| beyond | 2100 | 0.1% | 0.1% |

**At ΔR < 0.2:** prompt 96.4%; mixed 64.2% / 36.1%; tracker 43.5% / 11.5%; calo 2.4% / **33.2%**;
beyond 0.3% / 0.3%.
**PPflow, ΔR < 0.1:** prompt 91.1%; mixed 50.7% / 20.8%; tracker 24.5% / 3.9%; calo 0.8% /
**22.8%**; beyond 0.1% / 0.1% — HGPflow is ahead in every region, by the widest relative margin on
calorimeter decays (26.9% vs 22.8%).

Note the mixed row: at 52.0% (momentum) vs 21.9% (flight) it still prefers the momentum axis, as
expected for jets whose energy is majority-prompt.

> **Superseding an earlier bucketing error.** The first version of `llp_efficiency_by_region`
> lumped *everything* with `llp_frac < 0.9` into one "prompt" row, absorbing the partially
> displaced mixed population and reporting a blended **72.6%**. Only 18,253 of 99,291 truth jets
> have no LLP constituent at all, so roughly half that row was mixed. The corrected split
> reproduces the blend exactly (14923×92.4% + 14383×52.0% = 72.6%), confirming the fix, and the
> helper now takes `prompt_frac` (default 0.1, matching the §6.4 convention). The LLP rows were
> never affected — those are cleanly `llp_frac ≥ 0.9`.

### 11.1 What the real data confirms

1. **The decay-region tagging reproduces the truth-level study exactly.** Region shares of the LLP
   sample: 84.7% / 11.5% / 3.8% here, against 84.8% / 11.2% / 4.0% from §7.1. The classification
   transfers intact to real reconstruction.
2. **Region C is genuinely dead: 0.1% on *both* references.** §10's artifact warning was correct —
   the truth-level 97.6% was a degenerate-extrapolation artifact, and real reconstruction confirms
   these jets deposit nothing and are permanently unmatchable.
3. **A global cross-check passes.** Weighting all five rows by population (prompt 92.4%, mixed
   52.0%, LLP 21.5% on the momentum axis) gives **~39%** overall, against the independently
   measured **37.1%** true-jet efficiency in §1. The decomposition adds up to the number it
   decomposes.
4. **The flight-direction effect is real**, and the reference inversion holds: tracker-region jets
   score far better on the momentum axis (25.3% vs 4.2%), calorimeter-region jets far better on the
   flight direction (26.9% vs 0.9%). No single reference serves both.

### 11.2 What the real data revises — and a methodological lesson

**The calorimeter-decay gain is ~30×, not the 1.7× §10 predicted** (0.9% → 26.9%, versus the
truth-level 28.4% → 47.5%). The direction was right; the magnitude was badly off, and the reason is
instructive.

Truth-level (§10) over-predicted the *momentum-axis* number for calorimeter decays by a factor of
~30. The cause is a **momentum-space clustering bias built into the truth-level construction**: it
clustered truth particles by their *momenta* to define each jet, then repositioned those same
constituents at their deposit points. The constituent set was therefore momentum-defined, leaving
residual correlation with the momentum axis. Real reconstruction clusters in *deposit* space and
inherits no such correlation — so its jets land near the flight direction and essentially never
within ΔR<0.1 of the momentum axis.

Note this is not "reco failed to make a jet": 26.9% of calorimeter-decay jets *are* found, just at
the flight direction. The energy is reconstructed into a findable jet; the momentum axis is simply
the wrong place to look for it.

**Truth-level over-predicts absolute efficiency everywhere**, now quantified:

| | truth-level (§6.4, §10) | real (§11) | transfer |
|---|---|---|---|
| prompt (llp_frac < 0.1) | 95.8% | **92.4%** | −3.4 pts — excellent |
| mixed (0.1–0.9) | 57.7% | **52.0%** | −5.7 pts — good |
| tracker (momentum) | ~31% | 25.3% | −6 pts — good |
| **calo (momentum)** | 28.4% | **0.9%** | **−27.5 pts — collapse** |
| **calo (flight)** | 47.5% | **26.9%** | −20.6 pts — poor |

**The pattern is sharp, and it is exactly what the clustering-bias explanation predicts.** Prompt,
mixed and tracker jets transfer within a few points; only the calorimeter rows collapse. The bias
can only bite where the deposit position and the momentum axis diverge — which is negligible for
prompt jets (deposit ≈ momentum, §6.1), modest for tracker-region decays, and total for
calorimeter decays where `L/R → 1`. So the truth-level machinery in §6.4–§10 is **quantitatively
reliable except in region B**, rather than uniformly optimistic as an earlier version of this
section claimed.

This refines the "upper bound on axis recovery, not a predicted efficiency" caveat attached to
§6.4 and §7.2: treat those numbers as good estimates for prompt/mixed/tracker jets, and as upper
bounds only for calorimeter-region ones.

### 11.3 Net effect of the reference change

Applying the region-appropriate reference to every jet, versus the momentum axis for all:

| | momentum axis everywhere | region-appropriate reference |
|---|---|---|
| ΔR < 0.1 | 21.5% | **24.5%** |
| ΔR < 0.2 | 37.1% | **40.7%** |

About **+3 percentage points** across the whole LLP sample (+14% relative) — modest globally,
because calorimeter decays are only 11.5% of it, but on that subset it is the difference between
0.9% (invisible) and 26.9% (usable). It costs nothing: no retraining, no algorithm change, no
resimulation.

---

## 12. The tracking-reach payoff curve

§8 recommends regenerating the samples with extended tracking. This puts a number on that, as a
function of **how far the tracking reaches** — the question §7.1's realism caveat left open.

Method: cluster truth jets once (momentum space, unaffected by the assumption), then sweep an
assumed reach X. A constituent is treated as tracked if it is charged, within |η| < 2.5, and its 3D
production distance ≤ X. Tracked constituents contribute their momentum direction; untracked ones
their deposit position. Script: `llp_studies/tracking_reach_payoff.py` (7384 LLP jets).

Regions B and C are **excluded from the curve** and held at their §11 *measured* values — no
tracker reaches them, and their truth-level numbers are known-biased (§11.2). The curve is
therefore tracker-region jets only, the only population extended tracking can help.

| assumed reach | pT with a track | tracker <0.1 (truth) | tracker (real) | **all-LLP (real)** | gain |
|---|---|---|---|---|---|
| actual (COCOA) | 10.7% | 29.5% | 25.3% | **24.5%** | — |
| 200 mm | 12.0% | 29.3% | 25.1% | 24.3% | −0.2 |
| **300 mm** (LRT-precedented) | 17.5% | 32.8% | 28.6% | **27.3%** | **+2.8** |
| 500 mm | 26.0% | 39.8% | 35.6% | 33.2% | +8.7 |
| 750 mm | 34.6% | 47.6% | 43.4% | 39.8% | +15.3 |
| 1000 mm | 41.6% | 53.6% | 49.4% | 44.9% | +20.4 |
| **1500 mm** (full tracker volume) | 52.0% | 61.7% | 57.5% | **51.8%** | **+27.3** |

"Real" applies a 4.2-point truth→real transfer, self-calibrated so the `actual` row reproduces
§11's measured 25.3% for tracker-region jets. Two checks: the baseline all-LLP figure of **24.5%**
independently reproduces §11.3's region-appropriate number, and the *gains* are transfer-independent
(a constant shift cancels in the difference).

### 12.1 The payoff is concentrated beyond demonstrated tracking

**The LRT-precedented 300 mm point buys +2.8 points. The full tracker volume buys +27.3.** The
curve is nearly flat out to 300 mm and steep thereafter — because (§10.3) the decay distribution is
exponential with a median 3D flight of ~1100 mm, so 300 mm captures only its low tail. Essentially
all of the payoff lives in the 300–1500 mm range, which is exactly the range real experiments have
*not* demonstrated.

This is the central input to the resimulation decision. A conservative, precedent-respecting
tracking extension is not worth the regeneration cost. The large payoff requires assuming tracking
performance beyond the state of the art, and any proposal should say so explicitly rather than
quoting the optimistic endpoint alone.

### 12.2 This corrects §8's expected payoff

§8 point 6 estimated that extending coverage "should move those jets from ~10–30% toward the
93–97% seen where tracks exist." **That is too optimistic**, and the curve shows why.

The 93–97% figure comes from §6.5's 10–50 mm bin, where track coverage is 69.4% — essentially the
**charged fraction of jet energy (~68%, §5)**, which is the hard ceiling on coverage since neutrals
never have tracks. That bin is a favourable low-displacement subpopulation, not a target the whole
tracker-region population can reach.

Averaged over *all* tracker-region jets, full-volume tracking reaches only **52.0%** coverage — some
16 points below the charged-fraction ceiling, because constituents from secondary decays (e.g.
`K_S → ππ`) are produced beyond the reach, and |η| acceptance removes more. That 52% coverage maps
to ~61.7% truth-level / ~57.5% real, **not** 93–97%.

So the honest statement of the tracking-extension payoff is: **~25% → ~52% of all LLP jets**, at
full tracker volume, under optimistic tracking assumptions — a genuine and large gain, but roughly
half what §8 implied, and it does not approach prompt-jet performance.

### 12.3 Revised ceiling

§10.2's indicative ~85% ceiling assumed region A could reach ~95%. The curve says ~57.5% is the
realistic optimistic endpoint for region A. Recomputing with the measured region shares:

| | region A | region B | region C | weighted |
|---|---|---|---|---|
| today | 25.3% | 26.9% | 0.1% | **24.5%** |
| full-volume tracking + flight reference | 57.5% | 26.9% | 0.1% | **51.8%** |

**The achievable target is ~52%, not ~85%.** §10.2's figure should be read as superseded.

---

## 13. Jet-level energy response, including unmatched jets

This closes both gaps of §9.3. The measurement is match-independent: reconstructed particle pT
inside a cone around the truth jet's **region-appropriate** reference axis (momentum axis for
tracker/beyond, flight direction for calorimeter — §10.1), over the truth jet pT. No reco jet is
required, so unmatched jets get a number. Script: `llp_studies/energy_response_by_region.py`.

Median Eratio, and the reco class composition inside the widest cone:

| region | selection | N | R<0.4 | R<0.8 | R<1.2 | ⟨ch⟩ | ⟨nh⟩ | ⟨ph⟩ |
|---|---|---|---|---|---|---|---|---|
| prompt | matched | 13788 | 1.01 | 1.19 | 1.45 | 0.61 | 0.21 | 0.17 |
| prompt | unmatched | 1135 | 0.98 | 1.48 | 2.05 | 0.54 | 0.30 | 0.17 |
| mixed | matched | 7478 | 0.95 | 1.14 | 1.39 | 0.51 | 0.31 | 0.17 |
| mixed | unmatched | 6905 | 0.59 | 1.07 | 1.50 | 0.37 | 0.46 | 0.16 |
| tracker | matched | 11801 | 0.95 | 1.08 | 1.27 | 0.36 | 0.45 | 0.19 |
| **tracker** | **unmatched** | 34858 | 0.60 | **0.92** | 1.17 | 0.14 | 0.68 | 0.19 |
| calo | matched | 1713 | 0.94 | 1.05 | 1.19 | 0.09 | 0.86 | 0.05 |
| **calo** | **unmatched** | 4646 | **0.13** | **0.36** | 0.47 | 0.19 | 0.73 | 0.07 |
| beyond | unmatched | 2097 | 0.00 | 0.05 | 0.18 | 0.47 | 0.40 | 0.14 |

> **Wide cones over-count.** The prompt row reaches 1.19 at R<0.8 and 1.45 at R<1.2 — a jet that is
> by construction well reconstructed cannot recover 145% of its own energy, so roughly 20% (R<0.8)
> and 45% (R<1.2) is contamination from neighbouring jets. Read every wide-cone number against the
> prompt row of the same cone, not against 1.0. The R<0.4 column is the least contaminated and the
> most interpretable.

### 13.1 Gap 1, region A: the energy is there — confirmed

Unmatched **tracker-region** jets recover **0.92 at R<0.8** (0.60 at R<0.4). Against the prompt
row's 1.19 at the same cone, that is ~77% of the prompt-normalised value: most of the energy is
present and simply mispositioned. This confirms §3.2's cone result on the population that matters
and closes gap 1 for the region that dominates the sample — **HGPflow does recover the energy of
unmatched displaced jets when the decay is inside the tracker volume.**

### 13.2 Gap 1, region B: the energy is *not* there — a new negative result

Unmatched **calorimeter-region** jets recover only **0.13 at R<0.4 and 0.36 at R<0.8** — even
against the contaminated prompt baseline, that is a factor of ~3 below the tracker-region rows and
nowhere near recovery. Widening to R<1.2 reaches only 0.47.

**This is qualitatively different from every other region, and it is new.** For tracker-region
jets the failure is positional; for unmatched calorimeter-region jets **the energy is genuinely
missing**, not merely displaced. The proposed mechanism is late shower development: a decay deep in
the calorimeter starts its shower with little depth remaining, so a large fraction leaks out the
back (punch-through) or falls below topocluster thresholds.

> **§17 confirms this mechanism and corrects its scope.** Measured directly from
> `particle_dep_energy`, deposited energy collapses **0.71 → 0.27 → 0.00** as the remaining
> calorimeter depth falls below ~2 m. But the effect is confined to the *deep* part of region B:
> decays near the ECAL face deposit **better** than prompt jets (0.66–0.71 vs 0.543). The
> region-averaged 0.36 quoted below is a blend, and the bimodality noted at the end of this section
> is explained by decay depth.

**§9.2 must be qualified accordingly.** Its claim that HGPflow "recovers their energy — ~85%
capture" holds for tracker-region decays and **fails for unmatched calorimeter-region ones**.
Since §9.2 argued HGPflow is appropriate for displaced-jet *energy* measurement, that argument now
carries a region restriction.

Note the contrast within region B itself: *matched* calorimeter jets recover 0.94 / 1.05, entirely
normally. So the region splits into a well-reconstructed minority (27%, §11) and a majority whose
energy is largely lost — a bimodality that a single region-averaged number (0.46 / 0.60) hides.

### 13.3 Gap 2: no large calibration bias

Comparing **matched** jets at R<0.4, where contamination is smallest and geometry is not yet a
factor:

| prompt | mixed | tracker | calo |
|---|---|---|---|
| 1.01 | 0.95 | 0.95 | 0.94 |

All within 6% of unity and within 7% of each other, despite ⟨ch⟩ falling from 0.61 (prompt) to
0.09 (calo) — i.e. despite almost the entire charged component being relabelled and calibrated as
neutral hadrons. **The neutral-relabelling calibration bias of §9.3 gap 2 is therefore bounded at
≲6% on the jet energy scale**, not a leading effect. It should still be corrected for a precision
measurement, but it does not undermine the energy claims.

The class-composition columns also reproduce §5's finding region by region on real reco output:
⟨ch⟩ = 0.61 for prompt against 0.14 (tracker, unmatched) and 0.09 (calo, matched), with ⟨nh⟩
rising to 0.86 — the trackless-charged → neutral-hadron flip, exactly as predicted.

### 13.4 Region C confirmed dead on energy as well as position

Beyond-calorimeter jets recover **0.00 / 0.05 / 0.18**. §11 showed they are unmatchable; this shows
there is nothing to match — they deposit essentially no energy. (The non-zero ⟨ch⟩ = 0.47 is the
composition of the small contamination from neighbouring jets, not of their own energy.) The hard
floor is now established on both axes.

---

## 14. Reference-free matching: the ceiling was largely a metric artifact

Every efficiency up to here depends on choosing a reference axis, and §10/§11 showed the correct
axis is region-dependent. This drops the axis entirely (§8, point 5): for each truth jet, mark the
reco particles lying within `dr_assoc` of *any* of its constituents' calorimeter **deposit**
positions, then ask which reco jet contains the largest share of that energy. Matched if the best
overlap exceeds a threshold. No axis is chosen anywhere, so one number is meaningful in every
region. Script: `llp_studies/overlap_match.py` (`dr_assoc` = 0.15).

| region | N | ovl > 0.3 | ovl > 0.5 | ovl > 0.7 | ΔR < 0.1 (region-appropriate) |
|---|---|---|---|---|---|
| prompt | 14923 | 94.4% | 88.6% | 75.3% | 92.4% |
| mixed | 14383 | 79.1% | 69.9% | 52.0% | 52.0% |
| **tracker** | 46659 | **81.7%** | **73.8%** | 53.4% | **25.3%** |
| calo | 6359 | 34.3% | 27.9% | 19.2% | 26.9% |
| beyond | 2100 | 0.9% | 0.4% | 0.3% | 0.1% |

**The metric is calibrated, not inflated.** On prompt jets — where both metrics should agree —
ΔR<0.1 gives 92.4% while overlap>0.5 gives 88.6% and overlap>0.3 gives 94.4%. So overlap>0.5 is
*slightly stricter* than ΔR<0.1 on the control population. Any gain elsewhere is therefore real,
not a loosened cut.

### 14.1 Tracker-region jets: 25.3% → 73.8%

**Roughly three quarters of tracker-region LLP jets have a reco jet containing more than half their
energy** — against one quarter passing the ΔR<0.1 test. A factor of ~3, on the population that is
84.7% of the LLP sample. §14.4 scans the association radius: the value ranges **64–80%** across
`dr_assoc` 0.1–0.25, with 73.8% the central choice. The factor-of-3 conclusion holds at every
setting.

The reading is direct: **these jets are being reconstructed.** Their energy is found, and it is
clustered together into a single reco jet. What fails is only the *pointing* — the jet is not where
an origin-projected momentum axis expects it, for exactly the geometric reasons §6 derived. The
"~0.44 ceiling" of §2, and the 25.3% of §11, are to a large degree measuring a convention rather
than a reconstruction failure.

Aggregated over all LLP jets with the measured region shares:

| metric | all-LLP efficiency |
|---|---|
| ΔR < 0.1, region-appropriate axis (§11) | **24.5%** |
| overlap > 0.5 | **65.7%** (range 57–71%, §14.4) |
| overlap > 0.3 | **73.2%** |

### 14.2 This outranks the resimulation

§12 measured the tracking-extension payoff at **+27.3 points** (24.5% → 51.8%), requiring
regeneration of 295k events, retraining, and tracking performance beyond demonstrated
large-radius capability.

**Changing the metric is worth +41 points (24.5% → 65.7%), costs nothing, and needs no
retraining.** It is available today.

This survives the robustness scan. Even at the **most conservative** association radius tested
(`dr_assoc` = 0.1), the metric change gives **57.1%** — still above the **51.8%** that the full
tracking extension reaches under optimistic assumptions (§12). The ordering does not depend on the
parameter choice. (The two figures are fractions of LLP jets counted as reconstructed under their
respective definitions — overlap and ΔR — not the same quantity measured twice; the use-case split
below is the right way to choose between them.)

These are not alternatives that add: they measure different things, and the distinction is the
practical conclusion of this entire study.

- **If the analysis needs the jet's energy — does a jet exist, and how much energy is in it —**
  HGPflow already delivers ~74% on tracker-region displaced jets with no changes at all.
- **If the analysis needs the jet's direction** — pointing back at a vertex, resonance mass from
  the four-vector, association to an externally-found displaced vertex — then the axis genuinely is
  wrong, tracking is the only fix, and §12's curve says the realistic payoff is limited and
  concentrated beyond real-detector precedent.

So §9.2's "appropriate for what?" question now has a quantitative answer, and the resimulation
decision should be made on the *directional* use case alone — its value for the energy/existence
use case is close to zero, because that case already works.

### 14.3 Where overlap does not rescue the number

**Calorimeter-region jets stay low: 34.3% at overlap>0.3, barely above their 26.9% ΔR value.** That
is not a matching failure — §13.2 showed their unmatched population physically lacks the energy
(0.36 at R<0.8), and overlap is normalised by truth jet pT, so it is *capped* by that. (§17 shows
this 34.3% is a blend over decay depth: early-calorimeter decays contain well, the deep-HCAL tail
is dead. For a depth-selected population, use §17's numbers rather than this average.) The ~34%
recovered is essentially the well-reconstructed minority §11 already identified; the rest cannot be
recovered by any metric because the energy is not there.

**Beyond-calorimeter jets stay at ~0.9%**, confirming for a third time (position §11, energy §13.4,
overlap here) that region C is permanently lost.

So the region structure survives intact — the metric change rescues region A, is capped by physics
in region B, and cannot touch region C.

### 14.4 Robustness: scanning the association radius

`dr_assoc` is a position-resolution parameter, not physics — reco particles aggregate several truth
particles, so it must be loose enough to absorb that without reaching into neighbouring jets.
Scanning it:

| `dr_assoc` | prompt (ovl>0.5) | **tracker (ovl>0.5)** | calo | beyond | all-LLP (ovl>0.5) |
|---|---|---|---|---|---|
| 0.10 | 73.2% | **64.2%** | 23.7% | 0.3% | 57.1% |
| 0.15 | 88.6% | **73.8%** | 27.9% | 0.4% | 65.7% |
| 0.25 | 96.8% | **79.8%** | 31.1% | 0.8% | 71.2% |

**The absolute values are parameter-dependent** — tracker-region efficiency spans 64–80% — so any
single number quoted from this metric must state its `dr_assoc`. Two checks establish that the
*conclusions* are not parameter-dependent:

**1. A null test bounds false-positive association.** Region C jets have essentially no energy of
their own (§13.4: Eratio 0.00–0.18), so any overlap they register is spurious. Their rate stays at
**0.6% / 0.9% / 1.2%** across the scan — negligible even at the loosest setting. So the larger
`dr_assoc` values are not inflating the result by reaching into neighbouring jets, which was the
obvious worry.

**2. Prompt-calibrating the threshold gives a consistent answer.** Choosing, at each `dr_assoc`,
the overlap threshold at which *prompt* jets reproduce their ΔR<0.1 value of 92.4%, and reading
tracker-region efficiency at that same threshold:

| `dr_assoc` | calibrated threshold | tracker-region efficiency |
|---|---|---|
| 0.10 | 0.30 | 76.9% |
| 0.15 | 0.37 | 79.0% |
| 0.25 | 0.70 † | 62.9% |

63–79% — the same band, arrived at independently of the threshold convention.
(† clamped at the scan edge: at `dr_assoc` = 0.25 prompt overlap is still 93.0% at threshold 0.7,
so the true calibrated threshold lies beyond 0.7 and the true value slightly below 62.9%.)

**What to quote.** Tracker-region displaced jets are reconstructed with most of their energy in a
single reco jet **60–80% of the time**, against 25.3% by the ΔR<0.1 convention. The uncertainty on
that band is dominated by the association-radius convention, not by statistics, and it does not
threaten either conclusion of §14.1 or §14.2.

---

## 15. The model comparison, redone on the metric that matters

§1 compared the three reconstructions on jet-pT resolution (matched jets only); §2 found all three
hit the same ΔR ceiling and concluded in-domain training "washes out at high pT". This section
re-runs the comparison — hss-trained vs qqbar-trained vs PPflow, same 30k test set — under the
region split and the overlap metric. Files: `results/overlap_{hss,qqbar}.txt`,
`results/region_efficiency_{30k,qqbar}.txt`. The PPflow rows are byte-identical between the two
runs (they come from the shared truth file), which doubles as the alignment check.

**Energy-overlap efficiency (ovl > 0.5, `dr_assoc` = 0.15):**

| region | PPflow | HGPflow (qqbar) | HGPflow (hss) |
|---|---|---|---|
| prompt | 88.9% | 86.9% | 88.6% |
| mixed | 60.8% | 63.5% | **69.9%** |
| **tracker** | 50.0% | 62.4% | **73.8%** |
| **calo** | 19.3% | **12.6%** | **27.9%** |
| beyond | 0.2% | 0.2% | 0.4% |
| **all-LLP** | **44.6%** | **54.3%** | **65.7%** |

**ΔR < 0.1, region-appropriate axis, for contrast:**

| region | PPflow | HGPflow (qqbar) | HGPflow (hss) |
|---|---|---|---|
| prompt | 91.1% | 91.4% | 92.4% |
| tracker | 24.5% | 24.0% | 25.3% |
| calo (flight) | 22.8% | 15.0% | 26.9% |

### 15.1 The ΔR metric was hiding the training gain

On the ΔR metric the three tracker-region rows are indistinguishable — 24.5 / 24.0 / 25.3 —
which is exactly §2's "all three collections plateau at the same ceiling; training washes out".
On the overlap metric the same jets read **50.0 / 62.4 / 73.8**: in-domain training is worth
**+11.4 points** over qqbar and **+23.8** over PPflow.

The resolution: the ΔR ceiling is *geometric* (§6) and saturates identically for every
reconstruction, so it is blind to differences in how well the displaced energy is actually
collected. §2's conclusion was an artifact of measuring through a saturated observable. **In-domain
training does not wash out — it buys a large gain in energy containment**, on all jets (no survivor
selection, unlike §1's resolution numbers), and the gain grows with strictness: at ovl > 0.7 the
tracker-region ladder is 21.7 / 43.9 / **53.4**.

The prompt rows (88.9 / 86.9 / 88.6) show the three reconstructions equivalent where displacement
plays no role — the differences are displaced-jet-specific, not generic.

### 15.2 Out-of-domain ML loses to the parametric baseline where it matters most

The calorimeter-region row has a sharper lesson: **the qqbar-trained model (12.6%) is worse than
PPflow (19.3%)** — and the same inversion shows in the flight-direction ΔR column (15.0% vs 22.8%).
The out-of-domain network, which never saw a trackless HCAL-dominated shower in training, handles
region B *worse than a dumb parametric algorithm*; its learned priors actively hurt on the most
exotic population. In-domain training does not merely recover the gap — it more than doubles the
qqbar number (27.9%) and beats PPflow by ~9 points.

So the domain-adaptation story is region-dependent: negligible for prompt jets, large for
tracker-region displaced jets, and sign-flipping for calorimeter decays. Anyone deploying an
ML particle-flow model on LLP signatures should expect the out-of-domain penalty to be worst
exactly where the signature is most distinctive.

### 15.3 What this changes

- **§1's conclusion upgrades.** In-domain training buys resolution (~42% tighter IQR, matched jets)
  *and* containment (+11 pts overlap, all jets). The containment gain is the more important of the
  two, because it is measured without survivor selection.
- **§2's conclusion is corrected.** "Training cannot lift the ceiling" is true of the ΔR ceiling
  only, and that ceiling is largely a metric artifact (§14). On the physical question — is the
  displaced energy collected into a jet — training moves the answer substantially.
- **The evaluation recommendation of §8/§14 gains force**: under the ΔR convention, a 24-point
  genuine improvement between reconstructions was invisible. A metric that cannot see the
  difference between PPflow and an in-domain-trained HGPflow on displaced jets is not measuring
  reconstruction quality.

---

## 16. The background side: fake rate and feature migration

Everything through §15 is signal-side. A search's sensitivity is signal efficiency *against
background rejection*, and §15's +11-point containment gain says nothing about whether in-domain
training also manufactures signal-like background. Two mechanisms could:

- **(a) spurious jets** — reconstruction invents a jet from leftover clusters. Trackless and
  neutral-heavy by construction, so it lands in a displaced-jet tagger's signal region.
- **(b) feature migration** — *real* QCD jets reconstructed with artificially low charged fraction,
  drifting into the signal region with no fake involved. There are far more real QCD jets than
  fakes, so this is potentially the larger effect.

Scripts: `llp_studies/fake_rate_by_sample.py`, `llp_studies/feature_migration.py`.
Samples: COCOA `dijet_test` (35696 ev) and `ttbar` (22500 ev) as QCD benchmarks, plus the HSS
sample. **Only the qqbar-trained model has inference on the QCD samples**, so the hss-vs-qqbar
comparison is available on the HSS sample only.

### 16.1 §1's fake rate is an artifact — and the ordering was backwards

| sample | model | jets/ev | ΔR fake | **overlap fake (<0.3)** |
|---|---|---|---|---|
| dijet | HGPflow (qqbar) | 2.88 | 6.1% | **3.8%** |
| dijet | PPflow | 2.99 | 10.5% | **7.5%** |
| ttbar | HGPflow (qqbar) | 5.98 | 6.8% | **4.8%** |
| ttbar | PPflow | 6.19 | 10.6% | **8.1%** |
| hss | HGPflow (hss) | 2.16 | **51.7%** | **7.0%** |
| hss | HGPflow (qqbar) | 1.93 | 47.8% | 6.7% |
| hss | PPflow | 1.89 | 46.7% | 8.6% |

The ΔR column **reproduces §1's fake-rate table exactly** (51.7 / 47.8 / 46.7 against its 0.517 /
0.478 / 0.467), which validates the measurement — and then shows those numbers are inflated ~7× by
the displacement artifact of §14. A correctly reconstructed displaced jet has no truth jet pointing
at it, so ΔR calls it a fake.

**The real fake rate on signal is 7.0%, not 51.7%**, and on that measure HGPflow *beats* PPflow
(7.0% vs 8.6%) rather than losing to it. §1's table should be read as superseded: it ranks the
reconstructions in the wrong order. On QCD, where the artifact is absent (6.1% vs 3.8% — the two
definitions nearly agree), HGPflow is ~2× cleaner than PPflow.

**In-domain training adds real jets, not fakes.** On the HSS sample the hss model produces 12% more
jets than qqbar (2.16 vs 1.93/ev) at essentially unchanged fake rate (7.0% vs 6.7%): **+6,479 real
jets for +674 fakes, about 10 real per extra fake.** The §15 containment gain is genuine, not
bought with spurious objects.

### 16.2 Feature migration: a 7–9× effect, between reconstruction families

Median charged energy fraction of well-reconstructed jets (overlap > 0.3), and the fraction leaking
below charged-fraction thresholds — i.e. into trackless, signal-like territory:

| sample / category | model | med ch | **ch<0.1** | ch<0.2 |
|---|---|---|---|---|
| **dijet, QCD jets** | HGPflow | 0.67 | **1.0%** | 2.8% |
| **dijet, QCD jets** | PPflow | 0.32 | **7.7%** | 22.3% |
| **ttbar, QCD jets** | HGPflow | 0.69 | **1.2%** | 3.2% |
| **ttbar, QCD jets** | PPflow | 0.32 | **11.2%** | 26.3% |
| hss, LLP tracker (signal) | HGPflow (hss) | 0.00 | **78.7%** | 82.3% |
| hss, LLP tracker (signal) | PPflow | 0.00 | **79.2%** | 85.3% |

**PPflow reconstructs ordinary QCD jets with ~30% charged energy against HGPflow's ~67%**, leaking
7.7–11.2% of them below ch<0.1 versus HGPflow's 1.0–1.2% — a **7–9× difference in background
leakage, at identical signal efficiency** (78.7% vs 79.2%). The cause is visible in the constituent
counts: PPflow makes *more* particles per jet (12–17 vs 7–11) but *fewer* charged ones (2–3 vs 4–6),
i.e. it fragments charged energy into neutral pieces, which is precisely the tracklessness a
displaced-jet tagger keys on.

Folding in that HGPflow also reconstructs ~20% more signal jets well enough to tag (38,142 vs
31,834), the crude single-variable arithmetic on dijet:

| | signal in region | background in region | S/B |
|---|---|---|---|
| PPflow | 25,213 | 5,785 | 4.4 |
| HGPflow | 30,018 | 754 | **39.8** |

**≈9× better S/B, ≈3× better S/√B**, from the reconstruction choice alone.

**And the question this section was written to answer:** comparing *prompt* jets in the HSS sample,
hss-trained gives ch<0.1 = **2.2%** against qqbar-trained's **2.0%**. In-domain LLP training does
**not** push ordinary jets toward signal-like features. The 0.2-point difference is negligible
beside the 7–9× gap between reconstruction families.

### 16.3 Caveats, and an incidental finding

- **Single variable.** Charged fraction is the dominant tagger input but not the only one; a real
  tagger's gains will not scale linearly with this ratio.
- **Not your detector.** COCOA dijet is a benchmark, not any specific analysis's background, and
  beam-induced background is not simulated at all. The *relative* comparison between
  reconstructions is the portable result; absolute leakage rates are not.
- **The hss-vs-qqbar comparison on QCD is still missing** — no hss-trained inference exists on
  dijet/ttbar. §16.2's in-domain conclusion rests on prompt jets inside the HSS sample, which is a
  proxy. Closing it needs a GPU inference pass.
- **Incidental:** the ttbar sample contains 1,777 tracker-region and 65 calorimeter-region jets by
  the LLP tagging — genuinely displaced *SM* jets, presumably strange-hadron decays in flight. The
  calorimeter-region ones show median ch = 0.21 with 27.7% below ch<0.1: irreducible physics
  background appearing unprompted in a Standard Model sample. Low statistics, but it is a handle on
  the population any displaced-jet tagger must reject, and it is measurable with this machinery.

---

## 17. Region B is not one population: the calorimeter depth split

§13.2 proposed late shower development as the mechanism for region B's missing energy, and §14.3
and §16 quote region B as a single number. Both are too coarse: region B spans ρ = 1496 → 3825 mm,
and a decay at the ECAL face has ~2.3 m of calorimeter left while one deep in the HCAL has a few
hundred mm. This splits it and tests the mechanism directly, using `particle_dep_energy` — the
energy each truth particle actually deposited — so no reconstruction is involved.
Script: `llp_studies/calo_depth_split.py`.

| decay radius [mm] | N | calo remaining | **dep_E / E** | axis<0.1 (momentum) | axis<0.1 (flight) |
|---|---|---|---|---|---|
| 1496–1800 | 369 | 3681 mm | **0.656** | 5.1% | **63.1%** |
| 1800–2200 | 351 | 2930 mm | **0.706** | 12.5% | 56.7% |
| 2200–2800 | 349 | 1856 mm | **0.268** | 30.4% | 48.7% |
| 2800–3825 | 204 | 567 mm | **0.000** | *93.1%* † | 1.5% |

Prompt-jet baseline under the same fiducial cut: **dep_E/E = 0.543**.

> † The bottom row is the degenerate-extrapolation artifact again (cf. §10, §11.1). `dep_E/E = 0.000`
> means these jets deposit nothing at all, so their "93.1% momentum-axis accuracy" is meaningless —
> non-propagated particles return ΔR = 0. Read that row as *dead*, exactly like region C.

### 17.1 The late-shower mechanism is confirmed

Deposited energy collapses monotonically once the remaining calorimeter depth falls below ~2 m:
**0.71 → 0.27 → 0.00**. §13.2's mechanism was stated as plausible; this measures it, and the
controlling variable is clearly the depth still available for the shower to develop in.

Two results that were not anticipated:

**Early calorimeter decays deposit *better* than prompt jets** — 0.656 and 0.706 against the prompt
baseline of 0.543. A particle born at the calorimeter face puts all of its energy into the
calorimeter: no tracker material upstream, no losses before the active volume. So region B's
"the energy is missing" verdict (§13.2) was **never true of the whole region** — it was the deep
tail dragging a bimodal distribution's average down. §13.2's observation of bimodality was right;
its cause is now identified as decay depth.

**The two reference axes trade places with depth.** Flight-direction accuracy falls (63.1% → 48.7%)
while momentum-axis accuracy rises (5.1% → 30.4%) as decays go deeper. The `(1 − L/R)` collimation
of §6.2 is strongest for early decays — there is still a long lever arm to the deposit — and weakens
as the decay approaches the deposit itself. The flight direction remains the better reference
throughout the live bins, but by a shrinking margin.

### 17.2 What this means for a CalRatio-type analysis

The ECAL/HCAL boundary sits near ~2000–2300 mm (calorimeter layer radii ρ = 1547, 1782, 1994, 2297,
3041, 3825), so CalRatio's target — HCAL energy with little ECAL — is roughly the **2200–2800 mm**
band. That band specifically:

| | value |
|---|---|
| energy deposited | **27%** of true (half the prompt baseline) |
| flight-direction axis accuracy | **48.7%** |
| share of region B | 27% |
| **share of all A⁰ decays** | **~4%** |

So CalRatio-relevant decays do deposit — inefficiently but measurably — and their axis is usable
against the flight direction, which is the reference vertex association wants anyway.

**The binding constraint is statistics, not reconstruction.** Only ~4% of decays land in the
CalRatio window, so a 30k-event sample yields roughly 2,400 such jets, and §10.3 showed no cτ
choice concentrates them there (the calorimeter fraction maxes at ~24%, and pushing cτ up mainly
feeds the invisible beyond-calorimeter tail). Generate-and-select is the only route, and a
dedicated study needs several times the current sample.

**And the region-B averages elsewhere in this document should be read as blends.** §14.3's 34.3%
and §16's ~26% end-to-end estimate mix three populations that behave quite differently: early-calo
decays (good containment, 57–63% flight accuracy), the CalRatio band (27% energy, 49% axis), and
dead deep-HCAL decays. Quote the band you mean rather than the region-B average.

---

## Methodology notes & caveats

- Two `PerformanceCOCOA` objects (one per model) share one truth file, so the PPflow baseline and
  truth jets are identical across the comparison. Event alignment was checked (corr ≈ 0.9+).
- The tight `ΔR < 0.1` match cut is deliberate; absolute efficiencies are cut-dependent (see §3.1),
  but the *model comparisons* and the *displacement mechanism* are not.
- `Eratio` uses reconstructed-particle pT within a fixed cone of the **truth** axis; centroid shift is
  computed relative to the truth axis (no φ-wrap issues). Both use pT weighting, so the dominant
  energy sets the measured position.
- The `Lxy`-vs-shift figure re-reads the production vertex from the raw file and re-applies the exact
  loader fiducial cut; alignment is asserted against the stored per-particle LLP flag before plotting.
- Statistics thin out above ~80 GeV (few hundred LLP jets and fewer); high-pT points carry large
  binomial errors and should not be over-interpreted.

### Notes on the §5–§7 measurements

- All of §5–§7 is measured on the **raw** generation file
  `/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root` (`Out_Tree`), 200–3000 events depending on
  the quantity (cell-level studies use fewer events because the cell branches are large). None of it
  depends on a trained model, so it is a statement about the sample and the detector, not about
  HGPflow.
- Particle classes use `hgpflow_v2.utility.helper_dicts.pdgid_class_dict` (0 ch.had, 1 e, 2 mu,
  3 neut.had, 4 photon), matching the convention in the reco class columns of §3.2.
- **Calorimeter geometry** was determined empirically from the stored extrapolation: barrel
  ρ ≈ 1496.5 mm, endcap |z| ≈ 3220 mm. The straight-line ray-to-surface model was validated against
  `particle_eta_extrap_calo`/`particle_phi_extrap_calo`: median ΔR = **0.0075** for neutrals
  (confirming the model) and **0.278** for charged (which *is* the bending, consistent with the
  charge-signed measurement in §6.1). The §7.2 oracle uses the stored extrapolated positions, so
  bending is fully present.
- **Bending is isolated by charge-signing** Δφ and restricted to π±/K±/p, where `sign(pdgid)` equals
  the electric charge. Geometry contributes with random sign and cancels in the median.
- **Decay grouping** (§6.2, §6.3, §7.3) groups final-state particles by production vertex rounded to
  5 mm. This is imperfect: the raw file has no `particle_parent_idx`, so decay chains cannot be
  reconstructed and secondary decays (e.g. K_S → ππ downstream) contaminate the grouping. Quantities
  derived from it are indicative rather than exact — this is why §6.2 compares *like-with-like*
  centroids and reads the *slope* rather than absolute values.
- **A superseded test**: an earlier attempt compared a vector-sum axis against a coordinate centroid.
  That mismatch alone is worth ΔR ≈ 1.19 on prompt groups and invalidated the comparison. §6.2 uses
  two pT-weighted centroids throughout.
- **§7.2 is an upper bound on axis recovery, not a predicted efficiency.** It uses truth particles with
  the same constituent set for both jets — perfect energy, perfect clustering, no thresholds. Real
  performance will be lower: HGPflow particle positions carry resolution, reco clustering differs from
  truth clustering, and the 56%-too-wide fragmentation problem (§6.3) is untouched. Do not compare its
  28% baseline against the 0.44 plateau of §2 — different selections and different quantities. The
  trustworthy number is the ratio (~2×).
- **§7.3 uses truth cell→particle links** (`cell_parent_idx`) to group showers before fitting. A real
  algorithm would have to do that grouping itself, so the quoted ~1700 mm is a *best case*.
- **Is the transverse-only LLP tag a problem? No — checked.** `llp_lxy_mm = 10` tags on `Lxy`
  alone, which on a forward sample looks like it should miss decays at large |z| and small
  transverse radius. Measured, it barely does: only **2.3%** of displaced pT (3D displacement
  > 10 mm) fails the transverse tag, and at jet level only **81 of 11286** jets are "prompt by
  `Lxy`, LLP in 3D" (0.7%), with **zero** going the other way. Those 81 have track coverage 67.6%
  and 93.8% axis accuracy — indistinguishable from the true prompt jets (64.7%, 96.9%) — so they do
  not drag the prompt control. The reason is geometric: reaching |z| = 2000 mm with `Lxy` < 10 mm
  requires |η| ≳ 6, far outside acceptance, so within |η| < 2.5 any genuinely displaced decay
  accumulates `Lxy` > 10 mm.

  The distinction worth remembering: **the tag is a threshold near zero, where `Lxy` and 3D
  distance agree for anything in acceptance; the *binning* asks "how displaced", at large distances
  where they diverge badly.** The tag was never the problem — using `Lxy` as an ordering variable
  was (§6.5, §7.1). §1–§3 are unaffected.
- **The primary vertex is unsmeared**, which is what makes a 3D displacement measure valid at all:
  particles with `Lxy` < 0.5 mm have median |z| = 0.000 mm, and the per-event PV z has std
  0.157 mm. So 3D distance from the origin is a genuine displacement, not a beamspot artifact.
- **Scripts**: every table in §5–§17 is reproduced by a script in
  [`llp_studies/`](llp_studies/) — see that directory's `README.md` for the script → section map
  and the run command. §5–§10 read the raw file only and need no trained checkpoint; the §11–§14
  measurements (`region_efficiency_reco`, `energy_response_by_region`, `overlap_match`) run the
  real evaluation pipeline and need the model's merged predictions (for §15, both models').

*Figures generated from `cocoa_qqbar_vs_hss_performance.ipynb`. Sections 5–14 added following the
follow-up analysis: displacement mechanism (§6), feasibility of the two proposed fixes (§7),
appropriateness and baselines (§9), the decay-region split (§10), and the measurements on real
model output (§11–§14). Every §5–§14 table is reproduced by a script in `llp_studies/`.*
