# SH‑PheWAS Explorer

Analyze and visualize **localized** vs **broad** cross‑phenotype genetic effects using **spherical harmonics** (SH).  
This repository contains:

- `sph_phewas_finngen.py` — batch CLI to fit SH to PheWAS summaries (e.g., FinnGen R12) and export figures/metrics.
- `make_r12_top25_with_genes.py` — helper to merge gene labels and build a **Top‑25** ranked table by localization score.
- `app.py` — an interactive **Streamlit** app to upload data, run the pipeline, and explore volcano plots, \(l_{95}\) histograms, and per‑variant maps.

---

## Contents

- [Why spherical harmonics for PheWAS?](#why-spherical-harmonics-for-phewas)
- [Installation](#installation)
- [Data formats](#data-formats)
- [Quick start](#quick-start)
  - [Batch CLI (FinnGen‑style TSV)](#a-batch-cli-finngen-style-tsv)
  - [Top‑25 with gene labels](#b-create-top-25-csv-with-gene-labels)
  - [Interactive Streamlit app](#c-interactive-app)
- [Outputs and interpretation](#outputs-and-interpretation)
- [Method (intuition)](#method-intuition)
- [Function reference](#function-reference)
  - [Embedding & coordinates](#embedding--coordinates)
  - [Spherical harmonics & solver](#spherical-harmonics--solver)
  - [I/O & plotting helpers](#io--plotting-helpers)
  - [App orchestration](#app-orchestration)
- [Troubleshooting](#troubleshooting)
- [Reproducibility checklist](#reproducibility-checklist)
- [License](#license)
- [Contact](#contact)

---

## Why spherical harmonics for PheWAS?

Each variant has an effect profile across many phenotypes. If we map phenotypes to points on the unit sphere \(S^2\), each variant’s effect becomes a **function on the sphere**.  
Low SH degrees (\(l=0,1,2\)) capture **smooth/global** structure (monopole/dipole/bands); higher degrees capture **localized** hot spots. By fitting SH and inspecting the **degree‑wise power**, we get **rotation‑invariant** measures of localization:

- \(l_{95}\) — minimum degree needed to capture **95%** of spectral power (larger ⇒ finer structure).
- \(\mathrm{LI}_{>L_0}\) — fraction of power in degrees \(>L_0\) (larger ⇒ more localized).
- A nested **F‑test** compares a smooth model (\(L_0\)) to a richer one (\(L_{\max}\)) to test for **excess high‑\(l\)** content (multiple testing by BH‑FDR).

**Takeaway:** localized biology appears as **compact islands** on the sphere and as **heavy spectral tails** at moderate/high \(l\).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Base requirements
pip install numpy pandas scipy matplotlib

# For the Streamlit app
pip install streamlit
```

Optional `requirements.txt`:
```txt
streamlit>=1.28
numpy>=1.23
pandas>=1.5
scipy>=1.9
matplotlib>=3.6
```

Python ≥3.9 recommended.

---

## Data formats

Use either of the following:

### 1) Long TSV (FinnGen‑style)
```
pheno   variant   beta_add   sebeta_add   ...
```
- `pheno`: phenotype/endpoint ID  
- `variant` (or `rsid`): variant identifier  
- `beta_add`, `sebeta_add`: effect & SE (configurable column names)

### 2) Matrices (two aligned files)
- **betas** — rows = variants, columns = phenotypes  
- **SEs** — rows = variants, columns = phenotypes

Missing entries are allowed; the solver down‑weights them (`se = ∞` ⇒ weight 0).

---

## Quick start

### A) Batch CLI (FinnGen‑style TSV)

```bash
python sph_phewas_finngen.py \
  --tsv R12_coding_variant_results_1e-5_annotated.tsv \
  --prefix r12 \
  --embedding data \
  --top_k 6 \
  --L0 2
```

Outputs:
- `r12_metrics.csv` — per‑variant metrics
- `r12_volcano.png` — localization volcano
- `r12_l95_hist.png` — histogram of \(l_{95}\)
- `r12_top{1..6}_map.png` — normalized maps for top variants

**Tip:** Use `--embedding fibonacci` to avoid SVD in the embedding step.

### B) Create Top‑25 CSV with gene labels

```bash
python make_r12_top25_with_genes.py
# Produces: r12_metrics_top25.csv
```

### C) Interactive app

```bash
streamlit run app.py
```
In the UI: upload a long TSV or a pair of matrices (betas + SEs), choose embedding and \(L\) grid, run analysis, and explore volcano, \(l_{95}\) histogram, metrics table (downloadable), and interactive maps.

---

## Outputs and interpretation

- **Volcano** (`*_volcano.png`): x = \(\mathrm{LI}_{>2}\) (fraction of power beyond very smooth modes), y = \(-\log_{10} p\) from the high‑\(l\) test. **Upper‑right** = localized **and** significant.
- **Histogram** (`*_l95_hist.png`): distribution of \(l_{95}\). **Right tail** = variants needing high degrees (localized hotspots).
- **Maps** (`*_map.png`): normalized plate‑carrée reconstruction. **Bright islands** = compact phenotype neighborhoods with strong, coherent effects.
- **Metrics CSV** (`*_metrics.csv`):
  - `variant`: `CHR-POS-REF-ALT`
  - `Lmax`: BIC‑selected maximum degree
  - `l95`: degree to reach 95% energy (fineness)
  - `lbar`: spectral centroid (energy‑weighted mean degree)
  - `H`: normalized spectral entropy (spread across degrees)
  - `LI_2`: localization index (power fraction at \(l>2\))
  - `F_highl`, `p_highl`: nested F‑test for high‑\(l\) structure
  - `FDR_sig`: BH‑FDR flag
  - `score`: ranking (combination of \(-\log_{10}p\) and \(\mathrm{LI}_2\), with an FDR boost)

**Heuristics:** prioritize `FDR_sig=True`, \(l_{95}\ge 8\)–10, \(\mathrm{LI}_{>2}\ge 0.6\)–0.7.

---

## Method (intuition)

1. **Embed phenotypes on \(S^2\)**  
   - *Data‑driven*: PCA/SVD on phenotype‑by‑variant \(z\)‑matrix; rows normalized onto the sphere.  
   - *Fibonacci*: quasi‑uniform points (no SVD dependency).  
   Rotation‑invariant metrics (e.g., \(l_{95}\), \(\mathrm{LI}\)) don’t depend on global orientation.

2. **Fit spherical harmonics (per variant)**  
   Weighted least squares with a **Laplacian ridge** (penalize \(l(l+1)\) for \(l>0\)) stabilizes high‑degree estimates. Select \(L_{\max}\) by **BIC** over a grid (e.g., 6–16).

3. **Summarize localization**  
   **Degree‑wise power** \(P_l\) reveals where energy lives.  
   Broad effects: low \(l\), small \(l_{95}\).  
   Localized effects: moderate/high \(l\), large \(l_{95}\), large \(\mathrm{LI}\).

4. **Test high‑degree structure**  
   Nested **F‑test** compares \(L_0\) (smooth) vs \(L_{\max}\) (rich). BH‑FDR controls multiplicity across variants.

---

## Function reference

### Embedding & coordinates

- **`fibonacci_sphere(k) -> (k×3) np.ndarray`**  
  Quasi‑uniform points on \(S^2\). Neutral embedding without SVD.

- **`data_driven_embedding(betas, ses) -> (k×3) np.ndarray`**  
  Builds a phenotype similarity embedding from the \(z\)‑matrix (betas/SEs). Projects to 3D and normalizes onto \(S^2\).  
  *Intuition:* nearby points ↔ similar phenotype patterns across variants.

- **`cart_to_sph(xyz) -> (theta, phi)`**  
  Cartesian → spherical angles: \(\theta\) (colatitude), \(\phi\) (longitude).

### Spherical harmonics & solver

- **`design_matrix(theta, phi, L) -> (Y, lm_list)`**  
  \(Y \in \mathbb{C}^{k \times (L+1)^2}\) with entries \(Y_{lm}(\theta_j,\phi_j)\). `lm_list` lists \((l,m)\).

- **`ridge_WLS_chol(y, Y, se, lm_list, lam=1e-3, w_clip=1e6, jitter0=1e-8, max_tries=6) -> (a, SSE)`**  
  Weighted ridge (Laplacian) **solved by Cholesky** (no SVD in solver), with adaptive diagonal jitter to ensure the system is SPD. Returns complex SH coefficients `a` and weighted SSE.  
  *Intuition:* penalizing \(l(l+1)\) shrinks high‑\(l\) oscillations unless the data support them.

- **`degree_powers(a, lm_list, L) -> P`**  
  Aggregates \(|a_{lm}|^2\) over \(m\) to get per‑degree power \(P_l\).

- **`spectral_descriptors(P) -> {l95,lbar,H}`**  
  \(l_{95}\): degree to reach 95% cumulative power;  
  \( \bar{\ell}\): energy‑weighted mean degree;  
  \(H\): normalized spectral entropy across degrees.

- **`localization_index(P, L0) -> float`**  
  \(\mathrm{LI}_{>L_0} = \sum_{l>L_0} P_l / \sum_l P_l\). Larger ⇒ more localized.

- **`nested_F_test(beta, se, theta, phi, L0, L1, lam) -> (F, p)`**  
  Fits \(L_0\) and \(L_1\) models; compares SSEs with a standard nested \(F\)‑statistic.  
  *Intuition:* asks if extra high‑\(l\) degrees significantly improve fit beyond a smooth baseline.

- **`bh_fdr(pvals, alpha=0.05) -> mask`**  
  Benjamini–Hochberg FDR across variants; returns a boolean discovery mask.

- **`evaluate_on_grid(a, L, n_lon=360, n_lat=181) -> (f_grid, lon_deg, lat_deg)`**  
  Reconstructs the SH surface on a regular lat‑lon grid (for plotting).

### I/O & plotting helpers

- **`load_long_tsv(file, pheno_col, var_col, beta_col, se_col)`**  
  Reads a FinnGen‑style TSV into normalized internal format.

- **`build_mats_from_long(df, phenos, variants)`**  
  Builds \(M \times k\) betas/SEs matrices, using **smallest SE** when duplicates exist.

- **`load_matrix_csvs(beta_file, se_file)`**  
  Reads aligned betas/SEs matrices (rows=variants, cols=phenotypes).

- **`make_volcano(met, L0)`**  
  Scatter: \(\mathrm{LI}_{>L_0}\) vs \(-\log_{10} p\), highlighting FDR discoveries.

- **`make_l95_hist(met)`**  
  Histogram of \(l_{95}\) across variants.

- **`render_map(variant, met, betas, ses, theta, phi)`**  
  Re‑fits the chosen variant at its `Lmax` and renders the normalized map.

### App orchestration

- **`run_analysis(betas, ses, phenos, variants, embedding_method, L_grid, L0, lam)`**  
  Full pipeline: embedding → BIC over \(L\) → fit → metrics → FDR → ranking.  
  Returns the metrics table and the spherical coordinates for plotting.

---

## Troubleshooting

- **“SVD did not converge …”**  
  - Use `--embedding fibonacci` in CLI / switch to **Fibonacci** in the app to avoid SVD in embedding.  
  - The solver itself **does not use SVD** (Cholesky with jitter), so fits remain stable.

- **Slow or memory‑heavy runs**  
  - Lower max \(L\) (e.g., 12) or reduce grid density for maps.  
  - If \(k\) (phenotypes) is large, reduce the size of the \(L\) grid.

- **All \(p\) values near 0** (underflow)  
  - Expected for extremely significant tests; display as `<1e-300` if desired.

---

## Reproducibility checklist

- Record `L_grid`, `L0`, `lam`, and **embedding method** in logs/manuscripts.  
- Keep the **metrics CSV** alongside figures; it fully describes the ranking and FDR calls.  
- For manuscripts, consider rendering: volcano, \(l_{95}\) histogram, top‑K maps, and a Top‑25 table (CSV‑driven).

---

## License

Choose a license (e.g., MIT) and add it as `LICENSE` in your repo.

---

## Contact

- **Department of Biomedical Data Science, Stanford University**  
- **Manuel A. Rivas** — <mrivas@stanford.edu>

Authors: Manuel A. Rivas, GPT‑5‑Thinking, Grok4.
