#!/usr/bin/env python3
"""
FinnGen R12 coding variants: spherical-harmonics PheWAS analysis
- Input: TSV with columns including pheno, variant (or rsid), beta_add, sebeta_add
- Outputs: r12_metrics.csv, r12_volcano.png, r12_l95_hist.png, r12_top{k}_map.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.special import sph_harm
from scipy.stats import f as f_dist
from typing import Tuple, List, Dict, Optional
from sklearn.decomposition import TruncatedSVD

# -----------------------
# I/O and assembly
# -----------------------

def load_finngen_tsv(path: str,
                     pheno_col="pheno", var_col="variant",
                     beta_col="beta_add", se_col="sebeta_add") -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = pd.read_csv(path, sep="\t", dtype={pheno_col: str, var_col: str}, low_memory=False)
    # Some files may use 'rsid' instead of 'variant'
    if var_col not in df.columns and "rsid" in df.columns:
        var_col = "rsid"
    required = [pheno_col, var_col, beta_col, se_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only required cols
    df = df[[pheno_col, var_col, beta_col, se_col]].copy()
    df.rename(columns={pheno_col: "pheno", var_col: "variant", beta_col: "beta", se_col: "se"}, inplace=True)

    # Drop rows missing both beta and se
    df = df[~(df["beta"].isna() & df["se"].isna())].reset_index(drop=True)

    phenos = sorted(df["pheno"].unique().tolist())
    variants = sorted(df["variant"].unique().tolist())
    return df, phenos, variants

def build_matrices(df: pd.DataFrame, phenos: List[str], variants: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return betas (M x k) and ses (M x k) with np.inf se for missing."""
    k = len(phenos)
    M = len(variants)
    pheno_to_idx = {p: j for j, p in enumerate(phenos)}
    var_to_idx = {v: i for i, v in enumerate(variants)}

    betas = np.full((M, k), np.nan, dtype=float)
    ses   = np.full((M, k), np.nan, dtype=float)

    for _, row in df.iterrows():
        i = var_to_idx[row["variant"]]
        j = pheno_to_idx[row["pheno"]]
        b = row["beta"]
        s = row["se"]
        # Prefer smaller SE if duplicates exist
        if np.isnan(betas[i, j]) or (not np.isnan(s) and s < ses[i, j]):
            betas[i, j] = b
            ses[i, j] = s

    # Mark missing with se=inf and beta=0
    miss = np.isnan(betas) | np.isnan(ses) | (ses <= 0)
    ses[miss] = np.inf
    betas[miss] = 0.0
    return betas, ses

# -----------------------
# Sphere utilities
# -----------------------

def fibonacci_sphere(k: int) -> np.ndarray:
    pts = []
    phi_g = np.pi * (3 - np.sqrt(5))
    for i in range(k):
        y = 1 - 2 * i / float(max(k - 1, 1))
        r = np.sqrt(max(0.0, 1 - y*y))
        theta = phi_g * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        pts.append((x, y, z))
    return np.array(pts, dtype=float)

def data_driven_embedding(betas: np.ndarray, ses: np.ndarray) -> np.ndarray:
    """
    Phenotype embedding via truncated SVD on z-matrix (k x M),
    then project to unit sphere. Fallback to Fibonacci if degenerate.
    """
    M, k = betas.shape
    # Build z with zero-fill for missing (already zero-weighted)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = betas / ses
    z[np.isinf(z)] = 0.0
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    Z = z.T  # shape (k, M): phenotypes x variants
    # Center columns weakly (optional): not strictly necessary
    # Use rank at most 3
    r = min(3, min(Z.shape)-1)
    if r < 1:
        return fibonacci_sphere(k)

    svd = TruncatedSVD(n_components=r, random_state=42)
    X = svd.fit_transform(Z)  # (k x r)
    if X.shape[1] < 3:
        X = np.pad(X, ((0,0),(0, 3 - X.shape[1])), mode="constant", constant_values=0.0)

    # Normalize rows to lie on S^2
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    S = X / norms
    return S  # (k x 3)

def cart_to_sph(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = xyz.T
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2*np.pi
    return theta, phi

# -----------------------
# SH machinery
# -----------------------

def design_matrix(theta: np.ndarray, phi: np.ndarray, L: int) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    k = theta.shape[0]
    p = (L + 1)**2
    Y = np.zeros((k, p), dtype=complex)
    lm_list = []
    idx = 0
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y[:, idx] = sph_harm(m, l, phi, theta)
            lm_list.append((l, m))
            idx += 1
    return Y, lm_list

from scipy.linalg import cho_factor, cho_solve

def ridge_WLS(y: np.ndarray,
              Y: np.ndarray,
              se: np.ndarray,
              lm_list: List[Tuple[int,int]],
              lam: float = 1e-3,
              w_clip: float = 1e6,
              jitter0: float = 1e-8,
              max_tries: int = 6) -> Tuple[np.ndarray, float]:
    """
    Weighted ridge with Laplacian penalty; solved via Cholesky (no SVD).
    Ensures SPD by adding a small identity jitter that grows if needed.
    Returns (a_hat complex, SSE).
    """
    # Weights (clip extremes; zero-out non-finite)
    w = 1.0 / np.maximum(se, 1e-12)**2
    w[~np.isfinite(w)] = 0.0
    w = np.minimum(w, w_clip)
    sqrt_w = np.sqrt(w)

    # Weighted design
    Yw = Y * sqrt_w[:, None]
    yw = y * sqrt_w

    # Build Laplacian ridge on coefficients (block-real form)
    p = Y.shape[1]
    lap_diag = np.array([l*(l+1) if l > 0 else 0.0 for (l, m) in lm_list], dtype=float)
    # Real/imag block (size 2p)
    Pbig_diag = np.concatenate([lap_diag, lap_diag])

    # Real-augmented system
    Ywr = np.hstack([Yw.real, -Yw.imag])   # (k x 2p)
    Ywi = np.hstack([Yw.imag,  Yw.real])   # (k x 2p)
    Ybig = np.vstack([Ywr, Ywi])           # (2k x 2p)
    ybig = np.hstack([yw.real, yw.imag])   # (2k,)

    # Normal matrix + ridge (strictly SPD with jitter on the diagonal)
    A = Ybig.T @ Ybig
    # Add Laplacian ridge
    A += lam * np.diag(Pbig_diag)
    # Also add a tiny uniform ridge on all coefficients to guarantee SPD
    # (this does not bias low-ℓ meaningfully but stabilizes numerics)
    A += (lam * 1e-6) * np.eye(A.shape[0])

    b = Ybig.T @ ybig

    # Cholesky with increasing jitter if needed
    jitter = jitter0
    for _ in range(max_tries):
        try:
            c, lower = cho_factor(A + jitter * np.eye(A.shape[0]), lower=False, check_finite=False)
            ahat_big = cho_solve((c, lower), b, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        # Final fallback: add a larger diagonal and try once more
        c, lower = cho_factor(A + (10.0 * jitter + 1e-4) * np.eye(A.shape[0]), lower=False, check_finite=False)
        ahat_big = cho_solve((c, lower), b, check_finite=False)

    # Reassemble complex coefficients
    a = ahat_big[:p] + 1j * ahat_big[p:]

    # SSE on the original weighted residuals (real field)
    resid = y - (Y @ a).real
    SSE = float(np.sum(w * resid**2))
    return a, SSE

def degree_powers(a: np.ndarray, lm_list: List[Tuple[int,int]], L: int) -> np.ndarray:
    P = np.zeros(L + 1, dtype=float)
    for coeff, (l, m) in zip(a, lm_list):
        P[l] += float(np.abs(coeff)**2)
    return P

def spectral_descriptors(P: np.ndarray) -> Dict[str, float]:
    total = float(P.sum())
    if total <= 0:
        return {"l95": 0, "lbar": 0.0, "H": 0.0}
    cum = np.cumsum(P) / total
    l95 = int(np.searchsorted(cum, 0.95))
    lbar = float(np.sum(np.arange(len(P)) * P) / total)
    p = P / total
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -np.nansum(p * np.log(p))
    H = float(ent / np.log(len(P)))
    return {"l95": l95, "lbar": lbar, "H": H}

def localization_index(P: np.ndarray, L0: int) -> float:
    total = float(P.sum())
    if total <= 0:
        return 0.0
    hi = float(P[L0+1:].sum())
    return hi / total

def nested_F_test(beta: np.ndarray, se: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                  L0: int, L1: int, lam: float = 1e-3) -> Tuple[float, float]:
    Y0, lm0 = design_matrix(theta, phi, L0)
    a0, SSE0 = ridge_WLS(beta, Y0, se, lm0, lam=lam)
    Y1, lm1 = design_matrix(theta, phi, L1)
    a1, SSE1 = ridge_WLS(beta, Y1, se, lm1, lam=lam)
    k = len(beta)
    p0 = (L0 + 1)**2
    p1 = (L1 + 1)**2
    df1 = max(p1 - p0, 1)
    df2 = max(k - p1, 1)
    num = (SSE0 - SSE1) / df1
    den = SSE1 / df2
    F = float(num / den) if den > 0 else np.inf
    p = float(f_dist.sf(F, df1, df2)) if np.isfinite(F) else 0.0
    return F, p

def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    m = len(pvals)
    idx = np.argsort(pvals)
    thr = alpha * (np.arange(1, m+1) / m)
    passed = pvals[idx] <= thr
    k = np.where(passed)[0].max()+1 if passed.any() else 0
    cutoff = thr[k-1] if k>0 else 0.0
    return pvals <= cutoff

def evaluate_on_grid(a: np.ndarray, L: int, n_lon: int = 360, n_lat: int = 181) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.linspace(0, 2*np.pi, n_lon)
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon_g, lat_g = np.meshgrid(lon, lat)
    phi_g = lon_g
    theta_g = np.pi/2 - lat_g
    f = np.zeros_like(phi_g, dtype=float)
    idx = 0
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y = sph_harm(m, l, phi_g, theta_g)
            f += (a[idx] * Y).real
            idx += 1
    return f, lon_g * 180/np.pi, lat_g * 180/np.pi

def plate_carre_plot(path: str, fgrid: np.ndarray, lon_deg: np.ndarray, lat_deg: np.ndarray, title: str):
    plt.figure(figsize=(7.2, 3.6))
    A = fgrid.copy()
    mn, mx = np.min(A), np.max(A)
    if mx > mn:
        A = (A - mn) / (mx - mn)
    plt.pcolormesh(lon_deg, lat_deg, A, cmap='plasma', shading='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Normalized effect')
    plt.xlabel('Longitude (°)'); plt.ylabel('Latitude (°)')
    plt.title(title)
    plt.ylim(-90, 90); plt.xlim(0, 360)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

# -----------------------
# Main analysis
# -----------------------

def analyze_finngen(tsv_path: str,
                    out_prefix: str = "r12",
                    embedding: str = "data",   # "data" or "fibonacci"
                    L_grid: List[int] = [6,8,10,12,14,16],
                    L0: int = 2,
                    lam: float = 1e-3,
                    top_k_maps: int = 6):
    df, phenos, variants = load_finngen_tsv(tsv_path)
    betas, ses = build_matrices(df, phenos, variants)
    M, k = betas.shape

    # Phenotype coordinates on sphere
    if embedding == "data":
        S = data_driven_embedding(betas, ses)  # (k x 3)
    else:
        S = fibonacci_sphere(k)
    theta, phi = cart_to_sph(S)

    # Per-variant loop
    metrics = []
    for i in range(M):
        beta = betas[i, :]
        se = ses[i, :]

        # Choose Lmax by BIC
        bic_best = None
        Lmax_best = None
        a_best = None
        Y_best = None
        lm_best = None
        SSE_best = None

        for L in L_grid:
            Y, lm = design_matrix(theta, phi, L)
            a, SSE = ridge_WLS(beta, Y, se, lm, lam=lam)
            p = (L + 1)**2
            bic = k * np.log(max(SSE / max(k,1), 1e-12)) + p * np.log(max(k,1))
            if (bic_best is None) or (bic < bic_best):
                bic_best = bic
                Lmax_best = L
                a_best = a; Y_best = Y; lm_best = lm; SSE_best = SSE

        P = degree_powers(a_best, lm_best, Lmax_best)
        desc = spectral_descriptors(P)
        LI = localization_index(P, L0=L0)
        F, pval = nested_F_test(beta, se, theta, phi, L0=L0, L1=Lmax_best, lam=lam)

        metrics.append({
            "variant": variants[i],
            "Lmax": Lmax_best,
            "l95": int(desc["l95"]),
            "lbar": desc["lbar"],
            "H": desc["H"],
            f"LI_{L0}": LI,
            "F_highl": F,
            "p_highl": pval
        })

    met = pd.DataFrame(metrics)
    # BH-FDR
    met["FDR_sig"] = bh_fdr(met["p_highl"].values, alpha=0.05)

    # Rank for maps: FDR first, then -log10(p) * LI
    score = (-np.log10(np.maximum(met["p_highl"].values, 1e-300))) * met[f"LI_{L0}"].values
    met["score"] = score + (met["FDR_sig"].astype(int) * 1000.0)  # big boost if FDR significant
    met = met.sort_values(["FDR_sig", "score"], ascending=[False, False]).reset_index(drop=True)

    # Save metrics
    met.to_csv(f"{out_prefix}_metrics.csv", index=False)

    # Volcano
    plt.figure(figsize=(6.5, 5.0))
    x = met[f"LI_{L0}"].values
    y = -np.log10(np.maximum(met["p_highl"].values, 1e-300))
    colors = np.where(met["FDR_sig"].values, "tab:red", "tab:blue")
    plt.scatter(x, y, s=18, alpha=0.85, c=colors)
    # Annotate top 20 by score
    for r in range(min(20, len(met))):
        plt.annotate(met.loc[r, "variant"], (x[r], y[r]), fontsize=7, xytext=(3,3), textcoords='offset points')
    plt.xlabel(f'Localization Index LI (l>{L0})'); plt.ylabel(r'$-\log_{10} p_{\mathrm{high}\ell}$')
    plt.title('FinnGen R12: localization volcano')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_volcano.png", dpi=200); plt.close()

    # Histogram of l95
    plt.figure(figsize=(6.5, 4.0))
    plt.hist(met["l95"].values, bins=np.arange(met["l95"].max()+2)-0.5)
    plt.xlabel(r'$l_{95}$'); plt.ylabel('Count')
    plt.title('FinnGen R12: distribution of $l_{95}$ across variants')
    plt.tight_layout(); plt.savefig(f"{out_prefix}_l95_hist.png", dpi=200); plt.close()

    # Make maps for top_k variants
    # Refit with saved best L and coefficients for each selected variant
    selected = met.head(top_k_maps)
    # Precompute design matrices up to max L to avoid recompute? Simplicity first.
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        v = row["variant"]
        i = variants.index(v)
        beta = betas[i, :]
        se = ses[i, :]
        Lmax = int(row["Lmax"])
        Y, lm = design_matrix(theta, phi, Lmax)
        a, SSE = ridge_WLS(beta, Y, se, lm, lam=lam)
        fgrid, lon_deg, lat_deg = evaluate_on_grid(a, Lmax)
        plate_carre_plot(f"{out_prefix}_top{rank}_map.png", fgrid, lon_deg, lat_deg,
                         f"{v} (Lmax={Lmax}, l95={int(row['l95'])}, LI={row[f'LI_{L0}']:.2f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FinnGen R12 SH-PheWAS analyzer")
    ap.add_argument("--tsv", required=True, help="Path to R12_coding_variant_results_1e-5_annotated.tsv")
    ap.add_argument("--prefix", default="r12", help="Output prefix")
    ap.add_argument("--embedding", choices=["data","fibonacci"], default="data", help="Phenotype embedding method")
    ap.add_argument("--top_k", type=int, default=6, help="Number of top variant maps to render")
    ap.add_argument("--L0", type=int, default=2, help="Low-degree boundary for LI and F-test")
    args = ap.parse_args()

    analyze_finngen(tsv_path=args.tsv, out_prefix=args.prefix,
                    embedding=args.embedding, top_k_maps=args.top_k, L0=args.L0)
