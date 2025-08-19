# app.py — SH-PheWAS Explorer (fast + cap enrichment)
# streamlit run app.py

import io
import os
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import f as f_dist, norm

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SH-PheWAS Explorer (Fast)", page_icon="🧭", layout="wide")

# -----------------------------------------------------------------------------
# Sphere / embedding helpers
# -----------------------------------------------------------------------------
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
    """Phenotype embedding via PCA on (k×M) z-matrix. Falls back to Fibonacci if SVD fails."""
    M, k = betas.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        z = betas / ses
    z[np.isinf(z)] = 0.0
    Z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).T  # (k×M)
    if k < 3 or M == 0:
        return fibonacci_sphere(k)
    try:
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        X = U[:, :3]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms
    except Exception:
        return fibonacci_sphere(k)

def cart_to_sph(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = xyz.T
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-12), -1, 1))
    phi = np.arctan2(y, x); phi[phi < 0] += 2*np.pi
    return theta, phi

# -----------------------------------------------------------------------------
# Precompute SH basis (cache once)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def precompute_sh_basis(theta: np.ndarray,
                        phi: np.ndarray,
                        Lmax: int,
                        dtype=np.complex64
                        ) -> Tuple[np.ndarray, Dict[int, slice], List[Tuple[int,int]], np.ndarray]:
    """
    Returns:
      Y_all: (k × pmax) complex64 with pmax=(Lmax+1)^2
      cols_for_L: dict L -> slice(0, p(L))
      lm_list: [(l,m)] for columns
      lap_diag: Laplacian penalty diag for pmax columns (float32)
    """
    k = theta.shape[0]
    pmax = (Lmax + 1)**2
    Y_all = np.zeros((k, pmax), dtype=dtype)
    lm_list = []
    idx = 0
    for l in range(Lmax + 1):
        for m in range(-l, l + 1):
            Y_all[:, idx] = sph_harm(m, l, phi, theta).astype(dtype, copy=False)
            lm_list.append((l, m)); idx += 1
    cols_for_L = {L: slice(0, (L + 1)**2) for L in range(Lmax + 1)}
    lap = np.array([l*(l+1) if l > 0 else 0.0 for (l, m) in lm_list], dtype=np.float32)
    return Y_all, cols_for_L, lm_list, lap

# -----------------------------------------------------------------------------
# Ridge WLS solver (fast, complex Hermitian normal equations)
# -----------------------------------------------------------------------------
def ridge_WLS_chol_fast(beta: np.ndarray,
                        se: np.ndarray,
                        Y_all: np.ndarray,
                        cols: slice,
                        lap_diag: np.ndarray,
                        lam: float = 1e-3,
                        w_clip: float = 1e6,
                        jitter0: float = 1e-7,
                        max_tries: int = 6) -> Tuple[np.ndarray, float]:
    """Weighted ridge using precomputed Y_all; returns complex coefficients and SSE."""
    se = se.astype(np.float32, copy=False)
    w = 1.0 / np.maximum(se, 1e-12, dtype=np.float32)**2
    w[~np.isfinite(w)] = 0.0
    w = np.minimum(w, w_clip).astype(np.float32)
    sqrt_w = np.sqrt(w, dtype=np.float32)

    Y = Y_all[:, cols]                     # (k × p) complex64
    Yw = (Y.T * sqrt_w).T                  # row scaling
    yw = (beta.astype(np.float32) * sqrt_w).astype(np.float32)

    A = (Yw.conj().T @ Yw).astype(np.complex64)
    A += lam * np.diag(lap_diag[cols]).astype(np.complex64)
    A += (lam * 1e-6) * np.eye(A.shape[0], dtype=np.complex64)
    b = (Yw.conj().T @ yw.astype(np.complex64)).astype(np.complex64)

    jitter = jitter0
    for _ in range(max_tries):
        try:
            c, lower = cho_factor(A + jitter*np.eye(A.shape[0], dtype=np.complex64),
                                  lower=False, check_finite=False, overwrite_a=False)
            a = cho_solve((c, lower), b, check_finite=False).astype(np.complex64)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        c, lower = cho_factor(A + (10.0*jitter+1e-4)*np.eye(A.shape[0], dtype=np.complex64),
                              lower=False, check_finite=False)
        a = cho_solve((c, lower), b, check_finite=False).astype(np.complex64)

    resid = beta.astype(np.float32) - (Y @ a).real.astype(np.float32)
    SSE = float(np.sum(w * resid**2))
    return a, SSE

# -----------------------------------------------------------------------------
# Spectral summaries
# -----------------------------------------------------------------------------
def degree_powers(a: np.ndarray, lm_list: List[Tuple[int,int]], L: int) -> np.ndarray:
    P = np.zeros(L + 1, dtype=np.float32)
    for coeff, (l, m) in zip(a, lm_list[: (L + 1)**2 ]):
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
    H = float(ent / np.log(max(len(P), 2)))
    return {"l95": l95, "lbar": lbar, "H": H}

def localization_index(P: np.ndarray, L0: int) -> float:
    total = float(P.sum())
    if total <= 0:
        return 0.0
    return float(P[L0+1:].sum()) / total

# -----------------------------------------------------------------------------
# Evaluate on lat-lon grid (for maps)
# -----------------------------------------------------------------------------
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

def normalize01(A: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(A)), float(np.max(A))
    return (A - mn) / (mx - mn) if mx > mn else A

# -----------------------------------------------------------------------------
# BH-FDR
# -----------------------------------------------------------------------------
def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    pvals = np.asarray(pvals)
    m = len(pvals)
    idx = np.argsort(pvals)
    thr = alpha * (np.arange(1, m+1) / m)
    passed = pvals[idx] <= thr
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    cutoff = thr[k-1] if k > 0 else 0.0
    return pvals <= cutoff

# -----------------------------------------------------------------------------
# Spherical-cap enrichment
# -----------------------------------------------------------------------------
def _vec_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], dtype=float)

def _angdist(theta1: float, phi1: float, theta2: float, phi2: float) -> float:
    u = _vec_from_angles(theta1, phi1)
    v = _vec_from_angles(theta2, phi2)
    x = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(x))

def find_hotspots_from_SH(beta: np.ndarray,
                          se: np.ndarray,
                          theta: np.ndarray,
                          phi: np.ndarray,
                          Lmax: int,
                          lam: float,
                          Y_all: Optional[np.ndarray] = None,
                          cols_for_L: Optional[Dict[int, slice]] = None,
                          lap_diag: Optional[np.ndarray] = None,
                          grid_lon: int = 360,
                          grid_lat: int = 181,
                          take_both_signs: bool = True) -> List[Dict]:
    """Return global pos/neg maxima from SH reconstruction."""
    if Y_all is None:
        # slower path (not used in Fast mode)
        from math import isfinite
        k = len(beta)
        p = (Lmax + 1)**2
        Y = np.zeros((k, p), dtype=np.complex64)
        idx = 0
        for l in range(Lmax + 1):
            for m in range(-l, l + 1):
                Y[:, idx] = sph_harm(m, l, phi, theta).astype(np.complex64); idx += 1
        lap = np.array([l*(l+1) if l>0 else 0.0 for l in range(Lmax+1) for m in range(-l, l+1)], dtype=np.float32)
        a, _ = ridge_WLS_chol_fast(beta, se, Y, slice(0, p), lap, lam=lam)
    else:
        a, _ = ridge_WLS_chol_fast(beta, se, Y_all, cols_for_L[Lmax], lap_diag, lam=lam)

    lon = np.linspace(0, 2*np.pi, grid_lon)
    lat = np.linspace(-np.pi/2, np.pi/2, grid_lat)
    lon_g, lat_g = np.meshgrid(lon, lat)
    phi_g = lon_g; theta_g = np.pi/2 - lat_g

    f = np.zeros_like(phi_g, dtype=float); idx = 0
    for l in range(Lmax + 1):
        for m in range(-l, l + 1):
            f += (a[idx] * sph_harm(m, l, phi_g, theta_g)).real; idx += 1

    hotspots = []
    pos_idx = np.argmax(f); i_pos, j_pos = np.unravel_index(pos_idx, f.shape)
    hotspots.append({'theta': float(theta_g[i_pos, j_pos]),
                     'phi': float(phi_g[i_pos, j_pos]),
                     'value': float(f[i_pos, j_pos]), 'sign': 'pos'})
    if take_both_signs:
        neg_idx = np.argmin(f); i_neg, j_neg = np.unravel_index(neg_idx, f.shape)
        hotspots.append({'theta': float(theta_g[i_neg, j_neg]),
                         'phi': float(phi_g[i_neg, j_neg]),
                         'value': float(f[i_neg, j_neg]), 'sign': 'neg'})
    return hotspots

def cap_enrichment_at_center(beta: np.ndarray,
                             se: np.ndarray,
                             theta: np.ndarray,
                             phi: np.ndarray,
                             phenonames: List[str],
                             center_theta: float,
                             center_phi: float,
                             radius_deg: float,
                             side: str = "auto") -> Dict:
    """Inverse-variance meta-Z within a spherical cap."""
    radius = np.deg2rad(radius_deg)
    dists = np.array([_angdist(float(t), float(p), center_theta, center_phi) for t, p in zip(theta, phi)])
    idx = np.where(dists <= radius)[0]
    if idx.size == 0:
        return {'radius_deg': radius_deg, 'n_in_cap': 0, 'z_meta': 0.0, 'p_two': 1.0, 'p_one': 1.0, 'members': []}

    bcap = beta[idx]; secap = se[idx]; names = [phenonames[j] for j in idx]
    w = 1.0 / np.maximum(secap, 1e-12)**2
    num = float(np.sum(w * bcap)); den = float(np.sqrt(np.sum(w)))
    z_meta = num / den if den > 0 else 0.0
    p_two = 2.0 * (1.0 - norm.cdf(abs(z_meta)))
    if side == "two":
        p_one = p_two
    else:
        sgn = +1.0 if (side == "pos" or (side == "auto" and z_meta >= 0)) else -1.0
        p_one = 1.0 - norm.cdf(sgn * z_meta)

    contrib = (w * bcap) / den if den > 0 else np.zeros_like(bcap)
    zj = bcap / np.maximum(secap, 1e-12)
    rows = []
    for nm, bj, sej, zj_, dj, cj in zip(names, bcap, secap, zj, np.rad2deg(dists[idx]), contrib):
        rows.append({'pheno': nm, 'beta': float(bj), 'se': float(sej),
                     'z': float(zj_), 'dist_deg': float(dj), 'contribution': float(cj)})
    rows = sorted(rows, key=lambda r: abs(r['contribution']), reverse=True)
    return {'radius_deg': radius_deg, 'n_in_cap': int(idx.size),
            'z_meta': float(z_meta), 'p_two': float(p_two), 'p_one': float(p_one), 'members': rows}

def spherical_cap_enrichment(beta: np.ndarray,
                             se: np.ndarray,
                             theta: np.ndarray,
                             phi: np.ndarray,
                             phenonames: List[str],
                             Lmax: int,
                             lam: float,
                             Y_all: Optional[np.ndarray],
                             cols_for_L: Optional[Dict[int, slice]],
                             lap_diag: Optional[np.ndarray],
                             radii_deg: Tuple[int, ...] = (10, 15, 20, 25, 30),
                             take_both_signs: bool = True) -> Dict:
    hotspots = find_hotspots_from_SH(beta, se, theta, phi, Lmax, lam,
                                     Y_all=Y_all, cols_for_L=cols_for_L, lap_diag=lap_diag,
                                     take_both_signs=take_both_signs)
    out = {'hotspots': []}
    for hp in hotspots:
        per_r = []
        for r in radii_deg:
            per_r.append(
                cap_enrichment_at_center(beta, se, theta, phi, phenonames,
                                         center_theta=hp['theta'], center_phi=hp['phi'],
                                         radius_deg=r,
                                         side=("pos" if hp['sign']=="pos" else "neg"))
            )
        best = min(per_r, key=lambda d: d['p_one'])
        out['hotspots'].append({'sign': hp['sign'], 'center': hp, 'best': best, 'all': per_r})
    return out

def great_circle_cap_outline(center_theta: float, center_phi: float, radius_deg: float, n: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    r = np.deg2rad(radius_deg)
    c = _vec_from_angles(center_theta, center_phi)
    ref = np.array([0,0,1], dtype=float)
    if np.allclose(np.abs(np.dot(c, ref)), 1.0, atol=1e-6):
        ref = np.array([0,1,0], dtype=float)
    u = np.cross(c, ref); u /= np.linalg.norm(u)
    v = np.cross(c, u);   v /= np.linalg.norm(v)
    alphas = np.linspace(0, 2*np.pi, n, endpoint=True)
    pts = (np.cos(r) * c[None,:] +
           np.sin(r) * (np.cos(alphas)[:,None] * u[None,:] + np.sin(alphas)[:,None] * v[None,:]))
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    theta = np.arccos(np.clip(z / np.maximum(np.sqrt(x*x+y*y+z*z), 1e-12), -1, 1))
    phi = np.arctan2(y, x); phi[phi < 0] += 2*np.pi
    lon_deg = phi * 180/np.pi
    lat_deg = (np.pi/2 - theta) * 180/np.pi
    return lon_deg, lat_deg

# -----------------------------------------------------------------------------
# Data I/O
# -----------------------------------------------------------------------------
def load_long_tsv(file, pheno_col, var_col, beta_col, se_col) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = pd.read_csv(file, sep=None, engine="python", low_memory=False)  # auto-sep
    if var_col not in df.columns and "rsid" in df.columns:
        var_col = "rsid"
    required = [pheno_col, var_col, beta_col, se_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded file: {missing}")
    df = df[[pheno_col, var_col, beta_col, se_col]].copy()
    df.columns = ["pheno", "variant", "beta", "se"]
    df = df[~(df["beta"].isna() & df["se"].isna())].reset_index(drop=True)
    phenos = sorted(df["pheno"].astype(str).unique().tolist())
    variants = sorted(df["variant"].astype(str).unique().tolist())
    return df, phenos, variants

def build_mats_from_long(df: pd.DataFrame, phenos: List[str], variants: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    k = len(phenos); M = len(variants)
    pidx = {p: j for j, p in enumerate(phenos)}
    vidx = {v: i for i, v in enumerate(variants)}
    betas = np.full((M, k), 0.0, dtype=np.float32)
    ses   = np.full((M, k), np.inf, dtype=np.float32)
    for _, row in df.iterrows():
        i = vidx[str(row["variant"])]; j = pidx[str(row["pheno"])]
        b = row["beta"]; s = row["se"]
        if s is None or not np.isfinite(s) or s <= 0: continue
        if not np.isfinite(b): b = 0.0
        if ses[i, j] == np.inf or s < ses[i, j]:
            betas[i, j] = b; ses[i, j] = s
    return betas, ses

def load_matrix_csvs(beta_file, se_file) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    def read_any(f):
        name = f.name.lower()
        if name.endswith((".tsv", ".txt")):
            return pd.read_csv(f, sep="\t", index_col=0)
        return pd.read_csv(f, index_col=0)
    B = read_any(beta_file); S = read_any(se_file)
    B, S = B.align(S, join="inner", axis=0); B, S = B.align(S, join="inner", axis=1)
    betas = B.values.astype(np.float32); ses = S.values.astype(np.float32)
    variants = B.index.astype(str).tolist(); phenos = B.columns.astype(str).tolist()
    ses[~np.isfinite(ses) | (ses <= 0)] = np.inf; betas[~np.isfinite(betas)] = 0.0
    return betas, ses, phenos, variants

# -----------------------------------------------------------------------------
# FAST analysis (fixed L) — optimized path
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_analysis_fast(betas: np.ndarray,
                      ses: np.ndarray,
                      phenos: List[str],
                      variants: List[str],
                      embedding_method: str = "Fibonacci",
                      L_fixed: int = 16,
                      L0: int = 2,
                      lam: float = 1e-3) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict]:
    M, k = betas.shape
    # Embedding
    if embedding_method == "Fibonacci":
        S = fibonacci_sphere(k)
    else:
        S = data_driven_embedding(betas, ses)
    theta, phi = cart_to_sph(S)

    # Precompute basis once
    Y_all, cols_for_L, lm_list, lap_diag = precompute_sh_basis(theta, phi, L_fixed, dtype=np.complex64)

    rows = []
    # Reduce BLAS oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    for i in range(M):
        beta = betas[i, :]; se = ses[i, :]
        # Fit L_fixed
        a1, SSE1 = ridge_WLS_chol_fast(beta, se, Y_all, cols_for_L[L_fixed], lap_diag, lam=lam)
        # Fit L0 for nested F
        a0, SSE0 = ridge_WLS_chol_fast(beta, se, Y_all, cols_for_L[L0], lap_diag, lam=lam)
        p1 = (L_fixed + 1)**2; p0 = (L0 + 1)**2; k_ = beta.size
        df1 = max(p1 - p0, 1); df2 = max(k_ - p1, 1)
        F = ((SSE0 - SSE1)/df1) / (SSE1/df2) if SSE1 > 0 else np.inf
        p = float(f_dist.sf(F, df1, df2)) if np.isfinite(F) else 0.0

        P = degree_powers(a1, lm_list, L_fixed)
        desc = spectral_descriptors(P); LI = localization_index(P, L0)

        rows.append({"variant": variants[i], "Lmax": L_fixed,
                     "l95": int(desc["l95"]), "lbar": desc["lbar"], "H": desc["H"],
                     "LI_2": LI, "F_highl": float(F), "p_highl": p})

    met = pd.DataFrame(rows)
    met["FDR_sig"] = bh_fdr(met["p_highl"].values, alpha=0.05)
    score = (-np.log10(np.maximum(met["p_highl"].values, 1e-300))) * met["LI_2"].values
    met["score"] = score + (met["FDR_sig"].astype(int) * 1000.0)
    met = met.sort_values(["FDR_sig","score"], ascending=[False, False]).reset_index(drop=True)

    meta = {"theta": theta, "phi": phi, "Y_all_shape": Y_all.shape, "L_fixed": L_fixed}
    return met, theta, phi, S, meta

# -----------------------------------------------------------------------------
# PRECISION analysis (BIC over L) — optional, slower
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_analysis_precision(betas: np.ndarray,
                           ses: np.ndarray,
                           phenos: List[str],
                           variants: List[str],
                           embedding_method: str = "Data-driven",
                           L_grid: Tuple[int, ...] = (6, 8, 10, 12, 14, 16),
                           L0: int = 2,
                           lam: float = 1e-3) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    M, k = betas.shape
    if embedding_method == "Fibonacci":
        S = fibonacci_sphere(k)
    else:
        S = data_driven_embedding(betas, ses)
    theta, phi = cart_to_sph(S)

    Lmax = max(L_grid)
    Y_all, cols_for_L, lm_list, lap_diag = precompute_sh_basis(theta, phi, Lmax, dtype=np.complex64)

    rows = []
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
    for i in range(M):
        beta = betas[i, :]; se = ses[i, :]
        best = None
        for L in L_grid:
            a, SSE = ridge_WLS_chol_fast(beta, se, Y_all, cols_for_L[L], lap_diag, lam=lam)
            p = (L + 1)**2
            bic = k * np.log(max(SSE / max(k,1), 1e-300)) + p * np.log(max(k,1))
            if (best is None) or (bic < best["bic"]):
                best = {"L": L, "a": a, "SSE": SSE, "bic": bic}
        Lbest = best["L"]; a1 = best["a"]; SSE1 = best["SSE"]
        a0, SSE0 = ridge_WLS_chol_fast(beta, se, Y_all, cols_for_L[L0], lap_diag, lam=lam)
        p1 = (Lbest + 1)**2; p0 = (L0 + 1)**2; k_ = beta.size
        df1 = max(p1 - p0, 1); df2 = max(k_ - p1, 1)
        F = ((SSE0 - SSE1)/df1) / (SSE1/df2) if SSE1 > 0 else np.inf
        p = float(f_dist.sf(F, df1, df2)) if np.isfinite(F) else 0.0

        P = degree_powers(a1, lm_list, Lbest)
        desc = spectral_descriptors(P); LI = localization_index(P, L0)

        rows.append({"variant": variants[i], "Lmax": Lbest,
                     "l95": int(desc["l95"]), "lbar": desc["lbar"], "H": desc["H"],
                     "LI_2": LI, "F_highl": float(F), "p_highl": p})

    met = pd.DataFrame(rows)
    met["FDR_sig"] = bh_fdr(met["p_highl"].values, alpha=0.05)
    score = (-np.log10(np.maximum(met["p_highl"].values, 1e-300))) * met["LI_2"].values
    met["score"] = score + (met["FDR_sig"].astype(int) * 1000.0)
    met = met.sort_values(["FDR_sig","score"], ascending=[False, False]).reset_index(drop=True)
    return met, theta, phi, S

# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def make_volcano(met: pd.DataFrame, L0: int = 2) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7,5))
    x = met[f"LI_{L0}"].values if f"LI_{L0}" in met.columns else met["LI_2"].values
    y = -np.log10(np.maximum(met["p_highl"].values, 1e-300))
    colors = np.where(met["FDR_sig"].values, "tab:red", "tab:blue")
    ax.scatter(x, y, s=18, alpha=0.85, c=colors)
    for r in range(min(20, len(met))):
        ax.annotate(met.loc[r, "variant"], (x[r], y[r]), fontsize=7, xytext=(3,3), textcoords='offset points')
    ax.set_xlabel(f"Localization Index LI (l>{L0})")
    ax.set_ylabel(r"$-\log_{10} p_{\mathrm{high}\ell}$")
    ax.set_title("Localization Volcano")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig

def make_l95_hist(met: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7,4))
    bins = np.arange(int(met["l95"].max() if len(met) else 0)+2) - 0.5
    ax.hist(met["l95"].values, bins=bins, color="tab:gray", edgecolor="white")
    ax.set_xlabel(r"$l_{95}$"); ax.set_ylabel("Count"); ax.set_title(r"Distribution of $l_{95}$")
    fig.tight_layout(); return fig

def render_map(variant: str, met: pd.DataFrame, betas: np.ndarray, ses: np.ndarray,
               theta: np.ndarray, phi: np.ndarray,
               Y_all: Optional[np.ndarray], cols_for_L: Optional[Dict[int, slice]], lap_diag: Optional[np.ndarray],
               overlay_cap: Optional[Dict] = None, nlon=360, nlat=181) -> Optional[plt.Figure]:
    idx = met.index[met["variant"] == variant]
    if len(idx) == 0: return None
    i = int(idx[0]); Lmax = int(met.loc[i, "Lmax"])
    beta = betas[i, :]; se = ses[i, :]
    if Y_all is not None:
        a, SSE = ridge_WLS_chol_fast(beta, se, Y_all, cols_for_L[Lmax], lap_diag, lam=1e-3)
    else:
        # fallback: compute Y locally (slower)
        Y_local, lm_local, lap_local = precompute_sh_basis(theta, phi, Lmax)[0:3] + (None,)
        a, SSE = ridge_WLS_chol_fast(beta, se, Y_local, slice(0, (Lmax+1)**2), lap_local, lam=1e-3)

    fgrid, lon_deg, lat_deg = evaluate_on_grid(a, Lmax, n_lon=nlon, n_lat=nlat)
    A = normalize01(fgrid)
    fig = plt.figure(figsize=(7.8, 3.9)); ax = fig.add_subplot(111)
    pcm = ax.pcolormesh(lon_deg, lat_deg, A, cmap="plasma", shading="nearest", vmin=0, vmax=1)
    fig.colorbar(pcm, ax=ax, label="Normalized effect")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    LIcol = "LI_2" if "LI_2" in met.columns else [c for c in met.columns if c.startswith("LI_")][0]
    ax.set_title(f"{variant} (Lmax={Lmax}, l95={int(met.loc[i,'l95'])}, LI={met.loc[i,LIcol]:.2f})")
    ax.set_ylim(-90, 90); ax.set_xlim(0, 360)
    if overlay_cap is not None:
        lon_cap, lat_cap = great_circle_cap_outline(overlay_cap["theta"], overlay_cap["phi"], overlay_cap["radius_deg"], n=720)
        ax.plot(lon_cap, lat_cap, lw=2.0, color="white", alpha=0.9)
        ax.plot(lon_cap, lat_cap, lw=1.0, color="black", alpha=0.9)
    fig.tight_layout(); return fig

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🧭 SH-PheWAS Explorer — Fast")
st.write(
    "Analyze and visualize localized cross-phenotype effects with spherical harmonics. "
    "This optimized app precomputes the SH basis and supports a **Fast (fixed L)** mode suitable for very large matrices (e.g., 13k×3k)."
)

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.selectbox("Mode", ["Fast (fixed L)", "Precision (BIC over L)"], index=0)
    embedding_method = st.selectbox("Phenotype embedding", ["Data-driven", "Fibonacci"], index=0 if mode=="Precision (BIC over L)" else 1)
    if mode == "Fast (fixed L)":
        L_fixed = st.slider("Fixed L (Fast mode)", 8, 24, 16, 2)
    else:
        L_grid_min = st.slider("Min L (Precision)", 2, 10, 6, 2)
        L_grid_max = st.slider("Max L (Precision)", 8, 24, 16, 2)
        if L_grid_min >= L_grid_max:
            st.warning("Min L must be < Max L; adjusting.")
            L_grid_min = min(L_grid_min, L_grid_max - 2)
        L_grid = tuple(range(L_grid_min, L_grid_max + 1, 2))
    L0 = st.number_input("Low-degree cutoff L0", value=2, min_value=0, max_value=8, step=1)
    lam = st.number_input("Ridge λ (Laplacian)", value=1e-3, min_value=1e-6, max_value=1e-1, step=1e-3, format="%.4f")
    st.caption("Tip: Fibonacci avoids SVD in embedding; best for huge k.")

tabs = st.tabs(["Upload & Analyze", "Explore Results", "Cap Enrichment", "About"])

# --- Upload & Analyze ---
with tabs[0]:
    st.subheader("Upload data")
    input_style = st.radio("Input style:", ["Long table (pheno/variant/beta/se)", "Matrices: betas + SEs"], index=0)
    betas = ses = phenos = variants = None

    if input_style == "Long table (pheno/variant/beta/se)":
        tsv = st.file_uploader("Upload TSV/CSV (auto-delimiter)", type=["tsv","txt","csv"])
        c1, c2, c3, c4 = st.columns(4)
        with c1: pheno_col = st.text_input("pheno col", value="pheno")
        with c2: var_col   = st.text_input("variant col", value="variant")
        with c3: beta_col  = st.text_input("beta col", value="beta_add")
        with c4: se_col    = st.text_input("se col", value="sebeta_add")
        if tsv is not None:
            try:
                df_long, phenos, variants = load_long_tsv(tsv, pheno_col, var_col, beta_col, se_col)
                betas, ses = build_mats_from_long(df_long, phenos, variants)
                st.success(f"Loaded {len(variants)} variants × {len(phenos)} phenotypes.")
            except Exception as e:
                st.error(f"Problem reading file: {e}")
    else:
        beta_file = st.file_uploader("Upload betas (CSV/TSV, rows=variants, cols=phenotypes)", type=["csv","tsv","txt"])
        se_file   = st.file_uploader("Upload SEs (CSV/TSV, rows=variants, cols=phenotypes)", type=["csv","tsv","txt"])
        if beta_file is not None and se_file is not None:
            try:
                betas, ses, phenos, variants = load_matrix_csvs(beta_file, se_file)
                st.success(f"Loaded {len(variants)} variants × {len(phenos)} phenotypes.")
            except Exception as e:
                st.error(f"Problem reading matrices: {e}")

    run = st.button("Run analysis", type="primary", use_container_width=True)
    if run:
        if betas is None or ses is None:
            st.error("Please provide input files first.")
        else:
            with st.spinner("Fitting spherical harmonics and computing metrics..."):
                if mode == "Fast (fixed L)":
                    met, theta, phi, S, meta = run_analysis_fast(
                        betas=betas, ses=ses, phenos=phenos, variants=variants,
                        embedding_method=embedding_method,
                        L_fixed=int(L_fixed), L0=int(L0), lam=float(lam)
                    )
                    # Keep precomputed basis for cap enrichment / maps
                    Y_all, cols_for_L, lm_list, lap_diag = precompute_sh_basis(theta, phi, int(L_fixed))
                else:
                    met, theta, phi, S = run_analysis_precision(
                        betas=betas, ses=ses, phenos=phenos, variants=variants,
                        embedding_method=embedding_method,
                        L_grid=tuple(L_grid), L0=int(L0), lam=float(lam)
                    )
                    # Precompute up to max L for overlays
                    Y_all, cols_for_L, lm_list, lap_diag = precompute_sh_basis(theta, phi, max(L_grid))

            st.success("Analysis complete.")
            st.session_state.update(dict(
                met=met, betas=betas, ses=ses, phenos=phenos, variants=variants,
                theta=theta, phi=phi, Y_all=Y_all, cols_for_L=cols_for_L, lap_diag=lap_diag
            ))
            st.download_button(
                "Download metrics CSV",
                data=met.to_csv(index=False).encode("utf-8"),
                file_name="metrics.csv", mime="text/csv",
                use_container_width=True
            )

# --- Explore Results ---
with tabs[1]:
    st.subheader("Explore Results")
    if "met" not in st.session_state:
        st.info("Run an analysis first.")
    else:
        met = st.session_state["met"]
        betas = st.session_state["betas"]; ses = st.session_state["ses"]
        theta = st.session_state["theta"]; phi = st.session_state["phi"]
        Y_all = st.session_state["Y_all"]; cols_for_L = st.session_state["cols_for_L"]; lap_diag = st.session_state["lap_diag"]

        c0, c1 = st.columns([1,1])
        with c0: st.pyplot(make_volcano(met, L0=int(L0)), use_container_width=True)
        with c1: st.pyplot(make_l95_hist(met), use_container_width=True)

        st.markdown("### Metrics table")
        with st.expander("Show table"):
            st.dataframe(met, use_container_width=True, height=420)

        st.markdown("### Variant map")
        variant_choice = st.selectbox("Pick a variant", met["variant"].tolist(), index=0 if len(met) else None)
        if variant_choice:
            fig_map = render_map(variant_choice, met, betas, ses, theta, phi,
                                 Y_all=Y_all, cols_for_L=cols_for_L, lap_diag=lap_diag)
            if fig_map is not None:
                st.pyplot(fig_map, use_container_width=True)
                buf = io.BytesIO(); fig_map.savefig(buf, format="png", dpi=200)
                st.download_button("Download map PNG", data=buf.getvalue(),
                                   file_name=f"{variant_choice}_map.png", mime="image/png")

        st.markdown("### Quick filters")
        fdr_only = st.checkbox("FDR-significant only", value=True)
        l95_min = st.slider("Min l95", 0, int(met["l95"].max() if len(met) else 0), value=min(10, int(met["l95"].max() if len(met) else 0)))
        li_min  = st.slider("Min LI_2", 0.0, 1.0, 0.7, 0.05)
        dfq = met.copy()
        if fdr_only: dfq = dfq[dfq["FDR_sig"]]
        dfq = dfq[(dfq["l95"] >= l95_min) & (dfq["LI_2"] >= li_min)]
        st.write(f"{len(dfq)} variants pass filters.")
        st.dataframe(dfq, use_container_width=True, height=300)
        st.download_button("Download filtered CSV", data=dfq.to_csv(index=False).encode("utf-8"),
                           file_name="metrics_filtered.csv", mime="text/csv")

# --- Cap Enrichment ---
with tabs[2]:
    st.subheader("Spherical-cap enrichment")
    if "met" not in st.session_state:
        st.info("Run an analysis first.")
    else:
        met = st.session_state["met"]; betas = st.session_state["betas"]; ses = st.session_state["ses"]
        theta = st.session_state["theta"]; phi = st.session_state["phi"]; phenos = st.session_state["phenos"]
        Y_all = st.session_state["Y_all"]; cols_for_L = st.session_state["cols_for_L"]; lap_diag = st.session_state["lap_diag"]

        cA, cB = st.columns([2,1])
        with cA:
            variant_cap = st.selectbox("Variant", met["variant"].tolist(), index=0)
        with cB:
            radii_sel = st.multiselect("Radii (degrees)", [10,12,15,20,25,30,35], default=[10,15,20,25,30])
            topN = st.number_input("Top phenotypes to display", value=20, min_value=5, max_value=200, step=5)
            both_signs = st.checkbox("Analyze both signs (max & min)", value=True)

        run_cap = st.button("Run cap enrichment", type="primary")
        if run_cap and variant_cap:
            i = int(np.where(met["variant"].values == variant_cap)[0][0])
            Lmax = int(met.loc[i, "Lmax"])
            res = spherical_cap_enrichment(
                beta=betas[i, :], se=ses[i, :],
                theta=theta, phi=phi, phenonames=phenos,
                Lmax=Lmax, lam=float(lam),
                Y_all=Y_all, cols_for_L=cols_for_L, lap_diag=lap_diag,
                radii_deg=tuple(radii_sel) if radii_sel else (15, 25),
                take_both_signs=both_signs
            )
            for h in res["hotspots"]:
                st.markdown(f"#### {'Positive' if h['sign']=='pos' else 'Negative'} hotspot")
                b = h["best"]
                st.write(f"Best radius = **{b['radius_deg']}°**, in-cap = **{b['n_in_cap']}**, "
                         f"meta-z = **{b['z_meta']:.2f}**, one-sided p = **{b['p_one']:.2e}**, two-sided p = **{b['p_two']:.2e}**")
                overlay = {"theta": h["center"]["theta"], "phi": h["center"]["phi"], "radius_deg": b["radius_deg"]}
                fig_map = render_map(variant_cap, met, betas, ses, theta, phi,
                                     Y_all=Y_all, cols_for_L=cols_for_L, lap_diag=lap_diag,
                                     overlay_cap=overlay)
                st.pyplot(fig_map, use_container_width=True)
                dfm = pd.DataFrame(b["members"])
                if len(dfm):
                    st.dataframe(dfm.head(int(topN)), use_container_width=True, height=360)
                    st.download_button(
                        f"Download members CSV ({h['sign']})",
                        data=dfm.to_csv(index=False).encode("utf-8"),
                        file_name=f"{variant_cap}_cap_members_{h['sign']}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No phenotypes found in the best-radius cap.")

# --- About ---
with tabs[3]:
    st.subheader("About")
    st.markdown(
        """
**Optimizations in this build**
- Precomputes the SH basis **once** (cached).
- **Fast mode** uses a single fixed L per variant (e.g., L=16).
- Complex Hermitian **Cholesky** ridge solver (no SVD in solver).
- Float32/complex64 to reduce memory.
- Avoids BLAS oversubscription.

Switch to **Precision (BIC over L)** for smaller datasets when you want per-variant L selection.
        """
    )
