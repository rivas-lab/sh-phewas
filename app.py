# streamlit run app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import f as f_dist
from typing import Tuple, List, Dict, Optional

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="SH-PheWAS Explorer",
    page_icon="🧭",
    layout="wide"
)

# ----------------------------
# Helpers: sphere + SH + stats
# ----------------------------

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
    Phenotype embedding via truncated SVD-style projection (without requiring sklearn).
    For robustness (no SVD library dependency), we use a simple PCA via np.linalg.svd on (k x M),
    but allow user to switch to Fibonacci to avoid SVD entirely.
    """
    M, k = betas.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        z = betas / ses
    z[np.isinf(z)] = 0.0
    Z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).T  # (k x M)

    # Degenerate?
    if k < 3 or M == 0:
        return fibonacci_sphere(k)

    try:
        # Compute top-3 right singular vectors of Z via economical SVD
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        X = U[:, :3]  # (k x 3)
        # Normalize onto S^2
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Sph = X / norms
        return Sph
    except Exception:
        # Fallback: Fibonacci if SVD fails
        return fibonacci_sphere(k)

def cart_to_sph(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = xyz.T
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-12), -1, 1))
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2*np.pi
    return theta, phi

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

def ridge_WLS_chol(
    y: np.ndarray,
    Y: np.ndarray,
    se: np.ndarray,
    lm_list: List[Tuple[int,int]],
    lam: float = 1e-3,
    w_clip: float = 1e6,
    jitter0: float = 1e-8,
    max_tries: int = 6
) -> Tuple[np.ndarray, float]:
    """
    Weighted ridge (Laplacian) solved by Cholesky; strictly SPD via diagonal jitter.
    Returns (a_hat complex, SSE).
    """
    w = 1.0 / np.maximum(se, 1e-12)**2
    w[~np.isfinite(w)] = 0.0
    w = np.minimum(w, w_clip)
    sqrt_w = np.sqrt(w)

    Yw = Y * sqrt_w[:, None]
    yw = y * sqrt_w

    p = Y.shape[1]
    lap_diag = np.array([l*(l+1) if l > 0 else 0.0 for (l, m) in lm_list], dtype=float)
    Pbig_diag = np.concatenate([lap_diag, lap_diag])

    Ywr = np.hstack([Yw.real, -Yw.imag])   # (k x 2p)
    Ywi = np.hstack([Yw.imag,  Yw.real])   # (k x 2p)
    Ybig = np.vstack([Ywr, Ywi])           # (2k x 2p)
    ybig = np.hstack([yw.real, yw.imag])   # (2k,)

    A = Ybig.T @ Ybig
    A += lam * np.diag(Pbig_diag)
    A += (lam * 1e-6) * np.eye(A.shape[0])  # tiny global ridge for SPD
    b = Ybig.T @ ybig

    jitter = jitter0
    for _ in range(max_tries):
        try:
            c, lower = cho_factor(A + jitter * np.eye(A.shape[0]), lower=False, check_finite=False)
            ahat_big = cho_solve((c, lower), b, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        c, lower = cho_factor(A + (10.0 * jitter + 1e-4) * np.eye(A.shape[0]), lower=False, check_finite=False)
        ahat_big = cho_solve((c, lower), b, check_finite=False)

    a = ahat_big[:p] + 1j * ahat_big[p:]

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
    H = float(ent / np.log(max(len(P), 2)))
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
    a0, SSE0 = ridge_WLS_chol(beta, Y0, se, lm0, lam=lam)
    Y1, lm1 = design_matrix(theta, phi, L1)
    a1, SSE1 = ridge_WLS_chol(beta, Y1, se, lm1, lam=lam)
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
    pvals = np.asarray(pvals)
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

def normalize01(A: np.ndarray) -> np.ndarray:
    mn, mx = np.min(A), np.max(A)
    if mx > mn:
        return (A - mn) / (mx - mn)
    return A

# ----------------------------
# Data ingestion
# ----------------------------

def load_long_tsv(file, pheno_col, var_col, beta_col, se_col) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = pd.read_csv(file, sep="\t", low_memory=False)
    # Fallbacks
    if var_col not in df.columns and "rsid" in df.columns:
        var_col = "rsid"
    required = [pheno_col, var_col, beta_col, se_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded TSV: {missing}")

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
    betas = np.full((M, k), 0.0)
    ses   = np.full((M, k), np.inf)
    # Prefer smallest SE per (variant,pheno)
    for _, row in df.iterrows():
        i = vidx[str(row["variant"])]
        j = pidx[str(row["pheno"])]
        b = row["beta"]; s = row["se"]
        if s is None or not np.isfinite(s) or s <= 0:
            continue
        if not np.isfinite(b):
            b = 0.0
        if ses[i, j] == np.inf or s < ses[i, j]:
            betas[i, j] = b
            ses[i, j]   = s
    return betas, ses

def load_matrix_csvs(beta_file, se_file) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Expect CSV/TSV with header=phenotypes, index=variants.
    """
    # Support both CSV and TSV by sniffing delimiter
    def read_any(f):
        name = f.name.lower()
        if name.endswith(".tsv") or name.endswith(".txt"):
            return pd.read_csv(f, sep="\t", index_col=0)
        return pd.read_csv(f, index_col=0)

    B = read_any(beta_file)
    S = read_any(se_file)
    # Align
    B, S = B.align(S, join="inner", axis=0)
    B, S = B.align(S, join="inner", axis=1)
    betas = B.values.astype(float)
    ses = S.values.astype(float)
    variants = B.index.astype(str).tolist()
    phenos = B.columns.astype(str).tolist()
    # Clean ses
    ses[~np.isfinite(ses) | (ses <= 0)] = np.inf
    betas[~np.isfinite(betas)] = 0.0
    return betas, ses, phenos, variants

# ----------------------------
# Analysis pipeline
# ----------------------------

@st.cache_data(show_spinner=False)
def run_analysis(
    betas: np.ndarray,
    ses: np.ndarray,
    phenos: List[str],
    variants: List[str],
    embedding_method: str = "Data-driven",
    L_grid: Tuple[int, ...] = (6, 8, 10, 12, 14, 16),
    L0: int = 2,
    lam: float = 1e-3
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    M, k = betas.shape

    # Embedding
    if embedding_method == "Fibonacci":
        S = fibonacci_sphere(k)
    else:
        S = data_driven_embedding(betas, ses)
    theta, phi = cart_to_sph(S)

    rows = []
    for i in range(M):
        beta = betas[i, :]
        se   = ses[i, :]

        # BIC over L_grid
        best = None
        for L in L_grid:
            Y, lm = design_matrix(theta, phi, L)
            a, SSE = ridge_WLS_chol(beta, Y, se, lm, lam=lam)
            p = (L + 1)**2
            bic = k * np.log(max(SSE / max(k, 1), 1e-300)) + p * np.log(max(k,1))
            if (best is None) or (bic < best["bic"]):
                best = {"L": L, "a": a, "Y": Y, "lm": lm, "SSE": SSE, "bic": bic}

        Lmax = best["L"]
        a    = best["a"]; lm = best["lm"]
        P = degree_powers(a, lm, Lmax)
        desc = spectral_descriptors(P)
        LI = localization_index(P, L0)

        # high-l test
        F, pval = nested_F_test(beta, se, theta, phi, L0=L0, L1=Lmax, lam=lam)

        rows.append({
            "variant": variants[i],
            "Lmax": Lmax,
            "l95": int(desc["l95"]),
            "lbar": desc["lbar"],
            "H": desc["H"],
            f"LI_{L0}": LI,
            "F_highl": F,
            "p_highl": pval
        })

    met = pd.DataFrame(rows)
    met["FDR_sig"] = bh_fdr(met["p_highl"].values, alpha=0.05)
    score = (-np.log10(np.maximum(met["p_highl"].values, 1e-300))) * met[f"LI_{L0}"].values
    met["score"] = score + (met["FDR_sig"].astype(int) * 1000.0)
    met = met.sort_values(["FDR_sig", "score"], ascending=[False, False]).reset_index(drop=True)

    return met, theta, phi, S

def make_volcano(met: pd.DataFrame, L0: int = 2) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7,5))
    x = met[f"LI_{L0}"].values
    y = -np.log10(np.maximum(met["p_highl"].values, 1e-300))
    colors = np.where(met["FDR_sig"].values, "tab:red", "tab:blue")
    ax.scatter(x, y, s=18, alpha=0.85, c=colors)
    # annotate top 20
    for r in range(min(20, len(met))):
        ax.annotate(met.loc[r, "variant"], (x[r], y[r]), fontsize=7, xytext=(3,3), textcoords='offset points')
    ax.set_xlabel(f"Localization Index LI (l>{L0})")
    ax.set_ylabel(r"$-\log_{10} p_{\mathrm{high}\ell}$")
    ax.set_title("Localization Volcano")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def make_l95_hist(met: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7,4))
    bins = np.arange(met["l95"].max()+2) - 0.5 if len(met) else np.arange(1) - 0.5
    ax.hist(met["l95"].values, bins=bins, color="tab:gray", edgecolor="white")
    ax.set_xlabel(r"$l_{95}$")
    ax.set_ylabel("Count")
    ax.set_title(r"Distribution of $l_{95}$")
    fig.tight_layout()
    return fig

def render_map(variant: str, met: pd.DataFrame, betas: np.ndarray, ses: np.ndarray,
               theta: np.ndarray, phi: np.ndarray, nlon=360, nlat=181) -> plt.Figure:
    # Find variant index
    idx = met.index[met["variant"] == variant]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    Lmax = int(met.loc[i, "Lmax"])
    beta = betas[i, :]
    se   = ses[i, :]

    Y, lm = design_matrix(theta, phi, Lmax)
    a, SSE = ridge_WLS_chol(beta, Y, se, lm, lam=1e-3)
    fgrid, lon_deg, lat_deg = evaluate_on_grid(a, Lmax, n_lon=nlon, n_lat=nlat)
    A = normalize01(fgrid)

    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_subplot(111)
    pcm = ax.pcolormesh(lon_deg, lat_deg, A, cmap="plasma", shading="nearest", vmin=0, vmax=1)
    fig.colorbar(pcm, ax=ax, label="Normalized effect")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    title = f"{variant} (Lmax={Lmax}, l95={int(met.loc[i,'l95'])}, LI={met.loc[i,'LI_2']:.2f})" if "LI_2" in met.columns else f"{variant}"
    ax.set_title(title)
    ax.set_ylim(-90, 90); ax.set_xlim(0, 360)
    fig.tight_layout()
    return fig

# ----------------------------
# UI
# ----------------------------

st.title("🧭 SH-PheWAS Explorer")
st.write(
    "Analyze and visualize localized effects in PheWAS summary data using spherical harmonics. "
    "Upload either a long TSV (pheno/variant/beta/se) or an n×k pair of matrices (betas + standard errors). "
    "Explore volcano plots, $l_{95}$ histograms, and per-variant maps."
)

with st.sidebar:
    st.header("⚙️ Settings")
    embedding_method = st.selectbox("Phenotype embedding", ["Data-driven", "Fibonacci"], index=0)
    L_grid_max = st.slider("Max L in grid", min_value=8, max_value=24, value=16, step=2)
    L_grid_min = st.slider("Min L in grid", min_value=2, max_value=10, value=6, step=2)
    if L_grid_min >= L_grid_max:
        st.warning("Min L must be < Max L; adjusting.")
        L_grid_min = min(L_grid_min, L_grid_max-2)
    L_grid = tuple(range(L_grid_min, L_grid_max+1, 2))
    L0 = st.number_input("Low-degree cutoff L0", value=2, min_value=0, max_value=8, step=1)
    lam = st.number_input("Ridge λ (Laplacian)", value=1e-3, min_value=1e-6, max_value=1e-1, step=1e-3, format="%.4f")

    st.markdown("---")
    st.caption("Tip: choose Fibonacci if you want to avoid SVD entirely in the embedding step.")

tabs = st.tabs(["Upload & Analyze", "Explore Results", "About"])

with tabs[0]:
    st.subheader("Upload data")
    st.write("Choose an input style, provide files, and click **Run analysis**.")

    input_style = st.radio("Input style:", ["Long TSV (pheno/variant/beta/se)", "Matrices: betas + SEs"], index=0)

    betas = ses = None
    phenos = variants = None
    met = None
    theta = phi = S = None

    if input_style == "Long TSV (pheno/variant/beta/se)":
        tsv = st.file_uploader("Upload long-format TSV", type=["tsv","txt","csv"])
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
                st.error(f"Problem reading TSV: {e}")

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
                met, theta, phi, S = run_analysis(
                    betas=betas, ses=ses,
                    phenos=phenos, variants=variants,
                    embedding_method=embedding_method,
                    L_grid=tuple(L_grid),
                    L0=int(L0),
                    lam=float(lam)
                )
            st.success("Analysis complete.")
            st.session_state["met"] = met
            st.session_state["betas"] = betas
            st.session_state["ses"] = ses
            st.session_state["phenos"] = phenos
            st.session_state["variants"] = variants
            st.session_state["theta"] = theta
            st.session_state["phi"] = phi

            # Offer download
            csv_bytes = met.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download metrics CSV",
                data=csv_bytes,
                file_name="metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

with tabs[1]:
    st.subheader("Explore Results")
    # Require prior run
    if "met" not in st.session_state:
        st.info("Run an analysis in the 'Upload & Analyze' tab first.")
    else:
        met = st.session_state["met"]
        betas = st.session_state["betas"]
        ses = st.session_state["ses"]
        theta = st.session_state["theta"]
        phi = st.session_state["phi"]

        c0, c1 = st.columns([1,1])
        with c0:
            fig_v = make_volcano(met, L0=L0)
            st.pyplot(fig_v, use_container_width=True)
        with c1:
            fig_h = make_l95_hist(met)
            st.pyplot(fig_h, use_container_width=True)

        st.markdown("### Metrics table")
        with st.expander("Show table (sortable / filterable)"):
            st.dataframe(met, use_container_width=True, height=420)

        # Variant picker and map
        st.markdown("### Variant map")
        top_default = met.iloc[0]["variant"] if len(met) else ""
        variant_choice = st.selectbox("Pick a variant to map", met["variant"].tolist(), index=0 if len(met) else None)
        if variant_choice:
            fig_map = render_map(variant_choice, met, betas, ses, theta, phi)
            if fig_map is not None:
                st.pyplot(fig_map, use_container_width=True)
                buf = io.BytesIO()
                fig_map.savefig(buf, format="png", dpi=200)
                st.download_button("Download map PNG", data=buf.getvalue(), file_name=f"{variant_choice}_map.png", mime="image/png")

        # Quick filters
        st.markdown("### Quick filters")
        fdr_only = st.checkbox("Show FDR-significant only", value=True)
        l95_min = st.slider("Min l95", 0, int(met["l95"].max() if len(met) else 0), value=min(10, int(met["l95"].max() if len(met) else 0)))
        li_min = st.slider("Min LI_2", 0.0, 1.0, 0.7, 0.05)
        dfq = met.copy()
        if fdr_only:
            dfq = dfq[dfq["FDR_sig"] == True]
        if "LI_2" in dfq.columns:
            dfq = dfq[(dfq["l95"] >= l95_min) & (dfq["LI_2"] >= li_min)]
        st.write(f"{len(dfq)} variants pass filters.")
        st.dataframe(dfq, use_container_width=True, height=300)

        # Download filtered
        if len(dfq):
            st.download_button(
                "Download filtered metrics CSV",
                data=dfq.to_csv(index=False).encode("utf-8"),
                file_name="metrics_filtered.csv",
                mime="text/csv",
            )

with tabs[2]:
    st.subheader("About this app")
    st.markdown(
        """
**SH-PheWAS Explorer** fits spherical harmonics to variant-by-phenotype effect profiles.

- **Rotation-invariant diagnostics**: degree power, \(l_{95}\), entropy \(H\), localization index \(\mathrm{LI}_{>L_0}\).
- **Inference**: nested \(F\)-test for high-degree structure, BH-FDR across variants.
- **Embedding**: Data-driven on phenotype \(z\)-profiles or uniform Fibonacci sphere.
- **Solver**: robust Cholesky ridge with Laplacian penalty (no SVD in the solver).

*Tip:* For large \(k\) or limited CPU, lower the max \(L\) or switch to Fibonacci embedding.
        """
    )
