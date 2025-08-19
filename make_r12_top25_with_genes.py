# save as: make_r12_top25_with_genes.py
import pandas as pd

# Inputs
METRICS = "r12_metrics.csv"  # produced by the SH-PheWAS script
TSV     = "R12_coding_variant_results_1e-5_annotated.tsv"  # your original TSV with gene_most_severe
OUT     = "r12_metrics_top25.csv"

# Load metrics
met = pd.read_csv(METRICS)

# Load TSV with gene labels; keep only needed columns and dedupe per variant
df = pd.read_csv(TSV, sep="\t", low_memory=False)
use_cols = [c for c in ["variant","gene_most_severe","most_severe","rsid"] if c in df.columns]
df = df[use_cols].copy()

# If some files used rsid instead of variant, fill a 'variant' column with rsid
if "variant" not in df.columns and "rsid" in df.columns:
    df["variant"] = df["rsid"]

# Resolve multiple rows per variant -> pick the modal gene label (fallback: first non-null)
gene_map = (
    df.dropna(subset=["variant"])
      .assign(gene_most_severe=df["gene_most_severe"].fillna(""))
      .groupby("variant")["gene_most_severe"]
      .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iat[0] if s.dropna().size else ""))
      .rename("gene")
      .reset_index()
)

# Merge into metrics
met2 = met.merge(gene_map, on="variant", how="left")

# Rank: FDR first, then score
met2 = met2.sort_values(["FDR_sig","score"], ascending=[False, False])
top25 = met2.head(25)[["variant","gene","Lmax","l95","lbar","H","LI_2","p_highl","FDR_sig","score"]]

# Write
top25.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(top25)} rows.")
