"""
fast_eda.py — Fast EDA using vectorized pandas (no row-by-row loops).
Handles 2M+ GT rows efficiently.
"""
import json, sys, logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TRAIN = Path(r"D:\Deksto-1\Desktop\Hackthons\Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource\dataset\train")
OUT   = Path(r"D:\Deksto-1\Desktop\Hackthons\Projects\Amazon_ML\amazon_ml_solution\experiments")
OUT.mkdir(parents=True, exist_ok=True)

log.info("Loading files ...")
s1 = pd.read_csv(TRAIN/"train_source1.tsv", sep="\t", dtype=str).fillna("")
s2 = pd.read_csv(TRAIN/"train_source2.tsv", sep="\t", dtype=str).fillna("")
s3 = pd.read_csv(TRAIN/"train_source3.tsv", sep="\t", dtype=str).fillna("")
gt = pd.read_csv(TRAIN/"train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

log.info(f"S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  GT={len(gt):,}")

# Vectorized GT parse
log.info("Parsing ground truth (vectorized) ...")
gt["n_matches"] = gt["matched_entity_ids"].apply(lambda x: len(x.split(",")) if x.strip() else 0)
n_total     = len(gt)
n_singletons= (gt["n_matches"] == 0).sum()
max_matches = gt["n_matches"].max()
mean_match  = gt[gt["n_matches"]>0]["n_matches"].mean()

log.info(f"Singletons: {n_singletons:,} / {n_total:,} = {n_singletons/n_total:.1%}")
log.info(f"Max matches per S1: {max_matches}")
log.info(f"Avg matches (non-zero): {mean_match:.2f}")
log.info(f"Country dist S1: {s1['country'].value_counts().to_dict()}")
log.info(f"Country dist S2: {s2['country'].value_counts().to_dict()}")
log.info(f"Country dist S3: {s3['country'].value_counts().to_dict()}")

# Match distribution
dist = gt["n_matches"].value_counts().sort_index().to_dict()
log.info(f"Match distribution (count: n_entities): {dict(list(dist.items())[:10])}")

# S2 vs S3 contribution
has_s2 = gt["matched_entity_ids"].str.contains("S2-", na=False)
has_s3 = gt["matched_entity_ids"].str.contains("S3-", na=False)
n_has_match = (gt["n_matches"] > 0).sum()
log.info(f"Has S2 match: {has_s2.sum():,}  Has S3 match: {has_s3.sum():,}  Both: {(has_s2 & has_s3).sum():,}")

# Null analysis
for name, df in [("S1",s1),("S2",s2)]:
    empty_name = (df["business_name"].str.strip()=="").sum()
    empty_addr = (df["business_address"].str.strip()=="").sum()
    log.info(f"{name}: empty_name={empty_name:,}  empty_addr={empty_addr:,}")

report = {
    "s1":len(s1), "s2":len(s2), "s3":len(s3),
    "gt_rows":len(gt),
    "n_singletons":int(n_singletons),
    "singleton_rate":round(float(n_singletons/n_total),4),
    "max_matches":int(max_matches),
    "mean_matches_nonzero":round(float(mean_match),4),
    "country_s1": s1["country"].value_counts().to_dict(),
    "country_s2": s2["country"].value_counts().to_dict(),
    "country_s3": s3["country"].value_counts().to_dict(),
    "match_dist": {str(k):int(v) for k,v in list(dist.items())[:20]},
    "s2_matches":int(has_s2.sum()), "s3_matches":int(has_s3.sum()),
    "both_s2_s3":int((has_s2&has_s3).sum()),
}
with open(OUT/"eda_report.json","w") as f:
    json.dump(report, f, indent=2)
log.info(f"EDA done → {OUT/'eda_report.json'}")
