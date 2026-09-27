"""
Run inference over the FULL candidate_pairs.tsv (train or test) in chunks
and produce matching_results.tsv in the exact competition format.

This is separate from training: every candidate pair must be scored here
(unlike training, where we subsampled negatives), but it's still done in
chunks so memory stays bounded regardless of file size.

Usage:
    python predict.py \
        --source1 dataset/test/test_source1.tsv \
        --source2 dataset/test/test_source2.tsv \
        --source3 dataset/test/test_source3.tsv \
        --candidates output/candidate_pairs.tsv \
        --model model.json \
        --threshold-file threshold.txt \
        --out output/matching_results.tsv
"""

import argparse

import pandas as pd
import xgboost as xgb

from featurize import NameTfidf, featurize_pair, normalize_name
from build_training_set import load_records_as_dict, fit_tfidf_sample
from train_model import FEATURE_COLUMNS


def predict(
    source1_path,
    source2_path,
    source3_path,
    candidates_path,
    model_path,
    threshold_file,
    out_path,
    chunksize=100_000,
):
    print("Loading source records...")
    s1 = load_records_as_dict(source1_path)
    s2 = load_records_as_dict(source2_path)
    s3 = load_records_as_dict(source3_path)

    def lookup_candidate(cand_id):
        if cand_id.startswith("S2-"):
            return s2.get(cand_id)
        if cand_id.startswith("S3-"):
            return s3.get(cand_id)
        return None

    print("Fitting TF-IDF on a name sample (same procedure as training)...")
    tfidf = fit_tfidf_sample(s1, s2, s3)

    print("Loading model + threshold...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(threshold_file) as f:
        threshold = float(f.read().strip())
    print(f"Using threshold: {threshold:.3f}")

    seen_s1_ids = set()
    first_write = True

    for chunk in pd.read_csv(
        candidates_path, sep="\t", dtype=str, chunksize=chunksize
    ):
        chunk = chunk.fillna("")
        output_rows = []

        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            seen_s1_ids.add(s1_id)
            s1_record = s1.get(s1_id)
            if s1_record is None:
                output_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
                continue

            cand_str = row["candidate_entity_ids"]
            if not cand_str:
                output_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
                continue

            candidates = cand_str.split(",")
            feature_rows = []
            cand_ids = []
            for cand_id in candidates:
                cand_record = lookup_candidate(cand_id)
                if cand_record is None:
                    continue
                feats = featurize_pair(s1_record, cand_record, tfidf=tfidf)
                feature_rows.append(feats)
                cand_ids.append(cand_id)

            if not feature_rows:
                output_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
                continue

            feat_df = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
            proba = model.predict_proba(feat_df)[:, 1]
            matched = [cid for cid, p in zip(cand_ids, proba) if p >= threshold]

            output_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matched),
            })

        out_df = pd.DataFrame(output_rows)
        mode = "w" if first_write else "a"
        header = first_write
        out_df.to_csv(out_path, mode=mode, header=header, index=False, sep="\t")
        first_write = False
        print(f"  ... processed {len(seen_s1_ids)} S1 entities so far")

    # Any S1 entities with NO row in candidate_pairs.tsv at all (blocking
    # found zero candidates) still need a row -- fill those in as empty.
    all_s1_ids = set(s1.keys())
    missing = all_s1_ids - seen_s1_ids
    if missing:
        print(f"Filling {len(missing)} S1 entities with no candidates at all "
              f"(empty matches)...")
        missing_df = pd.DataFrame(
            {"source1_entity_id": list(missing), "matched_entity_ids": ""}
        )
        missing_df.to_csv(out_path, mode="a", header=False, index=False, sep="\t")

    print(f"Done -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold-file", required=True)
    parser.add_argument("--out", default="output/matching_results.tsv")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    predict(
        args.source1,
        args.source2,
        args.source3,
        args.candidates,
        args.model,
        args.threshold_file,
        args.out,
        chunksize=args.chunksize,
    )
