"""
Build a labeled training set from candidate_pairs.tsv without loading the
whole (potentially multi-GB) file into memory at once.

Strategy:
  - Read candidate_pairs.tsv in chunks.
  - For each S1 entity, split its candidates into positives (in ground
    truth) and negatives (candidate but not a true match).
  - Keep ALL positives (rare, valuable) and cap negatives per entity
    (default 5) so the training set stays a manageable size instead of
    exploding to hundreds of millions of rows.
  - Compute pairwise features immediately and write out the feature
    table -- we never hold the full candidate set in memory.

Usage:
    python build_training_set.py \
        --source1 dataset/train/train_source1.tsv \
        --source2 dataset/train/train_source2.tsv \
        --source3 dataset/train/train_source3.tsv \
        --candidates output/candidate_pairs.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --out training_features.parquet \
        --neg-per-entity 5
"""

import argparse

import pandas as pd

from featurize import NameTfidf, featurize_pair, normalize_name


def load_records_as_dict(path):
    """entity_id -> dict(business_name, business_address, country, entity_id)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.fillna("")
    return {
        row["entity_id"]: {
            "entity_id": row["entity_id"],
            "business_name": row["business_name"],
            "business_address": row["business_address"],
            "country": row["country"],
        }
        for _, row in df.iterrows()
    }


def load_ground_truth_lookup(path):
    """source1_entity_id -> set(matched_entity_ids)."""
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    lookup = {}
    for _, row in df.iterrows():
        ids = row["matched_entity_ids"]
        lookup[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()
    return lookup


def fit_tfidf_sample(s1_dict, s2_dict, s3_dict, sample_size=200_000):
    """
    Fit the TF-IDF vectorizer on a sample of normalized names across all
    three sources (fitting on all ~5-10M+ names is unnecessary; a large
    random sample gives a representative vocabulary much faster).
    """
    import itertools
    import random

    all_names = list(
        itertools.chain(
            (r["business_name"] for r in s1_dict.values()),
            (r["business_name"] for r in s2_dict.values()),
            (r["business_name"] for r in s3_dict.values()),
        )
    )
    if len(all_names) > sample_size:
        random.seed(42)
        all_names = random.sample(all_names, sample_size)
    normalized = [normalize_name(n) for n in all_names if n]
    tfidf = NameTfidf()
    tfidf.fit(normalized)
    return tfidf


def build(
    source1_path,
    source2_path,
    source3_path,
    candidates_path,
    ground_truth_path,
    out_path,
    neg_per_entity=5,
    chunksize=200_000,
):
    print("Loading source records into memory (id -> record dicts)...")
    s1 = load_records_as_dict(source1_path)
    s2 = load_records_as_dict(source2_path)
    s3 = load_records_as_dict(source3_path)

    def lookup_candidate(cand_id):
        if cand_id.startswith("S2-"):
            return s2.get(cand_id)
        if cand_id.startswith("S3-"):
            return s3.get(cand_id)
        return None

    print("Loading ground truth...")
    gt = load_ground_truth_lookup(ground_truth_path)

    print("Fitting TF-IDF on a name sample...")
    tfidf = fit_tfidf_sample(s1, s2, s3)

    print(f"Streaming {candidates_path} in chunks of {chunksize}...")
    first_write = True
    rows_written = 0

    for chunk in pd.read_csv(
        candidates_path, sep="\t", dtype=str, chunksize=chunksize
    ):
        chunk = chunk.fillna("")
        feature_rows = []

        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            s1_record = s1.get(s1_id)
            if s1_record is None:
                continue

            cand_str = row["candidate_entity_ids"]
            if not cand_str:
                continue
            candidates = cand_str.split(",")

            true_matches = gt.get(s1_id, set())
            positives = [c for c in candidates if c in true_matches]
            negatives = [c for c in candidates if c not in true_matches]

            # keep all positives, cap negatives
            sampled_negatives = negatives[:neg_per_entity]

            for cand_id, label in (
                [(c, 1) for c in positives] + [(c, 0) for c in sampled_negatives]
            ):
                cand_record = lookup_candidate(cand_id)
                if cand_record is None:
                    continue
                feats = featurize_pair(s1_record, cand_record, tfidf=tfidf)
                feats["source1_entity_id"] = s1_id
                feats["candidate_entity_id"] = cand_id
                feats["label"] = label
                feature_rows.append(feats)

        if not feature_rows:
            continue

        feat_df = pd.DataFrame(feature_rows)
        mode = "w" if first_write else "a"
        header = first_write
        feat_df.to_csv(out_path, mode=mode, header=header, index=False)
        first_write = False
        rows_written += len(feat_df)
        print(f"  ... wrote {rows_written} rows so far")

    print(f"Done. Total training rows: {rows_written} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out", default="training_features.csv")
    parser.add_argument("--neg-per-entity", type=int, default=5)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    build(
        args.source1,
        args.source2,
        args.source3,
        args.candidates,
        args.ground_truth,
        args.out,
        neg_per_entity=args.neg_per_entity,
        chunksize=args.chunksize,
    )
