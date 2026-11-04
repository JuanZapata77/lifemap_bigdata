import os
import json
import argparse
from datetime import datetime, timedelta
import random
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from pymongo import MongoClient, ASCENDING, GEOSPHERE
import pygeohash as pgh

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/lifemap")
DASK_SCHEDULER = os.getenv("DASK_SCHEDULER", "tcp://dask-scheduler:8786")
DATA_DIR = os.getenv("DATA_DIR", "/data")

# Default Metro Area & Category Family (configurable via env vars or CLI)
TARGET_CITY = os.getenv("TARGET_CITY", "Philadelphia")
TARGET_CATEGORY = os.getenv("TARGET_CATEGORY", "Restaurants")
MAX_REVIEWS = int(os.getenv("MAX_REVIEWS", "1000000"))


def download_kaggle_dataset():
    """E1.4: Automated download from Kaggle API inside the pipeline."""
    biz_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.json")
    rev_path = os.path.join(DATA_DIR, "yelp_academic_dataset_review.json")
    if os.path.exists(biz_path) and os.path.exists(rev_path):
        print("[Kaggle] Dataset files already present in /data.")
        return biz_path, rev_path

    print("[Kaggle] Downloading yelp-dataset/yelp-dataset via Kaggle API...")
    import kaggle
    pass # kaggle>=1.8 auto-authenticates KAGGLE_API_TOKEN on import
    kaggle.api.dataset_download_files("yelp-dataset/yelp-dataset", path=DATA_DIR, unzip=True)
    return biz_path, rev_path


def generate_demo_dataset():
    """Generates realistic Yelp-structured JSONL files in /data for instant end-to-end verification."""
    os.makedirs(DATA_DIR, exist_ok=True)
    biz_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.json")
    rev_path = os.path.join(DATA_DIR, "yelp_academic_dataset_review.json")

    random.seed(42)
    subcategories = [
        "Restaurants, Italian", "Restaurants, Mexican", "Restaurants, Burgers",
        "Restaurants, Pizza", "Restaurants, Chinese", "Restaurants, Cafes"
    ]
    pos_texts = [
        "Amazing food and great service! Best spot in the city.",
        "Delicious meal, very friendly staff and excellent atmosphere.",
        "Loved the pasta and pizza here, will definitely come back soon!"
    ]
    neg_texts = [
        "Terrible service and cold food. Worst experience ever, very rude waiter.",
        "Dirty tables, slow kitchen, and awful taste. Do not recommend.",
        "Overpriced and bad quality. Waited an hour for a cold burger."
    ]
    neu_texts = [
        "Decent food, average service. Nothing special but okay for a quick bite.",
        "Portions were normal, price is fair, standard neighborhood place."
    ]

    businesses = []
    reviews = []
    base_date = datetime(2018, 1, 1)

    for i in range(250):
        b_id = f"biz_{i:04d}"
        is_open = 0 if random.random() < 0.28 else 1
        lat = 39.9526 + random.uniform(-0.08, 0.08)
        lon = -75.1652 + random.uniform(-0.08, 0.08)
        cat = random.choice(subcategories)
        businesses.append({
            "business_id": b_id,
            "name": f"Philly Bistro {i}",
            "city": TARGET_CITY,
            "state": "PA",
            "latitude": lat,
            "longitude": lon,
            "stars": round(random.uniform(2.0, 5.0), 1),
            "review_count": 20,
            "is_open": is_open,
            "categories": cat
        })

        start_offset = random.randint(0, 800)
        lifespan_days = random.randint(120, 1400)
        for j in range(25):
            r_date = base_date + timedelta(days=start_offset + random.randint(0, lifespan_days))
            if is_open == 0 and random.random() < 0.55:
                stars = random.choice([1, 2])
                text = random.choice(neg_texts)
            else:
                stars = random.choices([1, 2, 3, 4, 5], weights=[10, 10, 15, 30, 35])[0]
                text = random.choice(pos_texts if stars >= 4 else (neg_texts if stars <= 2 else neu_texts))

            reviews.append({
                "review_id": f"rev_{i:04d}_{j:03d}",
                "business_id": b_id,
                "stars": stars,
                "text": f"{text} (ref {j})",
                "date": r_date.strftime("%Y-%m-%d %H:%M:%S")
            })

    # Inject a few invalid rows so cleaning rules demonstrate exact before/after drops
    businesses.append({"business_id": "bad_coords", "name": "Null Cafe", "city": TARGET_CITY, "state": "PA",
                       "latitude": None, "longitude": -75.16, "stars": 3.0, "review_count": 1, "is_open": 1, "categories": "Restaurants"})
    reviews.append({"review_id": "empty_rev", "business_id": "biz_0001", "stars": 4, "text": "", "date": "2021-05-01 12:00:00"})

    with open(biz_path, "w", encoding="utf-8") as f:
        for b in businesses:
            f.write(json.dumps(b) + "\n")
    with open(rev_path, "w", encoding="utf-8") as f:
        for r in reviews:
            f.write(json.dumps(r) + "\n")

    print(f"[Demo] Generated {len(businesses)} businesses and {len(reviews)} reviews in {DATA_DIR}.")
    return biz_path, rev_path


def run_ingestion(use_demo=False):
    print(f"[Dask] Connecting to cluster at {DASK_SCHEDULER}...")
    client = Client(DASK_SCHEDULER)
    print(f"[Dask] Connected: {client}")

    biz_path, rev_path = generate_demo_dataset() if use_demo else download_kaggle_dataset()

    metrics = {}

    # 1. Partitioned read of business.json
    biz_ddf = dd.read_json(biz_path, lines=True, blocksize="64MB")
    metrics["biz_raw_count"] = int(biz_ddf.shape[0].compute())

    # Rule 1: Drop null or out-of-range coordinates
    biz_valid_coords = biz_ddf.dropna(subset=["business_id", "latitude", "longitude", "categories"])
    biz_valid_coords = biz_valid_coords[
        (biz_valid_coords["latitude"] >= -90) & (biz_valid_coords["latitude"] <= 90) &
        (biz_valid_coords["longitude"] >= -180) & (biz_valid_coords["longitude"] <= 180)
    ]
    metrics["biz_after_coords"] = int(biz_valid_coords.shape[0].compute())

    # Rule 2: Filter by target city and category family
    biz_filtered = biz_valid_coords[
        (biz_valid_coords["city"].str.lower() == TARGET_CITY.lower()) &
        (biz_valid_coords["categories"].str.contains(TARGET_CATEGORY, case=False, na=False))
    ]
    biz_pdf = biz_filtered.compute().drop_duplicates(subset=["business_id"])
    metrics["biz_after_city_category_dedup"] = len(biz_pdf)

    valid_biz_ids = set(biz_pdf["business_id"].unique())

    # 2. Partitioned read of review.json
    rev_ddf = dd.read_json(rev_path, lines=True, blocksize="64MB")
    metrics["rev_raw_count"] = int(rev_ddf.shape[0].compute())

    # Rule 3: Filter reviews belonging to target businesses & non-empty text
    rev_filtered = rev_ddf[rev_ddf["business_id"].isin(valid_biz_ids)]
    rev_filtered = rev_filtered.dropna(subset=["review_id", "business_id", "stars", "text", "date"])
    rev_filtered = rev_filtered[rev_filtered["text"].str.strip() != ""]

    rev_pdf = rev_filtered.compute()
    metrics["rev_after_text_and_join"] = len(rev_pdf)

    # Rule 4: Deduplicate by review_id and cap at MAX_REVIEWS if needed
    rev_pdf = rev_pdf.drop_duplicates(subset=["review_id"])
    if len(rev_pdf) > MAX_REVIEWS:
        rev_pdf = rev_pdf.sample(n=MAX_REVIEWS, random_state=42)
    metrics["rev_final_count"] = len(rev_pdf)

    print("[Dask] Cleaning Metrics (Before/After):", json.dumps(metrics, indent=2))
    with open(os.path.join(DATA_DIR, "cleaning_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 3. Derive Sentiment Label (1-2: negative, 3: neutral, 4-5: positive)
    def map_sentiment(stars):
        if stars <= 2:
            return "negative"
        elif stars == 3:
            return "neutral"
        return "positive"

    def map_sentiment_score(stars):
        if stars <= 2:
            return -1.0
        elif stars == 3:
            return 0.0
        return 1.0

    rev_pdf["sentiment"] = rev_pdf["stars"].apply(map_sentiment)
    rev_pdf["sentiment_score"] = rev_pdf["stars"].apply(map_sentiment_score)
    rev_pdf["is_negative"] = (rev_pdf["stars"] <= 2).astype(int)
    rev_pdf["date_dt"] = pd.to_datetime(rev_pdf["date"], errors="coerce")
    rev_pdf = rev_pdf.dropna(subset=["date_dt"])

    # 4. Compute Lifespan per Business (E1.5)
    lifespan_agg = rev_pdf.groupby("business_id").agg(
        first_review=("date_dt", "min"),
        last_review=("date_dt", "max"),
        observed_reviews=("review_id", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        neg_reviews=("is_negative", "sum")
    ).reset_index()

    biz_merged = pd.merge(biz_pdf, lifespan_agg, on="business_id", how="inner")
    biz_merged["business_days"] = (
        (biz_merged["last_review"] - biz_merged["first_review"]).dt.total_seconds() / 86400.0
    ).round(2)
    # Ensure minimum 1.0 day for businesses with reviews on the same day
    biz_merged["business_days"] = biz_merged["business_days"].apply(lambda d: max(float(d), 1.0))

    # Event (is_open == 0 -> closure observed) & Censorship (is_open == 1 -> right-censored)
    biz_merged["event"] = (biz_merged["is_open"] == 0).astype(int)
    biz_merged["censored"] = (biz_merged["is_open"] == 1).astype(int)

    # Extract subcategory as main_category for meaningful grouping
    def pick_main_category(cats):
        parts = [c.strip() for c in str(cats).split(",") if c.strip()]
        for p in parts:
            if p.lower() != TARGET_CATEGORY.lower():
                return p
        return parts[0] if parts else TARGET_CATEGORY

    biz_merged["main_category"] = biz_merged["categories"].apply(pick_main_category)
    biz_merged["geohash"] = biz_merged.apply(
        lambda r: pgh.encode(float(r["latitude"]), float(r["longitude"]), precision=5), axis=1
    )

    # Save cleaned parquet files to /data so Spark jobs read them fast
    biz_Geo_Map = dict(zip(biz_merged["business_id"], biz_merged["geohash"]))
    biz_Cat_Map = dict(zip(biz_merged["business_id"], biz_merged["main_category"]))
    rev_pdf["geohash"] = rev_pdf["business_id"].map(biz_Geo_Map)
    rev_pdf["main_category"] = rev_pdf["business_id"].map(biz_Cat_Map)

    biz_merged.to_parquet(os.path.join(DATA_DIR, "businesses_clean.parquet"), index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    rev_pdf["date"] = rev_pdf["date"].astype(str); rev_pdf.drop(columns=["date_dt"]).to_parquet(os.path.join(DATA_DIR, "reviews_clean.parquet"), index=False, coerce_timestamps="us", allow_truncated_timestamps=True)

    # 5. Batch Load into MongoDB with GeoJSON and 2dsphere index
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["lifemap"]
    db.businesses.drop()
    db.reviews.drop()

    biz_docs = []
    for _, row in biz_merged.iterrows():
        biz_docs.append({
            "business_id": row["business_id"],
            "name": row["name"],
            "city": row["city"],
            "state": row["state"],
            "stars": float(row["stars"]),
            "review_count": int(row["observed_reviews"]),
            "is_open": int(row["is_open"]),
            "main_category": row["main_category"],
            "categories": row["categories"],
            "geohash": row["geohash"],
            "location": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])]
            },
            "first_review": row["first_review"].strftime("%Y-%m-%d %H:%M:%S"),
            "last_review": row["last_review"].strftime("%Y-%m-%d %H:%M:%S"),
            "business_days": float(row["business_days"]),
            "event": int(row["event"]),
            "censored": int(row["censored"]),
            "avg_sentiment": round(float(row["avg_sentiment"]), 4),
            "neg_reviews": int(row["neg_reviews"])
        })

    BATCH_SIZE = 1000
    for i in range(0, len(biz_docs), BATCH_SIZE):
        db.businesses.insert_many(biz_docs[i:i + BATCH_SIZE])

    db.businesses.create_index([("location", GEOSPHERE)])
    db.businesses.create_index([("business_id", ASCENDING)], unique=True)
    db.businesses.create_index([("main_category", ASCENDING)])

    rev_docs = rev_pdf[["review_id", "business_id", "stars", "text", "date", "sentiment", "is_negative", "geohash"]].to_dict("records")
    for i in range(0, len(rev_docs), BATCH_SIZE):
        db.reviews.insert_many(rev_docs[i:i + BATCH_SIZE])
    db.reviews.create_index([("business_id", ASCENDING)])

    print(f"[MongoDB] Loaded {len(biz_docs)} GeoJSON businesses (with 2dsphere index) and {len(rev_docs)} reviews.")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run with synthetic Yelp demo data for fast testing")
    args = parser.parse_args()
    run_ingestion(use_demo=args.demo)