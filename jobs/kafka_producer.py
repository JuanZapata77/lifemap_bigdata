import os
import json
import time
import argparse
from datetime import datetime
import pandas as pd
from kafka import KafkaProducer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
DATA_DIR = os.getenv("DATA_DIR", "/data")
TOPIC = "reviews"


def run_producer(limit=300, speed_month_sec=10.0, fast_mode=False):
    post_t_path = os.path.join(DATA_DIR, "reviews_post_T.parquet")
    df = pd.read_parquet(post_t_path)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    # E2.2: Strictly chronological order after cutoff T
    df = df.sort_values("date_dt").reset_index(drop=True)

    if limit and limit > 0:
        df = df.head(limit)

    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    print(f"[Producer E2.2] Publishing {len(df)} post-T reviews in date order to topic '{TOPIC}' (3 partitions, key=business_id)...")
    prev_dt = None
    published = 0

    for _, row in df.iterrows():
        cur_dt = row["date_dt"]
        if not fast_mode and prev_dt is not None:
            delta_days = max(0.0, (cur_dt - prev_dt).total_seconds() / 86400.0)
            # Configurable accelerated clock: 1 month (30 days) = speed_month_sec seconds
            sleep_sec = min(0.15, (delta_days / 30.0) * speed_month_sec)
            if sleep_sec > 0.005:
                time.sleep(sleep_sec)
        prev_dt = cur_dt

        msg = {
            "review_id": str(row["review_id"]),
            "business_id": str(row["business_id"]),
            "stars": float(row["stars"]),
            "text": str(row["text"]),
            "date": str(row["date"]),
            "geohash": str(row.get("geohash", "dr4e3"))
        }
        producer.send(TOPIC, key=msg["business_id"], value=msg)
        published += 1

    producer.flush()
    producer.close()
    print(f"[Producer E2.2] Successfully published {published} chronological post-T messages to Kafka.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=350, help="Max messages to publish")
    parser.add_argument("--month-sec", type=float, default=10.0, help="Seconds per simulated month")
    parser.add_argument("--fast", action="store_true", help="Publish immediately for smoke tests")
    args = parser.parse_args()
    run_producer(limit=args.limit, speed_month_sec=args.month_sec, fast_mode=args.fast)
