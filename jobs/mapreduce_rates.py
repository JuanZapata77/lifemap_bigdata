import os
import json
import time
import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/lifemap")
DATA_DIR = os.getenv("DATA_DIR", "/data")
MIN_BUSINESS_DAYS_DEFAULT = 365.0  # Justified minimum: at least 1 full business-year of observation


def calc_closure_rate(closures: int, business_days: float) -> float:
    """Annualized closure rate (%) = closures / (business_days / 365) * 100"""
    if business_days <= 0:
        return 0.0
    return round((closures / (business_days / 365.0)) * 100.0, 4)


def run_mapreduce_and_benchmark():
    spark = SparkSession.builder \
        .appName("LifeMap-MapReduceAndSkew") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    biz_df = spark.read.parquet(os.path.join(DATA_DIR, "businesses_clean.parquet")).cache()
    rev_df = spark.read.parquet(os.path.join(DATA_DIR, "reviews_clean.parquet")).cache()

    # =========================================================================
    # 1. E1.7: MapReduce with RDD reduceByKey by Main Category + DataFrame Check
    # =========================================================================
    print("[MapReduce] Running RDD reduceByKey by main_category...")
    rdd_cat = biz_df.rdd.map(lambda r: (r["main_category"], (int(r["event"]), float(r["business_days"]), 1)))
    reduced_cat = dict(rdd_cat.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])).collect())

    # Validate against DataFrame groupBy
    df_cat_rows = biz_df.groupBy("main_category").agg(
        F.sum("event").alias("closures"),
        F.sum("business_days").alias("business_days"),
        F.count("*").alias("n_businesses")
    ).collect()

    cat_docs = []
    for row in df_cat_rows:
        cat = row["main_category"]
        rdd_closures, rdd_days, rdd_n = reduced_cat[cat]
        # Verify exact match between RDD reduceByKey and DataFrame groupBy
        assert int(row["closures"]) == rdd_closures, f"Mismatch in closures for {cat}"
        assert abs(float(row["business_days"]) - rdd_days) < 1e-2, f"Mismatch in days for {cat}"

        cat_docs.append({
            "category": cat,
            "n_businesses": int(rdd_n),
            "closures": int(rdd_closures),
            "business_days": round(float(rdd_days), 2),
            "closure_rate_pct": calc_closure_rate(rdd_closures, rdd_days),
            "meets_min_days": float(rdd_days) >= MIN_BUSINESS_DAYS_DAYS if "MIN_BUSINESS_DAYS_DAYS" in globals() else float(rdd_days) >= MIN_BUSINESS_DAYS_DEFAULT
        })

    print(f"[Validation] RDD reduceByKey matched DataFrame groupBy across all {len(cat_docs)} categories!")

    # =========================================================================
    # 2. E1.7: MapReduce with RDD reduceByKey by Geohash + % Negative Reviews
    # =========================================================================
    print("[MapReduce] Running RDD reduceByKey by geohash cell...")
    rdd_geo = biz_df.rdd.map(lambda r: (r["geohash"], (int(r["event"]), float(r["business_days"]), 1)))
    reduced_geo = dict(rdd_geo.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])).collect())

    # Compute % negative reviews per geohash cell via RDD reduceByKey
    rdd_rev_geo = rev_df.rdd.map(lambda r: (r["geohash"], (int(r["is_negative"]), 1)))
    reduced_rev_geo = dict(rdd_rev_geo.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).collect())

    geo_docs = []
    for ghash, (closures, b_days, n_biz) in reduced_geo.items():
        neg_cnt, total_rev = reduced_rev_geo.get(ghash, (0, 0))
        pct_neg = round((neg_cnt / total_rev) * 100.0, 2) if total_rev > 0 else 0.0
        geo_docs.append({
            "geohash": ghash,
            "n_businesses": int(n_biz),
            "closures": int(closures),
            "business_days": round(float(b_days), 2),
            "closure_rate_pct": calc_closure_rate(closures, b_days),
            "total_reviews": int(total_rev),
            "negative_reviews": int(neg_cnt),
            "pct_negative_reviews": pct_neg
        })

    # Save both collections to MongoDB
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["lifemap"]
    db.closure_by_category.drop()
    db.closure_by_geohash.drop()
    if cat_docs:
        db.closure_by_category.insert_many(cat_docs)
    if geo_docs:
        db.closure_by_geohash.insert_many(geo_docs)
    mongo_client.close()
    print(f"[MongoDB] Saved {len(cat_docs)} category rates and {len(geo_docs)} geohash rates.")

    # =========================================================================
    # 3. E1.8: Load Balancing & Partition Skew Benchmark (2, 8, 32, 200 parts)
    # =========================================================================
    print("[Benchmark] Measuring 2, 8, 32, and 200 partitions (3 repetitions each, keyed vs unkeyed)...")
    partition_configs = [2, 8, 32, 200]
    benchmark_results = []

    for p in partition_configs:
        for mode in ["keyed_by_category", "unkeyed_round_robin"]:
            times = []
            part_counts = []

            for rep in range(3):
                t0 = time.perf_counter()
                if mode == "keyed_by_category":
                    repart_df = rev_df.repartition(p, "main_category")
                else:
                    repart_df = rev_df.repartition(p)

                # Run aggregation + count rows per partition with spark_partition_id()
                part_rows = (
                    repart_df.withColumn("part_id", F.spark_partition_id())
                    .groupBy("part_id")
                    .count()
                    .collect()
                )
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                if rep == 0:
                    # Fill 0 for empty partitions (e.g. when p=200 and only 6 categories exist)
                    observed = {int(r["part_id"]): int(r["count"]) for r in part_rows}
                    part_counts = [observed.get(i, 0) for i in range(p)]

            mean_t = sum(times) / len(times)
            variance = sum((t - mean_t) ** 2 for t in times) / len(times)
            std_t = math.sqrt(variance)

            max_rows = max(part_counts) if part_counts else 0
            min_rows = min(part_counts) if part_counts else 0
            non_empty_parts = sum(1 for c in part_counts if c > 0)

            benchmark_results.append({
                "partitions": p,
                "mode": mode,
                "runs_sec": [round(t, 4) for t in times],
                "mean_sec": round(mean_t, 4),
                "std_sec": round(std_t, 4),
                "non_empty_partitions": non_empty_parts,
                "max_rows_in_partition": max_rows,
                "min_rows_in_partition": min_rows
            })

    print("[Benchmark Table]:\n", json.dumps(benchmark_results, indent=2))
    with open(os.path.join(DATA_DIR, "partition_benchmark.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)

    spark.stop()
    print("[Spark] Step 8.2 (MapReduce & Partition Skew Benchmark) finished successfully.")


if __name__ == "__main__":
    run_mapreduce_and_benchmark()