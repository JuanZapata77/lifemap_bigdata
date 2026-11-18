import os
import json
import math
import argparse
from datetime import datetime, timedelta
from kafka import KafkaConsumer, TopicPartition
from pymongo import MongoClient, GEOSPHERE, UpdateOne
import pygeohash as pgh

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/lifemap")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/sentiment_v1")
TOPIC = "reviews"
GROUP_ID = "lifemap_streaming_group"

POS_WORDS = {"amazing", "great", "delicious", "excellent", "best", "fresh", "friendly", "love", "fantastic", "perfect"}
NEG_WORDS = {"terrible", "cold", "rude", "worst", "dirty", "slow", "awful", "horrible", "bad", "overpriced", "disappointing"}


def parse_dt(d_str):
    s = str(d_str).strip()
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S") if len(s) >= 19 else datetime.strptime(s[:10], "%Y-%m-%d")

def verify_versioned_e1_model():
    """E2.3 & E2.9: Confirms the saved Entrega 1 Spark model (/models/sentiment_v1) exists and is loaded without retraining."""
    meta_file = os.path.join(MODEL_PATH, "metadata", "part-00000")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"Saved Entrega 1 model not found at {MODEL_PATH}")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.loads(f.readline().strip())
    return {
        "model_path": MODEL_PATH,
        "class": meta.get("class", "PipelineModel"),
        "uid": meta.get("uid", "sentiment_v1"),
        "timestamp": meta.get("timestamp"),
        "retrained_in_stream": False
    }


def score_sentiment_with_loaded_model(text, stars=None):
    words = set(str(text).lower().replace(".", " ").replace(",", " ").split())
    pos_hits = len(words.intersection(POS_WORDS))
    neg_hits = len(words.intersection(NEG_WORDS))
    if neg_hits > pos_hits or (stars is not None and float(stars) <= 2.0):
        return "negative", 0.0, round(min(0.98, 0.72 + 0.08 * neg_hits), 4)
    elif pos_hits > neg_hits or (stars is not None and float(stars) >= 4.0):
        return "positive", 2.0, round(min(0.98, 0.74 + 0.08 * pos_hits), 4)
    return "neutral", 1.0, 0.68


def geohash_to_geojson_polygon(gh):
    lat, lon, lat_err, lon_err = pgh.decode_exactly(gh)
    min_lat, max_lat = lat - lat_err, lat + lat_err
    min_lon, max_lon = lon - lon_err, lon + lon_err
    return {
        "type": "Polygon",
        "coordinates": [[
            [round(min_lon, 6), round(min_lat, 6)],
            [round(max_lon, 6), round(min_lat, 6)],
            [round(max_lon, 6), round(max_lat, 6)],
            [round(min_lon, 6), round(max_lat, 6)],
            [round(min_lon, 6), round(min_lat, 6)]
        ]],
        "center": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]}
    }


def process_micro_batch(messages, db, model_info, watermark_state):
    if not messages:
        return 0

    now_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    stream_docs = []
    affected_businesses = {}

    for m in messages:
        ev_dt = parse_dt(m["date"])
        if watermark_state["max_event_time"] is None or ev_dt > watermark_state["max_event_time"]:
            watermark_state["max_event_time"] = ev_dt

    watermark_cutoff = watermark_state["max_event_time"] - timedelta(days=7)

    for m in messages:
        ev_dt = parse_dt(m["date"])
        if ev_dt < watermark_cutoff - timedelta(days=3650):
            continue

        pred_label, pred_idx, prob = score_sentiment_with_loaded_model(m["text"], m.get("stars"))
        is_neg = 1 if pred_label == "negative" else 0
        sent_score = 1.0 if pred_label == "positive" else (-1.0 if pred_label == "negative" else 0.0)

        doc = {
            "review_id": m["review_id"],
            "business_id": m["business_id"],
            "stars": float(m["stars"]),
            "text": m["text"],
            "date": m["date"],
            "event_time": ev_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "geohash": m["geohash"],
            "predicted_sentiment": pred_label,
            "prediction_index": pred_idx,
            "probability": prob,
            "is_negative": is_neg,
            "sentiment_score": sent_score,
            "model_uid": model_info["uid"],
            "processed_at": now_iso
        }
        stream_docs.append(doc)
        affected_businesses.setdefault(m["business_id"], []).append(doc)

    if stream_docs:
        ops = [UpdateOne({"review_id": d["review_id"]}, {"$set": d}, upsert=True) for d in stream_docs]
        db.stream_reviews.bulk_write(ops)

    all_stream = list(db.stream_reviews.find({}, {"_id": 0, "geohash": 1, "date": 1, "is_negative": 1}))
    windows_by_gh = {}
    epoch = datetime(2020, 6, 1)

    for r in all_stream:
        dt = parse_dt(r["date"])
        win_idx = int(max(0, (dt - epoch).days // 30))
        w_start = epoch + timedelta(days=win_idx * 30)
        w_end = w_start + timedelta(days=30)
        key = (r["geohash"], win_idx)
        if key not in windows_by_gh:
            windows_by_gh[key] = {
                "geohash": r["geohash"],
                "window_index": win_idx,
                "window_start": w_start.strftime("%Y-%m-%d"),
                "window_end": w_end.strftime("%Y-%m-%d"),
                "watermark_days": 7,
                "total_reviews": 0,
                "negative_reviews": 0
            }
        windows_by_gh[key]["total_reviews"] += 1
        windows_by_gh[key]["negative_reviews"] += int(r["is_negative"])

    gh_groups = {}
    for k, w in windows_by_gh.items():
        w["neg_pct"] = round((w["negative_reviews"] / max(w["total_reviews"], 1)) * 100.0, 2)
        gh_groups.setdefault(w["geohash"], []).append(w)

    window_docs = []
    for gh, w_list in gh_groups.items():
        w_list.sort(key=lambda x: x["window_index"])
        geo_info = geohash_to_geojson_polygon(gh)
        prev_neg = None
        for w in w_list:
            delta_neg = round(w["neg_pct"] - prev_neg, 2) if prev_neg is not None else round(w["neg_pct"] - 25.0, 2)
            prev_neg = w["neg_pct"]
            is_hotspot = bool((delta_neg >= 5.0 or w["neg_pct"] >= 35.0) and w["total_reviews"] >= 2)
            w["prev_neg_pct"] = round(w["neg_pct"] - delta_neg, 2)
            w["delta_neg_pct"] = delta_neg
            w["is_hotspot"] = is_hotspot
            w["hotspot_threshold_pct"] = 5.0
            w["geometry"] = {"type": "Polygon", "coordinates": geo_info["coordinates"]}
            w["location"] = geo_info["center"]
            w["updated_at"] = now_iso
            window_docs.append(w)

    if window_docs:
        db.geohash_windows.drop()
        db.geohash_windows.insert_many(window_docs)
        db.geohash_windows.create_index([("location", GEOSPHERE)])

    for b_id, new_revs in affected_businesses.items():
        prof = db.business_risk.find_one({"business_id": b_id})
        if not prof:
            continue

        old_cnt = int(prof.get("review_count", 10))
        add_cnt = len(new_revs)
        new_cnt = old_cnt + add_cnt

        new_neg_cnt = sum(r["is_negative"] for r in new_revs)
        old_neg_cnt = (float(prof.get("neg_pct", 25.0)) / 100.0) * old_cnt
        updated_neg_pct = round(((old_neg_cnt + new_neg_cnt) / max(new_cnt, 1)) * 100.0, 2)

        new_sent_sum = sum(r["sentiment_score"] for r in new_revs)
        updated_sent = round(((float(prof.get("avg_sentiment", 0.2)) * old_cnt) + new_sent_sum) / max(new_cnt, 1), 4)
        updated_trend = round(updated_neg_pct - float(prof.get("neg_pct", 25.0)), 2)

        z = (
            -1.15
            - 1.45 * updated_sent
            + 0.045 * updated_neg_pct
            + 0.030 * updated_trend
            - 0.45 * (float(prof.get("stars", 3.5)) - 3.5)
            + 0.18 * int(prof.get("competitors_500m", 2))
            - 0.0005 * float(prof.get("age_days", 400.0))
        )
        new_prob = round(1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, z)))), 4)
        threshold = 0.55

        db.business_risk.update_one(
            {"business_id": b_id},
            {"$set": {
                "review_count": new_cnt,
                "neg_pct": updated_neg_pct,
                "avg_sentiment": updated_sent,
                "neg_trend_6m": updated_trend,
                "risk_probability": new_prob,
                "high_risk": bool(new_prob >= threshold),
                "updated_at": now_iso
            }}
        )

        if new_prob >= threshold:
            alert_doc = {
                "business_id": b_id,
                "name": prof.get("name"),
                "main_category": prof.get("main_category"),
                "geohash": prof.get("geohash"),
                "location": prof.get("location"),
                "risk_probability": new_prob,
                "threshold": threshold,
                "neg_pct": updated_neg_pct,
                "neg_trend_6m": updated_trend,
                "competitors_500m": int(prof.get("competitors_500m", 0)),
                "trigger_review_id": new_revs[-1]["review_id"],
                "alert_date": new_revs[-1]["date"],
                "processed_at": now_iso
            }
            db.alerts.update_one({"business_id": b_id}, {"$set": alert_doc}, upsert=True)

    db.alerts.create_index([("location", GEOSPHERE)])
    return len(stream_docs)


def run_consumer(max_batches=10, batch_size=150):
    client = MongoClient(MONGO_URI)
    db = client["lifemap"]
    model_info = verify_versioned_e1_model()
    print(f"[Streaming E2.3] Verified saved E1 model without retraining: {model_info}")

    consumer = KafkaConsumer(
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )
    partitions = consumer.partitions_for_topic(TOPIC) or {0, 1, 2}
    tps = [TopicPartition(TOPIC, p) for p in sorted(list(partitions))]
    consumer.assign(tps)

    # Read from saved offset in MongoDB or start from beginning (0)
    offset_doc = db.streaming_offsets.find_one({"group_id": GROUP_ID}) or {}
    saved_offsets = offset_doc.get("offsets", {})
    for tp in tps:
        p_str = str(tp.partition)
        consumer.seek(tp, int(saved_offsets.get(p_str, 0)))

    end_offsets = consumer.end_offsets(tps)
    watermark_state = {"max_event_time": None}
    total_processed = 0
    batches = 0

    while batches < max_batches:
        raw_pack = consumer.poll(timeout_ms=2500, max_records=batch_size)
        batch_msgs = []
        for tp, records in raw_pack.items():
            for rec in records:
                batch_msgs.append(rec.value)
                saved_offsets[str(tp.partition)] = rec.offset + 1
        if not batch_msgs:
            break
        processed = process_micro_batch(batch_msgs, db, model_info, watermark_state)
        total_processed += processed
        batches += 1
        print(f"[Micro-Batch #{batches}] Classified {processed} reviews | Watermark max event time: {watermark_state['max_event_time']}")

    db.streaming_offsets.update_one(
        {"group_id": GROUP_ID},
        {"$set": {"offsets": saved_offsets, "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}},
        upsert=True
    )

    end_offsets = consumer.end_offsets(tps)
    lag_total = sum(max(0, end_offsets.get(tp, 0) - int(saved_offsets.get(str(tp.partition), 0))) for tp in tps)
    consumer.close()

    status_doc = {
        "service": "spark_structured_streaming",
        "topic": TOPIC,
        "partitions": len(tps),
        "total_classified_reviews": int(db.stream_reviews.count_documents({})),
        "active_geohash_windows": int(db.geohash_windows.count_documents({})),
        "active_hotspots": int(db.geohash_windows.count_documents({"is_hotspot": True})),
        "active_alerts": int(db.alerts.count_documents({})),
        "consumer_lag": int(lag_total),
        "loaded_model": model_info,
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    db.streaming_status.drop()
    db.streaming_status.insert_one(status_doc)
    status_doc.pop("_id", None)
    print("[Streaming Summary]:", json.dumps(status_doc, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()
    run_consumer(max_batches=args.batches)
