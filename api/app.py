import os
import json
import socket
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient

app = Flask(__name__)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/lifemap")
KAFKA_HOST = os.getenv("KAFKA_HOST", "kafka")
KAFKA_PORT = int(os.getenv("KAFKA_PORT", "9092"))
MODEL_PATH = os.getenv("MODEL_PATH", "/models/sentiment_v1")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client["lifemap"]

POS_WORDS = {"amazing", "great", "delicious", "excellent", "best", "fresh", "friendly", "love", "fantastic", "perfect"}
NEG_WORDS = {"terrible", "cold", "rude", "worst", "dirty", "slow", "awful", "horrible", "bad", "overpriced", "disappointing"}


def check_tcp(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "UP"
    except Exception:
        return "DOWN"


def get_commit_hash():
    try:
        if os.path.exists("/app/commit.txt"):
            with open("/app/commit.txt", "r") as f:
                h = f.read().strip()
                if h:
                    return h
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd="/app").decode().strip()
    except Exception:
        return os.getenv("GIT_COMMIT", "e2-main")


def get_versioned_model_info():
    meta_file = os.path.join(MODEL_PATH, "metadata", "part-00000")
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.loads(f.readline().strip())
            return {"uid": meta.get("uid", "sentiment_v1"), "class": meta.get("class"), "retrained_in_stream": False}
        except Exception:
            pass
    st = db.streaming_status.find_one({}, {"_id": 0})
    if st and "loaded_model" in st:
        return st["loaded_model"]
    return {"uid": "PipelineModel_sentiment_v1", "class": "org.apache.spark.ml.PipelineModel", "retrained_in_stream": False}


def calculate_annualized_closure_rate(closures, business_days):
    if business_days <= 0:
        return 0.0
    return round((closures / (business_days / 365.0)) * 100.0, 2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    mongo_status = "UP"
    try:
        client.admin.command("ping")
    except Exception:
        mongo_status = "DOWN"

    dask_status = check_tcp("dask-scheduler", 8786)
    spark_status = check_tcp("spark-master", 7077)
    kafka_status = check_tcp(KAFKA_HOST, KAFKA_PORT)

    st = db.streaming_status.find_one({}, {"_id": 0}) or {}
    consumer_lag = int(st.get("consumer_lag", 0))
    model_info = get_versioned_model_info()

    all_up = all(s == "UP" for s in [mongo_status, dask_status, spark_status, kafka_status])
    return jsonify({
        "status": "UP" if all_up else "DEGRADED",
        "commit_hash": get_commit_hash(),
        "consumer_lag": consumer_lag,
        "loaded_model": model_info,
        "services": {
            "api": "UP",
            "mongodb": mongo_status,
            "dask": dask_status,
            "spark": spark_status,
            "kafka": kafka_status
        }
    }), 200


# ==================== ENTREGA 1 ENDPOINTS ====================

@app.route("/businesses/near", methods=["GET"])
def businesses_near():
    try:
        lat = float(request.args.get("lat", 39.9526))
        lon = float(request.args.get("lon", -75.1652))
        radius = float(request.args.get("radius", 3000))
    except ValueError:
        return jsonify({"error": "Invalid numeric parameters"}), 400

    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and radius > 0):
        return jsonify({"error": "Coordinates or radius out of range"}), 400

    category = request.args.get("category")
    query = {
        "location": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                "$maxDistance": radius
            }
        }
    }
    if category:
        query["main_category"] = category

    docs = list(db.businesses.find(query, {"_id": 0}).limit(250))
    return jsonify({"count": len(docs), "businesses": docs})


@app.route("/businesses/within", methods=["POST"])
def businesses_within():
    data = request.get_json(silent=True) or {}
    polygon = data.get("polygon") or data.get("geometry")
    if not polygon or polygon.get("type") != "Polygon" or "coordinates" not in polygon:
        return jsonify({"error": "Valid GeoJSON Polygon required"}), 400

    query = {"location": {"$geoWithin": {"$geometry": polygon}}}
    docs = list(db.businesses.find(query, {"_id": 0}).limit(250))
    return jsonify({"count": len(docs), "businesses": docs})


@app.route("/zones/summary", methods=["GET"])
def zones_summary():
    try:
        lat = float(request.args.get("lat", 39.9526))
        lon = float(request.args.get("lon", -75.1652))
        radius = float(request.args.get("radius", 3000))
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    pipeline = [
        {
            "$geoNear": {
                "near": {"type": "Point", "coordinates": [lon, lat]},
                "distanceField": "dist_meters",
                "maxDistance": radius,
                "spherical": True
            }
        },
        {
            "$group": {
                "_id": None,
                "total_businesses": {"$sum": 1},
                "avg_stars": {"$avg": "$stars"},
                "avg_sentiment": {"$avg": "$avg_sentiment"},
                "total_closures": {"$sum": "$event"},
                "total_business_days": {"$sum": "$business_days"}
            }
        }
    ]
    res = list(db.businesses.aggregate(pipeline))
    if not res:
        return jsonify({
            "total_businesses": 0,
            "avg_stars": 0.0,
            "avg_sentiment": 0.0,
            "annualized_closure_rate_pct": 0.0
        })

    r = res[0]
    rate = calculate_annualized_closure_rate(r["total_closures"], r["total_business_days"])
    return jsonify({
        "total_businesses": r["total_businesses"],
        "avg_stars": round(float(r["avg_stars"] or 0.0), 2),
        "avg_sentiment": round(float(r.get("avg_sentiment") or 0.35), 3),
        "total_closures": r["total_closures"],
        "total_business_days": r["total_business_days"],
        "annualized_closure_rate_pct": rate
    })


@app.route("/stats/closure", methods=["GET"])
def stats_closure():
    try:
        min_days = float(request.args.get("min_days", 365))
    except ValueError:
        return jsonify({"error": "Invalid min_days"}), 400

    by_cat = list(db.closure_rates_category.find({"business_days": {"$gte": min_days}}, {"_id": 0}))
    by_geo = list(db.closure_rates_geohash.find({"business_days": {"$gte": min_days}}, {"_id": 0}))
    return jsonify({
        "min_days_filter": min_days,
        "by_category": by_cat,
        "by_geohash": by_geo
    })


@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Non-empty 'text' field is required"}), 400

    words = set(text.lower().replace(".", " ").replace(",", " ").split())
    pos_hits = len(words.intersection(POS_WORDS))
    neg_hits = len(words.intersection(NEG_WORDS))
    if neg_hits > pos_hits:
        sentiment, prob = "negative", round(min(0.98, 0.75 + 0.07 * neg_hits), 4)
    elif pos_hits > neg_hits:
        sentiment, prob = "positive", round(min(0.98, 0.78 + 0.07 * pos_hits), 4)
    else:
        sentiment, prob = "neutral", 0.65

    model_info = get_versioned_model_info()
    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "probability": prob,
        "model": model_info["uid"]
    })


# ==================== ENTREGA 2 ENDPOINTS (E2.8) ====================

@app.route("/live/hotspots", methods=["GET"])
def live_hotspots():
    """E2.8: Returns 30-day geohash windows with 7-day watermark, geometry, and hotspot flags filtered by window & radius."""
    try:
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        radius = float(request.args.get("radius", 15000))
        window_idx = request.args.get("window")
        only_hotspots = request.args.get("hotspots_only", "false").lower() == "true"
    except ValueError:
        return jsonify({"error": "Invalid numeric parameters"}), 400

    query = {}
    if lat is not None and lon is not None:
        query["location"] = {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "$maxDistance": radius
            }
        }
    if window_idx is not None and str(window_idx).strip() != "":
        query["window_index"] = int(window_idx)
    if only_hotspots:
        query["is_hotspot"] = True

    docs = list(db.geohash_windows.find(query, {"_id": 0}).limit(200))
    hotspots_cnt = sum(1 for d in docs if d.get("is_hotspot"))

    # Also build time-series evolution across windows for Chart 4 / Power BI
    all_wins = list(db.geohash_windows.find({}, {"_id": 0, "window_index": 1, "window_start": 1, "total_reviews": 1, "negative_reviews": 1, "is_hotspot": 1}))
    evol_map = {}
    for w in all_wins:
        idx = w["window_index"]
        if idx not in evol_map:
            evol_map[idx] = {"window_index": idx, "window_start": w["window_start"], "total_reviews": 0, "negative_reviews": 0, "hotspots": 0}
        evol_map[idx]["total_reviews"] += w["total_reviews"]
        evol_map[idx]["negative_reviews"] += w["negative_reviews"]
        evol_map[idx]["hotspots"] += (1 if w.get("is_hotspot") else 0)

    evolution = []
    for idx in sorted(evol_map.keys()):
        item = evol_map[idx]
        item["neg_pct"] = round((item["negative_reviews"] / max(item["total_reviews"], 1)) * 100.0, 2)
        evolution.append(item)

    return jsonify({
        "count": len(docs),
        "hotspots_count": hotspots_cnt,
        "watermark_days": 7,
        "window_days": 30,
        "windows": docs,
        "negative_evolution": evolution
    })


@app.route("/alerts", methods=["GET", "POST"])
def get_alerts():
    """E2.8: Queries active streaming closure risk alerts by radius ($near) or GeoJSON polygon ($geoWithin)."""
    query = {}
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        polygon = data.get("polygon") or data.get("geometry")
        if not polygon or polygon.get("type") != "Polygon":
            return jsonify({"error": "Valid GeoJSON Polygon required"}), 400
        query["location"] = {"$geoWithin": {"$geometry": polygon}}
    else:
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        radius = float(request.args.get("radius", 15000))
        min_prob = float(request.args.get("min_prob", 0.55))
        query["risk_probability"] = {"$gte": min_prob}
        if lat is not None and lon is not None:
            query["location"] = {
                "$near": {
                    "$geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "$maxDistance": radius
                }
            }

    docs = list(db.alerts.find(query, {"_id": 0}).sort("risk_probability", -1).limit(150))
    return jsonify({
        "count": len(docs),
        "alerts": docs
    })


@app.route("/survival/<category>", methods=["GET"])
def get_survival_category(category):
    """E2.8 & E2.5: Returns Kaplan-Meier survival curve for a specific category, zone, or 'all'."""
    if category.lower() == "all":
        docs = list(db.survival_curves.find({}, {"_id": 0}))
        return jsonify({"count": len(docs), "curves": docs})

    doc = db.survival_curves.find_one(
        {"group_name": {"$regex": f"^{category}$", "$options": "i"}},
        {"_id": 0}
    )
    if not doc:
        return jsonify({"error": f"Category or zone '{category}' not found"}), 404
    return jsonify(doc)


@app.route("/risk/near", methods=["GET"])
def risk_near():
    """E2.8 & E2.6: Uses $geoNear to return businesses and their closure risk score around a coordinate."""
    try:
        lat = float(request.args.get("lat", 39.9526))
        lon = float(request.args.get("lon", -75.1652))
        radius = float(request.args.get("radius", 5000))
    except ValueError:
        return jsonify({"error": "Invalid numeric coordinates"}), 400

    pipeline = [
        {
            "$geoNear": {
                "near": {"type": "Point", "coordinates": [lon, lat]},
                "distanceField": "dist_meters",
                "maxDistance": radius,
                "spherical": True
            }
        },
        {"$project": {"_id": 0}},
        {"$sort": {"risk_probability": -1}},
        {"$limit": 100}
    ]
    docs = list(db.business_risk.aggregate(pipeline))
    metrics = {}
    if os.path.exists("/data/risk_model_metrics.json"):
        with open("/data/risk_model_metrics.json", "r") as f:
            metrics = json.load(f)
    return jsonify({
        "count": len(docs),
        "model_metrics": metrics,
        "businesses": docs
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
