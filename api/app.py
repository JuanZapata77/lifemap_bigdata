import os
import re
import requests
from flask import Flask, jsonify, request, render_template
from pymongo import MongoClient

app = Flask(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/lifemap")
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client["lifemap"]


def calculate_annualized_closure_rate(closures: int, business_days: float) -> float:
    """E1.7 Formula: Tasa de cierre (%) = cierres / (dias-negocio / 365) * 100"""
    if business_days <= 0:
        return 0.0
    return round((closures / (business_days / 365.0)) * 100.0, 4)



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# 1. GET /health (E1.2)
@app.route("/health", methods=["GET"])
def health():
    mongo_ok = False
    try:
        client.admin.command("ping")
        mongo_ok = True
    except Exception:
        pass

    def check_http(url):
        try:
            return requests.get(url, timeout=2).status_code == 200
        except Exception:
            return False

    dask_ok = check_http(os.getenv("DASK_SCHEDULER_HTTP", "http://dask-scheduler:8787/health"))
    spark_ok = check_http(os.getenv("SPARK_MASTER_HTTP", "http://spark-master:8080"))

    all_up = mongo_ok and dask_ok and spark_ok
    return jsonify({
        "status": "UP" if all_up else "DEGRADED",
        "commit_hash": os.getenv("GIT_COMMIT", "local-dev"),
        "services": {
            "mongodb": "UP" if mongo_ok else "DOWN",
            "dask": "UP" if dask_ok else "DOWN",
            "spark": "UP" if spark_ok else "DOWN",
            "api": "UP"
        }
    }), 200 if all_up else 503


# 2. GET /businesses/near (E1.9 - uses $near)
@app.route("/businesses/near", methods=["GET"])
def businesses_near():
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
        radius = float(request.args.get("radius", 1000))
    except (KeyError, ValueError):
        return jsonify({"error": "Valid lat, lon, and optional radius (meters) required"}), 400

    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and radius > 0):
        return jsonify({"error": "Coordinates or radius out of range"}), 400

    query = {
        "location": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                "$maxDistance": radius
            }
        }
    }
    category = request.args.get("category")
    if category:
        query["main_category"] = category

    results = list(db.businesses.find(query, {"_id": 0}).limit(100))
    return jsonify({"count": len(results), "businesses": results}), 200


# 3. POST /businesses/within (E1.9 - uses $geoWithin)
@app.route("/businesses/within", methods=["POST"])
def businesses_within():
    body = request.get_json(silent=True) or {}
    polygon = body.get("polygon")
    if not polygon or polygon.get("type") != "Polygon" or "coordinates" not in polygon:
        return jsonify({"error": "Body must include a valid GeoJSON 'polygon' object"}), 400

    query = {
        "location": {
            "$geoWithin": {
                "$geometry": polygon
            }
        }
    }
    results = list(db.businesses.find(query, {"_id": 0}).limit(200))
    return jsonify({"count": len(results), "businesses": results}), 200


# 4. GET /zones/summary (E1.9 - uses $geoNear aggregation)
@app.route("/zones/summary", methods=["GET"])
def zones_summary():
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
        radius = float(request.args.get("radius", 2000))
    except (KeyError, ValueError):
        return jsonify({"error": "Valid lat, lon, and optional radius required"}), 400

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
                "total_closures": {"$sum": "$event"},
                "total_business_days": {"$sum": "$business_days"},
                "avg_sentiment_score": {"$avg": "$avg_sentiment"}
            }
        }
    ]
    agg = list(db.businesses.aggregate(pipeline))
    if not agg:
        return jsonify({"total_businesses": 0, "avg_sentiment": None, "annualized_closure_rate_pct": 0.0}), 200

    row = agg[0]
    rate = calculate_annualized_closure_rate(row["total_closures"], row["total_business_days"])
    return jsonify({
        "total_businesses": row["total_businesses"],
        "avg_sentiment": round(row["avg_sentiment_score"] or 0.0, 4),
        "annualized_closure_rate_pct": rate
    }), 200


# 5. GET /stats/closure (E1.9 - closure rates by category or geohash cell)
@app.route("/stats/closure", methods=["GET"])
def stats_closure():
    group_by = request.args.get("by", "category")  # 'category' or 'geohash'
    try:
        min_days = float(request.args.get("min_business_days", 365))
    except ValueError:
        return jsonify({"error": "min_business_days must be numeric"}), 400

    collection = db.closure_by_category if group_by == "category" else db.closure_by_geohash
    cursor = collection.find({"business_days": {"$gte": min_days}}, {"_id": 0}).sort("closure_rate_pct", -1).limit(100)
    return jsonify({"group_by": group_by, "min_business_days": min_days, "results": list(cursor)}), 200


# 6. POST /predict (E1.9 - sentiment classification of new text)
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Non-empty 'text' field is required"}), 400

    # Check if Spark exported vocabulary weights to MongoDB, otherwise fallback lexicon
    tokens = re.findall(r"[a-z]+", text.lower())
    try:
        model_doc = db.model_metadata.find_one({"model_name": "sentiment_v1"})
    except Exception:
        model_doc = None
    if model_doc and "token_weights" in model_doc:
        weights = model_doc["token_weights"]
        score = sum(weights.get(t, 0.0) for t in tokens)
    else:
        pos_words = {"great", "good", "amazing", "excellent", "love", "best", "delicious", "friendly"}
        neg_words = {"bad", "terrible", "worst", "awful", "rude", "slow", "dirty", "cold"}
        score = sum(1.0 for t in tokens if t in pos_words) - sum(1.0 for t in tokens if t in neg_words)

    label = "positive" if score > 0.2 else ("negative" if score < -0.2 else "neutral")
    return jsonify({
        "text": text,
        "predicted_sentiment": label,
        "score": round(score, 4),
        "model_version": model_doc.get("version", "v1") if model_doc else "v1"
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)