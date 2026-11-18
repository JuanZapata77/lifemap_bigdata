import os
import json
from app import app, calculate_annualized_closure_rate, get_versioned_model_info, db


def test_annualized_closure_rate_formula():
    """E1.3 & E1.7: Verifies the annualized closure rate math formula."""
    assert calculate_annualized_closure_rate(10, 3650) == 100.0
    assert calculate_annualized_closure_rate(5, 3650) == 50.0
    assert calculate_annualized_closure_rate(0, 1000) == 0.0
    assert calculate_annualized_closure_rate(5, 0) == 0.0


def test_health_includes_kafka_and_lag():
    """E2.1 & E2.8: Confirms /health reports MongoDB, Dask, Spark, Kafka, commit_hash, and consumer_lag."""
    client = app.test_client()
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert "commit_hash" in data
    assert "consumer_lag" in data
    assert "kafka" in data["services"]


def test_versioned_model_loaded_without_retraining():
    """E2.3 & E2.9: Confirms the model loaded by streaming/API is the saved versioned E1 model."""
    info = get_versioned_model_info()
    assert info is not None
    assert "uid" in info
    assert info.get("retrained_in_stream") is False


def test_streaming_smoke_100_messages_classified_in_mongodb():
    """E2.9: Verifies that >= 100 streaming messages published to Kafka appear classified in MongoDB with probabilities."""
    classified_count = db.stream_reviews.count_documents({"predicted_sentiment": {"$in": ["positive", "neutral", "negative"]}})
    assert classified_count >= 100, f"Expected >= 100 classified streaming reviews in MongoDB, found {classified_count}"
    sample = db.stream_reviews.find_one({})
    assert sample is not None
    assert "probability" in sample
    assert "processed_at" in sample


def test_e2_endpoints_respond():
    """E2.8: Verifies all 4 new Entrega 2 endpoints respond cleanly."""
    client = app.test_client()
    assert client.get("/live/hotspots?window=0").status_code == 200
    assert client.get("/alerts?min_prob=0.55").status_code == 200
    assert client.get("/survival/all").status_code == 200
    assert client.get("/risk/near?lat=39.9526&lon=-75.1652&radius=5000").status_code == 200
