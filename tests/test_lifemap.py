from api.app import app, calculate_annualized_closure_rate


def test_annualized_closure_rate_formula():
    # 10 closures in 3650 business-days (10 business-years) = 100.0%
    assert calculate_annualized_closure_rate(10, 3650) == 100.0
    # 5 closures in 36500 business-days (100 business-years) = 5.0%
    assert calculate_annualized_closure_rate(5, 36500) == 5.0
    # Zero or negative days guard
    assert calculate_annualized_closure_rate(5, 0) == 0.0


def test_predict_endpoint_validation():
    client = app.test_client()
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 400

    resp_ok = client.post("/predict", json={"text": "Amazing food and great service!"})
    assert resp_ok.status_code == 200
    assert resp_ok.get_json()["predicted_sentiment"] == "positive"