import os
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/lifemap")
MODEL_DIR = os.getenv("MODEL_PATH", "/models/sentiment_v1")
DATA_DIR = os.getenv("DATA_DIR", "/data")


def evaluate_classifier(predictions, model_name):
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    acc = eval_acc.evaluate(predictions)
    f1_macro = eval_f1.evaluate(predictions)

    class_names = {0.0: "negative", 1.0: "neutral", 2.0: "positive"}
    per_class = {}

    for label_val, name in class_names.items():
        tp = predictions.filter((F.col("label") == label_val) & (F.col("prediction") == label_val)).count()
        fp = predictions.filter((F.col("label") != label_val) & (F.col("prediction") == label_val)).count()
        fn = predictions.filter((F.col("label") == label_val) & (F.col("prediction") != label_val)).count()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(tp + fn)
        }

    cm_rows = predictions.groupBy("label", "prediction").count().collect()
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for r in cm_rows:
        cm[int(r["label"])][int(r["prediction"])] = int(r["count"])

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_macro, 4),
        "per_class": per_class,
        "confusion_matrix_0neg_1neu_2pos": cm
    }


def train_sentiment_models():
    spark = SparkSession.builder \
        .appName("LifeMap-SentimentTraining") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("[Spark] Reading cleaned reviews parquet...")
    reviews_path = os.path.join(DATA_DIR, "reviews_clean.parquet")
    df = spark.read.parquet(reviews_path)

    df = df.withColumn(
        "label",
        F.when(df["sentiment"] == "negative", 0.0)
         .when(df["sentiment"] == "neutral", 1.0)
         .otherwise(2.0)
    )

    total_reviews = df.count()
    dist_rows = df.groupBy("sentiment").count().collect()
    class_distribution = {
        r["sentiment"]: {
            "count": int(r["count"]),
            "pct": round((r["count"] / total_reviews) * 100.0, 2)
        }
        for r in dist_rows
    }
    print("[Spark] Class Distribution:", json.dumps(class_distribution, indent=2))

    df = df.withColumn("fold", F.abs(F.hash("business_id")) % 10)
    train_df = df.filter("fold < 8").cache()
    test_df = df.filter("fold >= 8").cache()

    print(f"[Spark] Zero-Leakage Split -> Train: {train_df.count()} rows, Test: {test_df.count()} rows")

    tokenizer = RegexTokenizer(inputCol="text", outputCol="raw_tokens", pattern="\\W+")
    remover = StopWordsRemover(
        inputCol="raw_tokens",
        outputCol="tokens",
        stopWords=StopWordsRemover.loadDefaultStopWords("english")
    )
    hashingTF = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")

    print("[Spark] Training Naive Bayes...")
    nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
    pipeline_nb = Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])
    model_nb = pipeline_nb.fit(train_df)
    metrics_nb = evaluate_classifier(model_nb.transform(test_df), "NaiveBayes")

    print("[Spark] Training Logistic Regression...")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=25, regParam=0.01)
    pipeline_lr = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    model_lr = pipeline_lr.fit(train_df)
    metrics_lr = evaluate_classifier(model_lr.transform(test_df), "LogisticRegression")

    if metrics_lr["f1_weighted"] >= metrics_nb["f1_weighted"]:
        winning_model = model_lr
        winner_metrics = metrics_lr
    else:
        winning_model = model_nb
        winner_metrics = metrics_nb

    report_payload = {
        "class_distribution": class_distribution,
        "train_rows": int(train_df.count()),
        "test_rows": int(test_df.count()),
        "naive_bayes": metrics_nb,
        "logistic_regression": metrics_lr,
        "selected_model": winner_metrics["model"]
    }
    print("[Spark] Full Sentiment Evaluation Report:\n", json.dumps(report_payload, indent=2))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "sentiment_metrics.json"), "w") as f:
        json.dump(report_payload, f, indent=2)

    print(f"[Spark] Saving winning model ({winner_metrics['model']}) to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    winning_model.write().overwrite().save(MODEL_DIR)

    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["lifemap"]
    db.model_metadata.update_one(
        {"model_name": "sentiment_v1"},
        {"$set": {
            "model_name": "sentiment_v1",
            "version": "v1",
            "selected_model": winner_metrics["model"],
            "metrics": report_payload
        }},
        upsert=True
    )
    mongo_client.close()
    spark.stop()
    print("[Spark] Step 8.1 (Sentiment ML) finished successfully.")


if __name__ == "__main__":
    train_sentiment_models()
