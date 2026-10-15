#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-aware hyperparameter tuning for Serie A clean-sheet model (PySpark).
- Matches main.py: true SoT_conceded, leak-free rollings, OneHotEncoder
- Split modes:
    * main (default): same seasons as main.py; Val = season before main.py's test seasons,
      Train = all earlier seasons; main.py's test seasons are never used for selection
    * percentile: Train+Val vs Test by date percentile; Train vs Val inside earlier slice
    * season: Test = last season, Val = previous season, Train = all earlier seasons
- Searches compact grids:
    * LR: regParam, elasticNetParam; tunes threshold for F1(positive) on validation
    * RF: numTrees, maxDepth, minInstancesPerNode, featureSubsetStrategy
    * Poisson (goals conceded -> P(clean sheet) = exp(-lambda)): regParam; tunes threshold like LR
- Saves sweep table to CSV and best params (incl. LR/Poisson thresholds) to JSON
- Windows-safe local file writes (no Hadoop native IO required)
"""

import os
import json
import csv
import itertools
import shutil
import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# -----------------------------
# Args
# -----------------------------
# Defaults resolve relative to this file: data at repo root, results next to this script
TUNING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TUNING_DIR)

def parse_args():
    p = argparse.ArgumentParser(description="Time-aware tuning for LR and RF (OHE + season split support)")
    p.add_argument("--csv_path", type=str, default=os.path.join(REPO_DIR, "matches_serie_A.csv"))
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--out_dir", type=str, default=TUNING_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_season_start", type=int, default=2024,
                   help="main mode: first season of main.py's test set (keep in sync with main.py TEST_SEASON_START)")
    p.add_argument("--split_mode", type=str, default="main", choices=["main","percentile","season"],
                   help="percentile: use date percentiles; season: last season=test, prev=train/val split by season")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Boundary between train+val and test for percentile mode")
    p.add_argument("--val_within_train", type=float, default=0.8, help="Boundary within earlier slice (train vs val) for percentile mode")
    return p.parse_args()


# -----------------------------
# Spark
# -----------------------------
def make_spark():
    spark = (
        SparkSession.builder
        .appName("SerieA_Tuning_OHE")
        .config("spark.hadoop.io.native.lib.available", "false")  # avoid Hadoop native on Windows
        .getOrCreate()
    )
    print("Spark version:", spark.version)
    return spark


# -----------------------------
# Utilities
# -----------------------------
def _ensure_writable_file(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except PermissionError:
            tmp = path + ".old"
            try:
                os.replace(path, tmp)
                os.remove(tmp)
            except Exception:
                pass


def write_csv_local(rows, header, out_file):
    _ensure_writable_file(out_file)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# -----------------------------
# Data and features
# -----------------------------
REQUIRED = ["Date", "Team", "Opponent", "Venue", "GF", "GA", "xG", "xGA", "SoT", "Poss"]

def load_data(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    for junk in ["_c0", "Unnamed: 0", "Unnamed:0", "index"]:
        if junk in df.columns:
            df = df.drop(junk)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # unify team names ("Internazionale" in Team vs "Inter" in Opponent breaks the opponent join)
    df = df.replace({"Internazionale": "Inter"}, subset=["Team", "Opponent"])
    df = df.withColumn("Date", F.to_date("Date", "yyyy-MM-dd"))
    for col in ["GF", "GA", "xG", "xGA", "SoT", "Poss"]:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))
    df = df.filter(F.col("Date").isNotNull())
    return df


def add_sot_conceded(df):
    opp = df.select(
        F.col("Date").alias("d"),
        F.col("Team").alias("opp_team"),
        F.col("Opponent").alias("opp_opp"),
        F.col("SoT").alias("Opp_SoT")
    )
    cond = (
        (df["Date"] == opp["d"]) &
        (df["Team"] == opp["opp_opp"]) &
        (df["Opponent"] == opp["opp_team"])
    )
    df = (
        df.join(opp, cond, "left")
          .drop("d", "opp_team", "opp_opp")
          .withColumnRenamed("Opp_SoT", "SoT_conceded")
    )
    return df


def add_rollings(df, N):
    w_prev = Window.partitionBy("Team").orderBy("Date").rowsBetween(-N, -1)
    df = (
        df
        .withColumn("roll_GA", F.avg("GA").over(w_prev))
        .withColumn("roll_xGA", F.avg("xGA").over(w_prev))
        .withColumn("roll_SoT_conceded", F.avg("SoT_conceded").over(w_prev))
        .withColumn("roll_poss", F.avg("Poss").over(w_prev))
        .withColumn("team_roll_GF", F.avg("GF").over(w_prev))
        .withColumn("team_roll_xG", F.avg("xG").over(w_prev))
        .withColumn("team_roll_SoT", F.avg("SoT").over(w_prev))
    )
    opp_rolls = df.select(
        F.col("Date").alias("d"),
        F.col("Team").alias("opp_team"),
        F.col("Opponent").alias("opp_opp"),
        F.col("team_roll_GF").alias("opp_roll_GF"),
        F.col("team_roll_xG").alias("opp_roll_xG"),
        F.col("team_roll_SoT").alias("opp_roll_SoT"),
    )
    cond = (
        (df["Date"] == opp_rolls["d"]) &
        (df["Team"] == opp_rolls["opp_opp"]) &
        (df["Opponent"] == opp_rolls["opp_team"])
    )
    df = df.join(opp_rolls, cond, "left").drop("d", "opp_team", "opp_opp")
    df = df.withColumn("label", (F.col("GA") == F.lit(0)).cast(IntegerType()))
    needed = ["roll_GA","roll_xGA","roll_SoT_conceded","roll_poss","opp_roll_GF","opp_roll_xG","opp_roll_SoT"]
    df = df.dropna(subset=needed)
    return df


# -----------------------------
# Splits
# -----------------------------
def split_percentile(df, train_ratio=0.8, val_within_train=0.8):
    ts_col = F.col("Date").cast("timestamp").cast("long")
    b1 = df.select(F.percentile_approx(ts_col, train_ratio).alias("b")).collect()[0]["b"]
    b1_date = F.to_date(F.from_unixtime(F.lit(b1)))
    df_tv = df.filter(F.col("Date") <= b1_date)
    df_test = df.filter(F.col("Date") > b1_date)

    b2 = df_tv.select(F.percentile_approx(ts_col, val_within_train).alias("b")).collect()[0]["b"]
    b2_date = F.to_date(F.from_unixtime(F.lit(b2)))
    df_train = df_tv.filter(F.col("Date") <= b2_date)
    df_val = df_tv.filter(F.col("Date") > b2_date)

    return df_train, df_val, df_test, ("percentile", b1, b2)


def split_season(df):
    # Add season label: season_end_year (e.g., 2023 for 2022/23)
    df = df.withColumn(
        "season_start",
        F.year("Date") + F.when(F.month("Date") >= 7, F.lit(1)).otherwise(F.lit(0))
    )
    last_season = df.agg(F.max("season_start").alias("s")).first()["s"]
    prev_season = last_season - 1

    df_test = df.filter(F.col("season_start") == last_season)
    df_val  = df.filter(F.col("season_start") == prev_season)
    df_train = df.filter(F.col("season_start") < prev_season)

    # Fallback: if train is empty, fold val into train and split by percentile
    if df_train.count() == 0:
        earlier = df.filter(F.col("season_start") < last_season)
        t, v, _, (_, _, b2) = split_percentile(earlier, train_ratio=0.8, val_within_train=0.8)
        return t, v, df_test, ("season-fallback", last_season, b2)

    return df_train, df_val, df_test, ("season", last_season, prev_season)


def split_main(df, test_season_start):
    # Same boundary as main.py: seasons >= test_season_start are main.py's test set and are never
    # used here; the season just before it is validation, everything earlier is training.
    # (Season = end year, e.g. 2023 for 2022/23)
    val_season = test_season_start - 1
    df_train = df.filter(F.col("Season") < val_season)
    df_val = df.filter(F.col("Season") == val_season)
    df_test = df.filter(F.col("Season") >= test_season_start)
    return df_train, df_val, df_test, ("main", val_season, test_season_start)


# -----------------------------
# Weights
# -----------------------------
def add_class_weights(train):
    counts = train.groupBy("label").count().cache()
    total = counts.agg(F.sum("count")).first()[0]
    n0 = counts.filter(F.col("label") == 0).first()["count"]
    n1 = counts.filter(F.col("label") == 1).first()["count"]
    w0 = float(total) / (2.0 * n0) if n0 > 0 else 1.0
    w1 = float(total) / (2.0 * n1) if n1 > 0 else 1.0
    print(f"Class counts -> 0: {n0}, 1: {n1} | weights -> w0: {w0:.3f}, w1: {w1:.3f}")
    train_w = train.withColumn("weightCol", F.when(F.col("label") == 1, F.lit(w1)).otherwise(F.lit(w0)))
    return train_w, w0, w1


# -----------------------------
# Tuning helpers
# -----------------------------
def threshold_sweep(df_pred, prob_col, thresholds):
    best = (-1.0, 0.5)  # (F1, threshold)
    best_triplet = (0.0, 0.0, 0.0)  # precision, recall, f1
    for t in thresholds:
        tmp = df_pred.withColumn("yhat_t", (F.col(prob_col) >= F.lit(t)).cast("int"))
        agg = tmp.select(
            F.sum(F.when((F.col("label")==1) & (F.col("yhat_t")==1), 1).otherwise(0)).alias("TP"),
            F.sum(F.when((F.col("label")==0) & (F.col("yhat_t")==1), 1).otherwise(0)).alias("FP"),
            F.sum(F.when((F.col("label")==1) & (F.col("yhat_t")==0), 1).otherwise(0)).alias("FN"),
            F.sum(F.when((F.col("label")==0) & (F.col("yhat_t")==0), 1).otherwise(0)).alias("TN"),
        ).collect()[0]
        TP, FP, FN, TN = agg["TP"], agg["FP"], agg["FN"], agg["TN"]
        prec = float(TP)/(TP+FP) if TP+FP>0 else 0.0
        rec  = float(TP)/(TP+FN) if TP+FN>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if prec+rec>0 else 0.0
        if f1 > best[0]:
            best = (f1, t)
            best_triplet = (prec, rec, f1)
    return best[1], best_triplet


def evaluate_rf(val_pred):
    val_eval = val_pred.withColumn("yhat_best", (F.col("p1") >= F.lit(0.5)).cast("double"))
    agg = val_eval.select(
        F.sum(F.when((F.col("label")==1) & (F.col("yhat_best")==1), 1).otherwise(0)).alias("TP"),
        F.sum(F.when((F.col("label")==0) & (F.col("yhat_best")==1), 1).otherwise(0)).alias("FP"),
        F.sum(F.when((F.col("label")==1) & (F.col("yhat_best")==0), 1).otherwise(0)).alias("FN"),
        F.sum(F.when((F.col("label")==0) & (F.col("yhat_best")==0), 1).otherwise(0)).alias("TN"),
    ).collect()[0]
    TP, FP, FN, TN = agg["TP"], agg["FP"], agg["FN"], agg["TN"]
    prec = float(TP)/(TP+FP) if TP+FP>0 else 0.0
    rec  = float(TP)/(TP+FN) if TP+FN>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if prec+rec>0 else 0.0

    auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="raw_rf", metricName="areaUnderROC")
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="yhat_best", metricName="accuracy")
    auc = auc_eval.evaluate(val_pred)
    acc = acc_eval.evaluate(val_eval)
    return prec, rec, f1, auc, acc


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    spark = make_spark()

    # Load + features
    df = load_data(spark, args.csv_path)
    df = add_sot_conceded(df)
    df = add_rollings(df, args.window)

    # Splits
    if args.split_mode == "main":
        train, val, test, meta = split_main(df, args.test_season_start)
    elif args.split_mode == "season":
        train, val, test, meta = split_season(df)
    else:
        train, val, test, meta = split_percentile(df, args.train_ratio, args.val_within_train)
    print("Split meta:", meta)
    print("Train:", train.count(), "Val:", val.count(), "Test:", test.count())

    # Weights (train only)
    train_w, w0, w1 = add_class_weights(train)

    # Pipeline (OHE + assembler)
    CAT_COLS = ["Team", "Opponent", "Venue"]
    IDX_COLS = [f"{c}_idx" for c in CAT_COLS]
    OHE_COLS = [f"{c}_ohe" for c in CAT_COLS]
    NUM_COLS = ["roll_GA", "roll_xGA", "roll_SoT_conceded", "roll_poss", "opp_roll_GF", "opp_roll_xG", "opp_roll_SoT"]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CAT_COLS]
    enc = OneHotEncoder(inputCols=IDX_COLS, outputCols=OHE_COLS, dropLast=True)

    assembler = VectorAssembler(inputCols=OHE_COLS + NUM_COLS, outputCol="features", handleInvalid="keep")
    pipe = Pipeline(stages=indexers + [enc, assembler])
    pipe_model = pipe.fit(train_w)
    fe_train = pipe_model.transform(train_w)
    fe_val   = pipe_model.transform(val)

    # Grids
    lr_grid = list(itertools.product(
        [0.0, 1e-3, 1e-2, 1e-1],   # regParam
        [0.0, 0.5, 1.0],           # elasticNetParam
        [100]                      # maxIter
    ))
    rf_grid = list(itertools.product(
        [200, 400, 800],           # numTrees
        [6, 8, 10, 12],            # maxDepth
        [1, 3, 5],                 # minInstancesPerNode
        ["auto", "sqrt", "log2"]   # featureSubsetStrategy
    ))
    pois_grid = [0.0, 1e-3, 1e-2, 1e-1, 1.0]  # regParam (L2)

    rows = []
    best_lr = {"f1": -1.0}
    best_rf = {"f1": -1.0}
    best_pois = {"f1": -1.0}

    # Tune LR
    for reg, l1ratio, iters in lr_grid:
        lr = LogisticRegression(
            featuresCol="features", labelCol="label", weightCol="weightCol",
            predictionCol="prediction_lr", probabilityCol="probability_lr",
            rawPredictionCol="raw_lr", maxIter=int(iters), regParam=float(reg), elasticNetParam=float(l1ratio)
        )
        lr_model = lr.fit(fe_train)
        val_pred = lr_model.transform(fe_val).withColumn("p1", vector_to_array("probability_lr")[1])

        # threshold sweep for LR
        thr, (prec, rec, f1) = threshold_sweep(val_pred, "p1", thresholds=[round(x,2) for x in [0.3,0.35,0.4,0.45,0.5,0.55,0.6]])
        val_eval = val_pred.withColumn("yhat_best", (F.col("p1") >= F.lit(thr)).cast("double"))

        auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="raw_lr", metricName="areaUnderROC")
        acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="yhat_best", metricName="accuracy")
        auc = auc_eval.evaluate(val_pred)
        acc = acc_eval.evaluate(val_eval)

        params = {"regParam": reg, "elasticNetParam": l1ratio, "maxIter": iters, "threshold": thr}
        rows.append(["LR", json.dumps(params), prec, rec, f1, auc, acc])

        if f1 > best_lr["f1"]:
            best_lr = {"params": params, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "acc": acc}

    # Tune Poisson: models goals conceded, P(clean sheet) = exp(-expected goals conceded).
    # Unweighted (it predicts counts, not the imbalanced label); features scaled so the L2 penalty is even.
    scaler = StandardScaler(inputCol="features", outputCol="features_std", withMean=False).fit(fe_train)
    fe_train_std, fe_val_std = scaler.transform(fe_train), scaler.transform(fe_val)
    for reg in pois_grid:
        pois = GeneralizedLinearRegression(
            featuresCol="features_std", labelCol="GA", family="poisson", link="log",
            predictionCol="exp_ga", maxIter=100, regParam=float(reg)
        )
        val_pred = pois.fit(fe_train_std).transform(fe_val_std).withColumn("p1", F.exp(-F.col("exp_ga")))

        thr, (prec, rec, f1) = threshold_sweep(val_pred, "p1", thresholds=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        val_eval = val_pred.withColumn("yhat_best", (F.col("p1") >= F.lit(thr)).cast("double"))
        auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="p1", metricName="areaUnderROC").evaluate(val_pred)
        acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="yhat_best", metricName="accuracy").evaluate(val_eval)

        params = {"regParam": reg, "threshold": thr}
        rows.append(["POISSON", json.dumps(params), prec, rec, f1, auc, acc])

        if f1 > best_pois["f1"]:
            best_pois = {"params": params, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "acc": acc}

    # Tune RF
    for trees, depth, minInst, fss in rf_grid:
        rf = RandomForestClassifier(
            featuresCol="features", labelCol="label", weightCol="weightCol",
            predictionCol="prediction_rf", probabilityCol="probability_rf", rawPredictionCol="raw_rf",
            numTrees=int(trees), maxDepth=int(depth), minInstancesPerNode=int(minInst), featureSubsetStrategy=str(fss), seed=args.seed
        )
        rf_model = rf.fit(fe_train)
        val_pred = rf_model.transform(fe_val).withColumn("p1", vector_to_array("probability_rf")[1])

        prec, rec, f1, auc, acc = evaluate_rf(val_pred)

        params = {"numTrees": trees, "maxDepth": depth, "minInstancesPerNode": minInst, "featureSubsetStrategy": fss}
        rows.append(["RF", json.dumps(params), prec, rec, f1, auc, acc])

        if f1 > best_rf["f1"]:
            best_rf = {"params": params, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "acc": acc}

    # Save sweep
    out_dir = args.out_dir
    write_csv_local(rows, ["model","params_json","precision","recall","f1","auc","acc"], os.path.join(out_dir, "tuning_results.csv"))

    # Save best params
    best = {
        "lr": best_lr["params"],
        "rf": best_rf["params"],
        "poisson": best_pois["params"]
    }
    _ensure_writable_file(os.path.join(out_dir, "best_params.json"))
    with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("Best LR:", best_lr)
    print("Best RF:", best_rf)
    print("Best Poisson:", best_pois)
    print(f"Saved grid results and best params under: {out_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
