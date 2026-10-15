import os
import json
import argparse
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegression as LRCal
from pyspark.ml.feature import VectorAssembler as VAsm

# configuration (paths default to locations relative to this file, so it runs from any cwd)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Serie A clean-sheet forecasting")
parser.add_argument("--csv_path", type=str, default=os.path.join(BASE_DIR, "matches_serie_A.csv"))
parser.add_argument("--out_dir", type=str, default=os.path.join(BASE_DIR, "outputs"))
parser.add_argument("--params", type=str, default=os.path.join(BASE_DIR, "tuning", "best_params.json"),
                    help="model hyperparameters selected by tuning/tune.py")
args = parser.parse_args()

# hyperparameters chosen on validation data by tuning/tune.py
with open(args.params, encoding="utf-8") as f:
    BEST = json.load(f)
LR_PARAMS = BEST["lr"]
RF_PARAMS = dict(
    numTrees=int(BEST["rf"]["numTrees"]),
    maxDepth=int(BEST["rf"]["maxDepth"]),
    minInstancesPerNode=int(BEST["rf"]["minInstancesPerNode"]),
    featureSubsetStrategy=str(BEST["rf"]["featureSubsetStrategy"]),
)
POIS_PARAMS = BEST["poisson"]

CFG = {
    "CSV_PATH": args.csv_path,
    "OUT_DIR": args.out_dir,
    "N_BACK": 5,
    "TEST_SEASON_START": 2024,   # tuning/tune.py validates on the season before this (keep in sync)
    "SEED": 15,
    "WITH_WEIGHTS": True,
    "CALIBRATE_RF": True,
}

# spark session
spark = (
    SparkSession.builder
    .appName("SerieA_CleanSheet_Forecasting")
    .getOrCreate()
)
print("Spark version:", spark.version)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
os.makedirs(CFG["OUT_DIR"], exist_ok=True)

# helpers
def brier_score(df_probs, prob_col: str, label_col: str = "cs") -> float:
    return df_probs.select(F.pow(F.col(prob_col) - F.col(label_col), 2).alias("sqerr")).agg(F.avg("sqerr")).first()[0]

def write_csv_pd(pdf, name: str):
    import pandas as pd
    path = os.path.join(CFG["OUT_DIR"], name)
    pdf.to_csv(path, index=False)



# 1) load and basic clean
df = (spark.read.csv(CFG["CSV_PATH"], header=True, inferSchema=True)
                .withColumn("Date", F.to_date("Date")))
if "_c0" in df.columns:
    df = df.drop("_c0")

# normalise column names if needed
rename_map = {"Opp Formation": "Opp_Formation", "Match Report": "Match_Report"}
for old, new in rename_map.items():
    if old in df.columns:
        df = df.withColumnRenamed(old, new)

# unify team names: the source uses "Internazionale" in Team but "Inter" in Opponent,
# which breaks the mirror join and drops Inter's matches
df = df.replace({"Internazionale": "Inter"}, subset=["Team", "Opponent"])

# labels and simple context
df = df.withColumn("cs", (F.col("GA") == 0).cast("int"))
df = df.withColumn("home", (F.col("Venue").isin(["Home", "H"]).cast("int")))



# 2) mirror join to derive conceded stats
mirror = (
    df.select(
        "Date",
        F.col("Team").alias("Opponent"),
        F.col("Opponent").alias("Team"),
        F.col("SoT").alias("SoT_opp"),
        F.col("Sh").alias("Sh_opp"),
        F.col("xG").alias("xG_opp"),
        F.col("GF").alias("GF_opp"),
        F.col("GA").alias("GA_opp"),
        F.col("Dist").alias("Dist_opp"),
        (F.col("PK").alias("PK_opp") if "PK" in df.columns else F.lit(None).cast("double").alias("PK_opp"))
    )
)
df = df.join(mirror, on=["Date", "Team", "Opponent"], how="left")
df = (df.withColumn("SoT_conceded", F.col("SoT_opp"))
        .withColumn("Sh_conceded", F.col("Sh_opp"))
        .withColumn("Dist_allowed", F.col("Dist_opp"))
        .withColumn("PK_conceded", F.col("PK_opp")))




# 3) rolling windows and engineered features
N = CFG["N_BACK"]
w_team = Window.partitionBy("Team").orderBy("Date").rowsBetween(-N, -1)
w_opp  = Window.partitionBy("Opponent").orderBy("Date").rowsBetween(-N, -1)

# core engineered signals
df = df.withColumn("def_overperf_roll", F.avg(F.col("xGA") - F.col("GA")).over(w_team))
# goals conceded by the opponent beyond the xG it allowed (positive = leakier than expected)
df = df.withColumn("opp_def_underperf_roll", F.avg(F.col("GF") - F.col("xG")).over(w_opp))
df = df.withColumn("CS_lastN", F.avg(F.col("cs")).over(w_team))

# team rollings
team_roll_cols = {
    "roll_GA": F.avg("GA").over(w_team),
    "roll_xGA": F.avg("xGA").over(w_team),
    "roll_SoT_conceded": F.avg("SoT_conceded").over(w_team),
    "roll_poss": F.avg("Poss").over(w_team)
}
for c, expr in team_roll_cols.items():
    df = df.withColumn(c, expr)

# opponent defence: w_opp groups rows by Opponent, so GF/xG/SoT/Sh there are what other
# teams produced against that opponent, i.e. the opponent's goals/xG/shots conceded
opp_roll_cols = {
    "opp_roll_GA": F.avg("GF").over(w_opp),
    "opp_roll_xGA": F.avg("xG").over(w_opp),
    "opp_roll_SoT_conceded": F.avg("SoT").over(w_opp),
    "opp_roll_Sh_conceded": F.avg("Sh").over(w_opp)
}
for c, expr in opp_roll_cols.items():
    df = df.withColumn(c, expr)

# drop early rows without full history
key_rolls = [
    "def_overperf_roll", "opp_def_underperf_roll", "CS_lastN",
    "roll_GA", "roll_xGA", "roll_SoT_conceded", "roll_poss",
    "opp_roll_GA", "opp_roll_xGA", "opp_roll_SoT_conceded", "opp_roll_Sh_conceded"
]
for c in key_rolls:
    df = df.filter(F.col(c).isNotNull())




# 4) season hold-out split
train_df = df.filter(F.col("Season") < F.lit(CFG["TEST_SEASON_START"]))
test_df  = df.filter(F.col("Season") >= F.lit(CFG["TEST_SEASON_START"]))
print("Train rows:", train_df.count(), "> Test rows:", test_df.count())




# 5) feature pipeline
cat_cols = ["Team", "Opponent"]
num_cols = [
    "home",
    "def_overperf_roll", "opp_def_underperf_roll", "CS_lastN",
    "roll_GA", "roll_xGA", "roll_SoT_conceded", "roll_poss",
    "opp_roll_GA", "opp_roll_xGA", "opp_roll_SoT_conceded", "opp_roll_Sh_conceded"
]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                            outputCols=[f"{c}_ohe" for c in cat_cols],
                            dropLast=True)]
assembler = VectorAssembler(inputCols=num_cols + [f"{c}_ohe" for c in cat_cols], outputCol="features")
fe_pipe = Pipeline(stages=indexers + encoders + [assembler])
fe_model = fe_pipe.fit(train_df)
train_fe = fe_model.transform(train_df)
test_fe  = fe_model.transform(test_df)



# 6) class weights
if CFG["WITH_WEIGHTS"]:
    stats = train_fe.groupBy("cs").count().collect()
    cnt = {row["cs"]: row["count"] for row in stats}
    n0, n1 = cnt.get(0, 1), cnt.get(1, 1)
    total = n0 + n1
    w0, w1 = total / (2.0 * n0), total / (2.0 * n1)
    print(f"Class counts -> 0: {n0}, 1: {n1} | weights -> w0: {w0:.3f}, w1: {w1:.3f}")
    train_fe = train_fe.withColumn("class_weight", F.when(F.col("cs") == 1, F.lit(float(w1))).otherwise(F.lit(float(w0))))
    test_fe  = test_fe.withColumn("class_weight",  F.when(F.col("cs") == 1, F.lit(float(w1))).otherwise(F.lit(float(w0))))
else:
    train_fe = train_fe.withColumn("class_weight", F.lit(1.0))
    test_fe  = test_fe.withColumn("class_weight", F.lit(1.0))



# 7) train models 
lr = LogisticRegression(
    featuresCol="features", labelCol="cs", weightCol="class_weight",
    predictionCol="prediction_lr", rawPredictionCol="raw_lr",
    probabilityCol="probability_lr",
    maxIter=int(LR_PARAMS["maxIter"]), regParam=float(LR_PARAMS["regParam"]),
    elasticNetParam=float(LR_PARAMS["elasticNetParam"]),
    threshold=float(LR_PARAMS.get("threshold", 0.5))
)
rf = RandomForestClassifier(
    featuresCol="features", labelCol="cs", weightCol="class_weight",
    predictionCol="prediction_rf", rawPredictionCol="raw_rf",
    probabilityCol="probability_rf",
    seed=CFG["SEED"], **RF_PARAMS
)
print("LR params:", LR_PARAMS, "| RF params:", RF_PARAMS, "| Poisson params:", POIS_PARAMS)
lr_model = lr.fit(train_fe)
rf_model = rf.fit(train_fe)
pred_test = lr_model.transform(test_fe)
pred_test = rf_model.transform(pred_test)

# Poisson model of goals conceded: a clean sheet is GA = 0, so P(clean sheet) = exp(-expected GA).
# Learns from full scorelines and needs no class weights; features scaled so the L2 penalty is even.
scaler = StandardScaler(inputCol="features", outputCol="features_std", withMean=False).fit(train_fe)
pois_model = GeneralizedLinearRegression(
    featuresCol="features_std", labelCol="GA", family="poisson", link="log",
    predictionCol="exp_goals_conceded", maxIter=100, regParam=float(POIS_PARAMS["regParam"])
).fit(scaler.transform(train_fe))
pred_test = pois_model.transform(scaler.transform(pred_test))
pred_test = (pred_test
             .withColumn("p1_pois", F.exp(-F.col("exp_goals_conceded")))
             .withColumn("prediction_pois", (F.col("p1_pois") >= F.lit(float(POIS_PARAMS["threshold"]))).cast("double")))



# 8) calibration for rf (platt scaling) 
if CFG["CALIBRATE_RF"]:
    tr_ts = train_fe.withColumn("ts", F.unix_timestamp("Date"))
    cutoff_tr = tr_ts.approxQuantile("ts", [0.8], 0.01)[0]  # 80/20 inside train
    sub_train = tr_ts.filter(F.col("ts") <= cutoff_tr).drop("ts")
    valid     = tr_ts.filter(F.col("ts") >  cutoff_tr).drop("ts")

    rf_cal = RandomForestClassifier(
        featuresCol="features", labelCol="cs", weightCol="class_weight",
        predictionCol="pred_rf_cal", rawPredictionCol="raw_rf_cal",
        probabilityCol="prob_rf_cal",
        seed=CFG["SEED"], **RF_PARAMS
    ).fit(sub_train)
    valid_scored = rf_cal.transform(valid)

    valid_scored = valid_scored.withColumn("rf_p1", vector_to_array("prob_rf_cal")[1])
    va = VAsm(inputCols=["rf_p1"], outputCol="calib_feat")
    valid_scored = va.transform(valid_scored)

    platt = LRCal(featuresCol="calib_feat", labelCol="cs",
                    predictionCol="calib_pred", probabilityCol="calib_prob",
                    maxIter=50, regParam=0.0, elasticNetParam=0.0)
    platt_model = platt.fit(valid_scored)

    pred_test = pred_test.withColumn("rf_p1", vector_to_array("probability_rf")[1])
    pred_test = va.transform(pred_test)
    pred_test = platt_model.transform(pred_test).withColumnRenamed("calib_prob", "probability_rf_calib")
else:
    pred_test = pred_test.withColumn("probability_rf_calib", F.col("probability_rf"))

# extract p1 columns
pred_test = (
    pred_test
    .withColumn("p1_lr", vector_to_array("probability_lr")[1])
    .withColumn("p1_rf", vector_to_array("probability_rf")[1])
    .withColumn("p1_rf_calib", vector_to_array("probability_rf_calib")[1])
)

# readiness bands from calibrated rf using train cutoffs (40th, 80th pct)
if CFG["CALIBRATE_RF"]:
    train_cal = rf_model.transform(train_fe).withColumn("rf_p1", vector_to_array("probability_rf")[1])
    va = VAsm(inputCols=["rf_p1"], outputCol="calib_feat")
    train_cal = va.transform(train_cal)
    train_cal = platt_model.transform(train_cal).withColumnRenamed("calib_prob", "probability_rf_calib")
    train_cal = train_cal.withColumn("p1_rf_calib", vector_to_array("probability_rf_calib")[1])
    q_lo, q_hi = train_cal.approxQuantile("p1_rf_calib", [0.40, 0.80], 0.001)
    band_source_col = "p1_rf_calib"
else:
    train_raw = rf_model.transform(train_fe).withColumn("p1_rf", vector_to_array("probability_rf")[1])
    q_lo, q_hi = train_raw.approxQuantile("p1_rf", [0.40, 0.80], 0.001)
    band_source_col = "p1_rf"

print(f"[TRAIN cutoffs] Low < {q_lo:.3f} | Medium < {q_hi:.3f} | High >= {q_hi:.3f}")

pred_test = pred_test.withColumn(
    "readiness_band",
    F.when(F.col(band_source_col) >= F.lit(q_hi),  F.lit("High"))
     .when(F.col(band_source_col) >= F.lit(q_lo), F.lit("Medium"))
     .otherwise(F.lit("Low"))
)




# 9) metrics 
# Precision / Recall / F1 are for the clean-sheet class (cs = 1), the outcome being forecast
def model_metrics(name, raw_col, pred_col, prob_col):
    row = {"Model": name}
    for key, metric in [("ROC_AUC", "areaUnderROC"), ("PR_AUC", "areaUnderPR")]:
        row[key] = float(BinaryClassificationEvaluator(
            labelCol="cs", rawPredictionCol=raw_col, metricName=metric).evaluate(pred_test))
    for key, metric in [("Accuracy", "accuracy"), ("Precision", "precisionByLabel"),
                        ("Recall", "recallByLabel"), ("F1", "fMeasureByLabel")]:
        row[key] = float(MulticlassClassificationEvaluator(
            labelCol="cs", predictionCol=pred_col, metricName=metric, metricLabel=1.0).evaluate(pred_test))
    row["Brier"] = float(brier_score(pred_test, prob_col))
    return row

import pandas as pd
metrics_pd = pd.DataFrame([
    model_metrics("LR", "raw_lr", "prediction_lr", "p1_lr"),
    model_metrics("RF", "raw_rf", "prediction_rf", "p1_rf_calib"),
    model_metrics("Poisson", "p1_pois", "prediction_pois", "p1_pois"),
])
write_csv_pd(metrics_pd, "model_metrics.csv")




# 10) export predictions
keep_cols = [
    "Date", "Season", "Team", "Opponent", "Venue", "home", "cs",
    "p1_lr", "p1_rf", "p1_rf_calib", "prediction_lr", "prediction_rf", "readiness_band",
    "exp_goals_conceded", "p1_pois", "prediction_pois"
]
pred_pdf = (pred_test.select(*keep_cols)
            .withColumn("prediction_lr", F.col("prediction_lr").cast("int"))
            .withColumn("prediction_rf", F.col("prediction_rf").cast("int"))
            .withColumn("prediction_pois", F.col("prediction_pois").cast("int"))
            .toPandas())
pred_pdf.sort_values("Date", inplace=True)
write_csv_pd(pred_pdf, "model_predictions.csv")




# 11) rf feature importances 
attrs_meta = train_fe.schema["features"].metadata.get("ml_attr", {}).get("attrs", {})
feature_names = []
for section in ("numeric", "binary"):
    if section in attrs_meta:
        sorted_attrs = sorted(attrs_meta[section], key=lambda a: a["idx"])
        feature_names.extend(a["name"] for a in sorted_attrs)

importances = rf_model.featureImportances.toArray().tolist()
pairs = []
for i, imp in enumerate(importances):
    name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
    pairs.append((name, float(imp)))

fi_rows = sorted(pairs, key=lambda x: x[1], reverse=True)
fi_pd = pd.DataFrame(fi_rows, columns=["feature", "importance"])
write_csv_pd(fi_pd, "rf_feature_importances.csv")




# 12) dataset summary views 
view_td = (
    df.groupBy("cs").count()
        .select(
            F.lit("target_distribution").alias("view"),
            F.col("cs").cast("string").alias("key"),
            F.col("count").cast("double").alias("value"),
        )
)
view_ha = (
    df.groupBy("home")
        .agg(F.avg(F.col("cs").cast("double")).alias("value"))
        .select(
            F.lit("home_away_rate").alias("view"),
            F.when(F.col("home") == 1, "Home").otherwise("Away").alias("key"),
            F.col("value").cast("double").alias("value"),
        )
)
view_sc = (
    df.groupBy("Season").count()
      .select(
            F.lit("season_match_counts").alias("view"),
            F.col("Season").cast("string").alias("key"),
            F.col("count").cast("double").alias("value"),
        )
)
summary_df = view_td.unionByName(view_ha).unionByName(view_sc)
write_csv_pd(summary_df.toPandas(), "dataset_summary.csv")

print("Wrote 4 CSVs to:", CFG["OUT_DIR"])
spark.stop()
