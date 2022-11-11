# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC This notebook performs exploratory data analysis on the dataset.
# MAGIC To expand on the analysis, attach this notebook to the **Dev-002-ML** cluster,
# MAGIC edit [the options of pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/advanced_usage.html), and rerun it.
# MAGIC - Explore completed trials in the [MLflow experiment](#mlflow/experiments/4418363507156853/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/4418363507156854) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC 
# MAGIC Runtime Version: _10.4.x-cpu-ml-scala2.12_

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd
import databricks.automl_runtime

from mlflow.tracking import MlflowClient

# Download input data from mlflow into a pandas DataFrame
# Create temporary directory to download data
temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(temp_dir)

# Download the artifact and read it
client = MlflowClient()
training_data_path = client.download_artifacts("40f53136e33e46929f578dd65c690a6f", "data", temp_dir)
df = pd.read_parquet(os.path.join(training_data_path, "training_data"))

# Delete the temporary data
shutil.rmtree(temp_dir)

target_col = "col_86"

# Convert columns detected to be of semantic type categorical
categorical_columns = ["col_2", "col_3", "col_4", "col_5", "col_6", "col_7", "col_8", "col_9", "col_10", "col_11", "col_12", "col_13", "col_14", "col_15", "col_16", "col_17", "col_18", "col_19", "col_20", "col_21", "col_22", "col_23", "col_24", "col_25", "col_26", "col_27", "col_28", "col_29", "col_30", "col_31", "col_32", "col_33", "col_34", "col_35", "col_36", "col_37", "col_38", "col_39", "col_40", "col_41", "col_42", "col_43", "col_44", "col_45", "col_46", "col_47", "col_48", "col_49", "col_50", "col_51", "col_52", "col_53", "col_54", "col_55", "col_56", "col_57", "col_58", "col_59", "col_60", "col_61", "col_62", "col_63", "col_64", "col_65", "col_66", "col_67", "col_68", "col_69", "col_70", "col_71", "col_72", "col_73", "col_74", "col_75", "col_76", "col_77", "col_78", "col_79", "col_80", "col_81", "col_82", "col_83", "col_84", "col_85"]
df[categorical_columns] = df[categorical_columns].applymap(str)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Semantic Type Detection Alerts
# MAGIC 
# MAGIC For details about the definition of the semantic types and how to override the detection, see
# MAGIC [Databricks documentation on semantic type detection](https://docs.microsoft.com/azure/databricks/applications/machine-learning/automl#semantic-type-detection).
# MAGIC 
# MAGIC - Semantic type `categorical` detected for columns `col_10`, `col_11`, `col_12`, `col_13`, `col_14`, `col_15`, `col_16`, `col_17`, `col_18`, `col_19`, `col_2`, `col_20`, `col_21`, `col_22`, `col_23`, `col_24`, `col_25`, `col_26`, `col_27`, `col_28`, `col_29`, `col_3`, `col_30`, `col_31`, `col_32`, `col_33`, `col_34`, `col_35`, `col_36`, `col_37`, `col_38`, `col_39`, `col_4`, `col_40`, `col_41`, `col_42`, `col_43`, `col_44`, `col_45`, `col_46`, `col_47`, `col_48`, `col_49`, `col_5`, `col_50`, `col_51`, `col_52`, `col_53`, `col_54`, `col_55`, `col_56`, `col_57`, `col_58`, `col_59`, `col_6`, `col_60`, `col_61`, `col_62`, `col_63`, `col_64`, `col_65`, `col_66`, `col_67`, `col_68`, `col_69`, `col_7`, `col_70`, `col_71`, `col_72`, `col_73`, `col_74`, `col_75`, `col_76`, `col_77`, `col_78`, `col_79`, `col_8`, `col_80`, `col_81`, `col_82`, `col_83`, `col_84`, `col_85`, `col_9`. Training notebooks will encode features based on categorical transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Results

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(df, minimal=True, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)
