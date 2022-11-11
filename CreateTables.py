# Databricks notebook source
# MAGIC %md
# MAGIC https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/

# COMMAND ----------

import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt"
df = pd.read_csv(url, sep="\t", header=None)
df.set_axis(df.columns + 1, axis=1,inplace=True)
df = df.add_prefix('col_')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("atchoudhury.ticdata2000")

# COMMAND ----------

eval = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt"
df_eval = pd.read_csv(eval, sep="\t", header=None)
df_eval.set_axis(df_eval.columns + 1, axis=1,inplace=True)
df_eval = df_eval.add_prefix('col_')

tgts = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt"
df_tgts = pd.read_csv(tgts, sep="\t", header=None)
df_tgts.rename(columns={0: "col_86"}, inplace=True)

result = pd.concat([df_eval, df_tgts], axis=1)
spark.createDataFrame(df_eval).write.mode("overwrite").saveAsTable("atchoudhury.ticeval2000")


# COMMAND ----------


