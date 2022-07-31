from spark_init import sc,spark
import pyspark.sql.functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DecimalType, FloatType
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("sex", IntegerType(), True),
    StructField("cp", IntegerType(), True),
    StructField("trestbps", IntegerType(), True),
    StructField("chol", IntegerType(), True),
    StructField("fbs", IntegerType(), True),
    StructField("restecg", IntegerType(), True),
    StructField("thalach", IntegerType(), True),
    StructField("exang", IntegerType(), True),
    StructField("oldpeak", DecimalType(), True),
    StructField("slope", IntegerType(), True),
    StructField("ca", IntegerType(), True),
    StructField("thal", IntegerType(), True),
    StructField("target", IntegerType(), True)
    ])
df = spark.read.option("multiline","true").option("header", "true").csv("data/mllib/heart.csv",schema=schema)
df_target = df.groupby("target").count()
df_sex = df.groupby(["sex","target"]).count()
vect = VectorAssembler(inputCols=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'], outputCol="features")

transofrmedOutput=vect.transform(df)

final_data = transofrmedOutput.select("features", "target").withColumnRenamed("target","label")
trainData, testData = final_data.randomSplit([0.7, 0.3])
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(trainData)
predictions = rfModel.transform(testData)
predictions.select('rawPrediction', 'prediction', 'probability').show(10)
predictions.select("label", "prediction").show(10)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))
preds_and_labels = predictions.select(['prediction','label']).withColumn('label', F.col('label').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())
newRes = preds_and_labels.withColumn("prediction",preds_and_labels["prediction"].cast(DoubleType())).withColumn("label",preds_and_labels["label"].cast(DoubleType()))
mMetrics = MulticlassMetrics(preds_and_labels.rdd)
#F measure
metrics.fMeasure(0.0, 2.0)
print(metrics.accuracy)
print(metrics.weightedFalsePositiveRate)
print(metrics.weightedPrecision)
print(metrics.weightedFMeasure())


