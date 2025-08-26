import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.Pipeline

object ChurnPrediction {
  def main(args: Array[String]): Unit = {
    // Step 1: Start Spark session
    val spark = SparkSession.builder()
      .appName("Customer Churn Prediction")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Step 2: Load dataset
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/churn.csv")

    // Step 3: Data cleaning and transformation
    val cleanedDF = df.select(
      $"customer_id",
      $"tenure".cast("int"),
      $"monthly_charges".cast("double"),
      $"total_charges",
      $"contract_type",
      $"payment_method",
      when($"churn" === "Yes", 1).otherwise(0).alias("churn")
    )

    // Step 4: Handle categorical features using Pipeline
    val contractIndexer = new StringIndexer()
      .setInputCol("contract_type")
      .setOutputCol("contract_index")
      .setHandleInvalid("keep")

    val paymentIndexer = new StringIndexer()
      .setInputCol("payment_method")
      .setOutputCol("payment_index")
      .setHandleInvalid("keep")

    val contractEncoder = new OneHotEncoder()
      .setInputCol("contract_index")
      .setOutputCol("contract_vec")
      .setHandleInvalid("keep")

    val paymentEncoder = new OneHotEncoder()
      .setInputCol("payment_index")
      .setOutputCol("payment_vec")
      .setHandleInvalid("keep")

    val assembler = new VectorAssembler()
      .setInputCols(Array("tenure", "monthly_charges", "contract_vec", "payment_vec"))
      .setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(
      contractIndexer,
      paymentIndexer,
      contractEncoder,
      paymentEncoder,
      assembler
    ))

    val finalDF = pipeline.fit(cleanedDF).transform(cleanedDF).select("features", "churn")

    // Step 5: Split data
    val Array(train, test) = finalDF.randomSplit(Array(0.8, 0.2), seed = 42)

    // Step 6: Train model
    val lr = new LogisticRegression()
      .setLabelCol("churn")
      .setFeaturesCol("features")

    val model = lr.fit(train)

    // Step 7: Make predictions
    val predictions = model.transform(test)

    // Step 8: Evaluate model
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("churn")
      .setMetricName("areaUnderROC")

    val auc = evaluator.evaluate(predictions)
    println(s"Model AUC: $auc")

    // Step 9: Show predictions
    predictions.select("features", "churn", "prediction", "probability").show()
    cleanedDF.show()

    // Step 10: Stop Spark
    spark.stop()
  }
}
