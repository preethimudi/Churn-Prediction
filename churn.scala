import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.Pipeline

object ImprovedChurnPrediction {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Improved Churn Prediction")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Load the dataset
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/churn.csv")

    // Clean and transform data
    val cleanedDF = df.select(
      $"tenure".cast("int"),
      $"monthly_charges".cast("double"),
      $"total_charges".cast("double"),
      $"contract_type",
      $"payment_method",
      when($"churn" === "Yes", 1).otherwise(0).alias("label")
    )

    // Index and encode categorical features
    val contractIndexer = new StringIndexer().setInputCol("contract_type").setOutputCol("contract_index").setHandleInvalid("keep")
    val paymentIndexer = new StringIndexer().setInputCol("payment_method").setOutputCol("payment_index").setHandleInvalid("keep")

    val contractEncoder = new OneHotEncoder().setInputCol("contract_index").setOutputCol("contract_vec").setHandleInvalid("keep")
    val paymentEncoder = new OneHotEncoder().setInputCol("payment_index").setOutputCol("payment_vec").setHandleInvalid("keep")

    // Assemble features
    val assembler = new VectorAssembler()
      .setInputCols(Array("tenure", "monthly_charges", "total_charges", "contract_vec", "payment_vec"))
      .setOutputCol("features")

    // Define the model
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(100)

    // Build pipeline
    val pipeline = new Pipeline().setStages(Array(
      contractIndexer,
      paymentIndexer,
      contractEncoder,
      paymentEncoder,
      assembler,
      rf
    ))

    // Split data
    val Array(train, test) = cleanedDF.randomSplit(Array(0.8, 0.2), seed = 42)

    // Train model
    val model = pipeline.fit(train)

    // Make predictions
    val predictions = model.transform(test)

    // Evaluate model
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")

    val auc = evaluator.evaluate(predictions)
    println(s"Improved Model AUC: $auc")

    predictions.select("features", "label", "prediction", "probability").show(false)

    spark.stop()
  }
}
