/* SimpleApp.scala */
import com.micvog.ml.{AnomalyDetection, FeaturesParser}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object MainRun {

  val rawFilePath = "./src/test/resources/training.csv"  // Training set
  val cvFilePath = "./src/test/resources/cross_val.csv"  // cross validation set

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("Anomaly Detection Spark").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val rawdata = sc.textFile(rawFilePath, 2).cache()
    val cvData = sc.textFile(cvFilePath, 2).cache()
    
    var start = System.currentTimeMillis
    
    //convert raw data to vectors
    println("Create vectors")
    val trainingVec: RDD[Vector] = FeaturesParser.parseFeatures(rawdata)
    val cvLabeledVec: RDD[LabeledPoint] = FeaturesParser.parseFeaturesWithLabel(cvData)

    val data = trainingVec.cache()
    var stop = System.currentTimeMillis
    println(".. in %s ms".format(stop - start))
    
    val anDet: AnomalyDetection = new AnomalyDetection()
    //derive model
    println("Derive model")
    start = System.currentTimeMillis
    val model = anDet.run(data)

    val dataCvVec = cvLabeledVec.cache()
    val optimalModel = anDet.optimize(dataCvVec, model)
    stop = System.currentTimeMillis
    println(".. in %s ms".format(stop - start))
    
    //find outliers in CV
    println("Find outliers")
    start = System.currentTimeMillis
    val cvVec = cvLabeledVec.map(_.features)
    val results = optimalModel.predict(cvVec)
    val outliers = results.filter(_._2).collect()
    outliers.foreach(v => println(v._1))
    println("\nFound %s outliers\n".format(outliers.length))
    stop = System.currentTimeMillis
    println(".. in %s ms".format(stop - start))
  }

}