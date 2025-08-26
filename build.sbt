name := "customerchurning"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.13.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.1",
  "org.apache.spark" %% "spark-sql" % "3.3.1",
  "org.apache.spark" %% "spark-mllib" % "3.3.1",
  "org.apache.spark" %% "spark-streaming" % "3.3.1",
  "org.twitter4j" % "twitter4j-core" % "4.0.4",
  "org.twitter4j" % "twitter4j-stream" % "4.0.4"
)
