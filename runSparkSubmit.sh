#!/bin/bash
SPARK_HOME=/home/moises/Moises/Installs/spark-1.2.1-bin-hadoop2.4/bin
LAUNCHER_CLASS=com.integration.weka.spark.utils.Launcher
WEKA_JAR_PATH=./target
CLASSIFIER=weka.classifiers.bayes.NaiveBayes
INPUT_FILES_PATH=/home/moises/Moises/Projects/WekaOnSpark/testing_files

#Execution 1
$SPARK_HOME/spark-submit --class $LAUNCHER_CLASS $OPTIONS $WEKA_JAR_PATH/integration-weka-spark-0.0.1-SNAPSHOT.jar $CLASSIFIER $INPUT_FILES_PATH/diabetes.csv $INPUT_FILES_PATH/diabetes_attr.csv
