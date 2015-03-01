package com.integration.weka.spark.utils;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * Application entry point.
 * 
 * @author Moises
 *
 */
public class Launcher {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("SparkWekaIntegration");
		JavaSparkContext context = new JavaSparkContext(conf);

		String inputFile = "diabetes.csv";
		String attributesFile = "diabetes_attr.csv";
		String outputFile = "./target/output";

		// CSVHeaderSparkTask.loadCVSFile(conf, context, inputFile,
		// attributesFile, outputFile);
		// ClassifierSparkJob.buildClassifier(conf, context, inputFile,
		// attributesFile, outputFile);
	}

}
