package com.integration.weka.spark.utils;

import java.util.Date;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import com.integration.weka.spark.jobs.ClassifierSparkJob;

/**
 * Application entry point.
 * 
 * @author Moises
 *
 */
public class Launcher {

	private static Logger LOGGER = Logger.getLogger(Launcher.class);

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("SparkWekaIntegration");
		JavaSparkContext context = new JavaSparkContext(conf);

		if (args.length == 0) {
			LOGGER.error("Please provide some arguments");
			return;
		}
		String classifierName = args[0];
		if (classifierName == null) {
			LOGGER.error("Classifier name not provided");
			return;
		}

		String inputFile = args[1];
		if (inputFile == null) {
			LOGGER.error("Please provide an input CSV file");
			return;
		}

		String attributesFile = args[2];
		if (attributesFile == null) {
			LOGGER.error("Please provide an attributes CSV file");
			return;
		}

		String outputFile = "output_" + classifierName + "_" + String.valueOf(new Date().getTime()) + ".model";

		ClassifierSparkJob.buildClassifier(conf, context, classifierName, inputFile, attributesFile, outputFile);
	}
}
