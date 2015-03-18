package com.integration.weka.spark.utils;

import java.util.Date;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import com.integration.weka.spark.jobs.ClassifierSparkJob;
import com.integration.weka.spark.jobs.ScoringSparkJob;

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
		} else {
			String job = args[0];
			if (job.equals("CLASSIFY")) {
				// Lets assume for the time being everything is alright on input
				String classifierName = args[1];
				String trainingData = args[2];
				String attributesFile = args[3];
				String outputFile = "output_" + classifierName + "_" + String.valueOf(new Date().getTime()) + ".model";
				ClassifierSparkJob.buildClassifier(conf, context, classifierName, trainingData, attributesFile, outputFile);
			} else {
				if (job.equals("SCORE")) {
					String modelFile = args[1];
					String predictionData = args[2];
					String attributesFile = args[3];
					String outputFolder = "output_prediction_" + String.valueOf(new Date().getTime());
					ScoringSparkJob.scoreDataSet(conf, context, modelFile, predictionData, attributesFile, outputFolder);
				} else {
					LOGGER.error("Unknown JOB. Option are CLASSIFY or SCORE");
				}
			}
		}
	}
}
