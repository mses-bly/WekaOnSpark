package com.integration.weka.spark.utils;

import java.util.Arrays;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import com.integration.weka.spark.jobs.CSVHeaderSparkJob;
import com.integration.weka.spark.jobs.ClassifierSparkJob;
import com.integration.weka.spark.jobs.EvaluationSparkJob;
import com.integration.weka.spark.jobs.RandomShuffleJob;
import com.integration.weka.spark.jobs.ScoreSparkJob;

/**
 * Application entry point.
 * 
 * @author Moises
 *
 */
public class Launcher {

	private static Logger LOGGER = Logger.getLogger(Launcher.class);

	public static void main(String[] args) {
		System.out.println("*****************************************************************************************************************************************");
		SparkConf conf = new SparkConf().setAppName("SparkWekaIntegration");
		JavaSparkContext context = new JavaSparkContext(conf);
		if (args.length == 0) {
			LOGGER.error("Please provide some arguments");
			//			String trainDataInputFilePath = "/home/moises/Moises/Projects/WekaOnSpark/testing_files/datapolytest_reduced.csv";
			//			List<String> classifiers = new ArrayList<String>();
			//			classifiers.add("weka.classifiers.trees.RandomForest");
			//			launchKFoldClassifierEvaluationJob(conf, context, classifiers, 10, trainDataInputFilePath);
//			String[] input = { "-input-file", "/home/moises/Moises/Projects/WekaOnSpark/testing_files/diabetes.csv", "-folds", "10", "-shuffle", "true", "-classifier-name",
			//					"weka.classifiers.trees.RandomForest" };
			//			launchClassifierEvaluationJob(conf, context, input);
		} else {
			String job = args[0];
			switch (job) {
				case Constants.JOB_HEADER:
					launchHeaderJob(conf, context, Arrays.copyOfRange(args, 1, args.length));
					break;
				case Constants.JOB_CLASSIFY:
					launchClassifierJob(conf, context, Arrays.copyOfRange(args, 1, args.length));
					break;
				case Constants.JOB_EVALUATION:
					launchClassifierEvaluationJob(conf, context, Arrays.copyOfRange(args, 1, args.length));
					break;
				case Constants.JOB_SCORE:
					launchScoreJob(conf, context, Arrays.copyOfRange(args, 1, args.length));
					break;
				case Constants.JOB_SHUFFLE:
					launchRandomShuffleJob(conf, context, Arrays.copyOfRange(args, 1, args.length));
					break;
			}
		}
	}

	private static void launchHeaderJob(SparkConf conf, JavaSparkContext context, String[] options) {
		LOGGER.info("------- Launching header construction job -------");
		try {
			Options opts = Utils.parseOptions(options);
			CSVHeaderSparkJob csvHeaderSparkJob = new CSVHeaderSparkJob(context, opts);
			csvHeaderSparkJob.computeHeaderAndWriteToFile();
		} catch (Exception ex) {
			LOGGER.error("Could not complete HEADER job. Error: " + ex);
		}

		LOGGER.info("------- Finished header construction job -------");
	}

	private static void launchClassifierJob(SparkConf conf, JavaSparkContext context, String[] options) {
		LOGGER.info("------- Launching classifier training job -------");
		try {
			Options opts = Utils.parseOptions(options);
			ClassifierSparkJob.buildClassifier(context, opts);
		} catch (Exception ex) {
			LOGGER.error("Could not complete CLASSIFY job. Error: " + ex);
		}
		LOGGER.info("------- Finished classifier training job -------");
	}

	private static void launchScoreJob(SparkConf conf, JavaSparkContext context, String[] options) {
		LOGGER.info("------- Launching score job -------");
		try {
			Options opts = Utils.parseOptions(options);
			ScoreSparkJob.scoreDataSet(context, opts);
		} catch (Exception ex) {
			LOGGER.error("Could not complete SCORE job. Error: " + ex);
		}
		LOGGER.info("------- Finished score job -------");
	}

	private static void launchClassifierEvaluationJob(SparkConf conf, JavaSparkContext context, String[] options) {
		LOGGER.info("------- Launching evaluation job -------");
		try {
			Options opts = Utils.parseOptions(options);
			EvaluationSparkJob.evaluateClassifier(context, opts);
		} catch (Exception ex) {
			LOGGER.error("Could not complete EVALUATION job. Error: " + ex);
		}
		LOGGER.info("------- Finished evaluation job -------");
	}

	private static void launchRandomShuffleJob(SparkConf conf, JavaSparkContext context, String[] options) {
		LOGGER.info("------- Launching shuffle job -------");
		try {
			Options opts = Utils.parseOptions(options);
			RandomShuffleJob.randomlyShuffleData(context, opts);
		} catch (Exception ex) {
			LOGGER.error("Could not complete SHUFFLE job. Error: " + ex);
		}
		LOGGER.info("------- Finished shuffle job -------");
	}
}
