package com.integration.weka.spark.utils;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import com.integration.weka.spark.jobs.CSVHeaderSparkJob;
import com.integration.weka.spark.jobs.ClassifierSparkJob;
import com.integration.weka.spark.jobs.EvaluationSparkJob;
import com.integration.weka.spark.jobs.KFoldClassifierSparkJob;
import com.integration.weka.spark.jobs.ScoreSparkJob;

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
			//			String trainDataInputFilePath = "/home/moises/Moises/Projects/WekaOnSpark/testing_files/datapolytest_reduced.csv";
			//			launchKFoldClassifierJob(conf, context, "weka.classifiers.trees.RandomForest", trainDataInputFilePath, 10);
		} else {
			String job = args[0];
			if (job.equals("HEADER")) {
				//SparkConf conf, JavaSparkContext context, String inputFilePath
				launchHeaderJob(conf, context, args[1]);
			} else {
				if (job.equals("CLASSIFY")) {
					//String classifierFullName, String inputFilePath
					launchClassifierJob(conf, context, args[1], args[2]);
				} else {
					if (job.equals("SCORE")) {
						//String modelFilePath, String inputFilePath, String trainFilePath
						launchScoreJob(conf, context, args[1], args[2], args[3]);
					} else {
						if (job.equals("EVALUATION")) {
							//List<String> classifierFullNameList, String trainDataInputFilePath, String evaluaeDataInputFilePath
							List<String> classifiersToEvaluate = new ArrayList<String>();
							int nClassifiers = Integer.valueOf(args[1]);
							for (int i = 2; i <= nClassifiers + 1; i++) {
								classifiersToEvaluate.add(args[i]);
							}
							String trainDataInputFilePath = args[nClassifiers + 2];
							String evaluateDataInputFilePath = args[nClassifiers + 3];
							launchClassifierEvaluationJob(conf, context, classifiersToEvaluate, trainDataInputFilePath, evaluateDataInputFilePath);

						} else {
							if (job.equals("KFOLD_CLASSIFY")) {
								//String classifierFullName, String inputFilePath, int kFolds
								launchKFoldClassifierJob(conf, context, args[1], args[2], Integer.valueOf(args[3]));
							} else {
								LOGGER.error("Unknown JOB.");
							}
						}
					}
				}
			}
		}
	}

	private static void launchHeaderJob(SparkConf conf, JavaSparkContext context, String inputFilePath) {
		LOGGER.info("------- Launching header construction job -------");
		String outputFilePath = "output_header_" + Utils.getDateAsStringFormat(new Date(), "yyyymmddhhmmss") + ".header";
		CSVHeaderSparkJob.loadCVSFile(conf, context, inputFilePath, outputFilePath);
		LOGGER.info("------- Finished header construction job -------");
	}

	private static void launchClassifierJob(SparkConf conf, JavaSparkContext context, String classifierFullName, String inputFilePath) {
		LOGGER.info("------- Launching classifier training job -------");
		String outputFilePath = "output_model_" + classifierFullName + "_" + Utils.getDateAsStringFormat(new Date(), "yyyymmddhhmmss") + ".model";
		ClassifierSparkJob.buildClassifier(conf, context, classifierFullName, inputFilePath, outputFilePath);
		LOGGER.info("------- Finished classifier training job -------");
	}

	private static void launchScoreJob(SparkConf conf, JavaSparkContext context, String modelFilePath, String inputFilePath, String trainFilePath) {
		LOGGER.info("------- Launching score job -------");
		String outputFilePath = "output_score_" + Utils.getDateAsStringFormat(new Date(), "yyyymmddhhmmss") + ".score";
		ScoreSparkJob.scoreDataSet(conf, context, modelFilePath, inputFilePath, trainFilePath, outputFilePath);
		LOGGER.info("------- Finished score job -------");
	}

	private static void launchClassifierEvaluationJob(SparkConf conf, JavaSparkContext context, List<String> classifierFullNameList, String trainDataInputFilePath, String evaluateDataInputFilePath) {
		LOGGER.info("------- Launching evaluation job -------");
		String outputFilePath = "output_evaluation_" + Utils.getDateAsStringFormat(new Date(), "yyyymmddhhmmss") + ".evaluation";
		EvaluationSparkJob.evaluate(conf, context, classifierFullNameList, trainDataInputFilePath, evaluateDataInputFilePath, outputFilePath);
		LOGGER.info("------- Finished evaluation job -------");
	}

	private static void launchKFoldClassifierJob(SparkConf conf, JavaSparkContext context, String classifierFullName, String inputFilePath, int kFolds) {
		LOGGER.info("------- Launching kFold classifier training job -------");
		String outputFilePathPrefix = "output_model_" + classifierFullName + "_" + Utils.getDateAsStringFormat(new Date(), "yyyymmddhhmmss");
		KFoldClassifierSparkJob.buildClassifiers(conf, context, classifierFullName, kFolds, inputFilePath, outputFilePathPrefix);
		LOGGER.info("------- Finished kFold classifier training job -------");
	}

}
