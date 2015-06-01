package com.integration.weka.spark.jobs;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.AggregateableEvaluation;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

import com.integration.weka.spark.classifiers.ClassifierMapFunction;
import com.integration.weka.spark.classifiers.ClassifierReduceFunction;
import com.integration.weka.spark.data.Dataset;
import com.integration.weka.spark.evaluation.EvaluationMapFunction;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Constants;
import com.integration.weka.spark.utils.IntegerPartitioner;
import com.integration.weka.spark.utils.Options;
import com.integration.weka.spark.utils.Utils;

/**
 * K-Fold evaluation of classifiers.
 * @author Moises
 *
 */

public class EvaluationSparkJob {
	private static Logger LOGGER = Logger.getLogger(EvaluationSparkJob.class);

	public static void evaluateClassifier(JavaSparkContext context, Options opts) throws Exception {
		if (!opts.hasOption(Constants.OPTION_INPUT_FILE)) {
			throw new Exception("Must provide training data file for EVALUATION job");
		}
		if (!opts.hasOption(Constants.OPTION_TEST_FILE) && !opts.hasOption(Constants.OPTION_FOLDS)) {
			throw new Exception("Must provide a test file or perform cross valdation for EVALUATION job");
		}
		if (!opts.hasOption(Constants.OPTION_CLASSIFIER_NAME)) {
			throw new Exception("Must provide a classifier name for EVALUATION job");
		}
		Evaluation evaluation = null;
		int nFolds = 1;
		//Ignore cross validation, use test file for evaluate classifier
		if (opts.hasOption(Constants.OPTION_TEST_FILE)) {
			evaluation = evaluateOnTestFile(context, opts.getOption(Constants.OPTION_CLASSIFIER_NAME), opts.getOption(Constants.OPTION_INPUT_FILE), opts.getOption(Constants.OPTION_TEST_FILE));
		} else {
			// perform cross validation
			nFolds = Integer.valueOf(opts.getOption(Constants.OPTION_FOLDS));
			JavaRDD<String> data = null;
			if (opts.hasOption(Constants.OPTION_SHUFFLE)) {
				LOGGER.info("------- Launching shuffle job -------");
				data = RandomShuffleJob.randomlyShuffleData(context, opts);
				LOGGER.info("------- Finished shuffle job -------");
			} else {
				data = context.textFile(opts.getOption(Constants.OPTION_INPUT_FILE));
			}
			data.persist(StorageLevel.MEMORY_AND_DISK());
			evaluation = crossValidation(context, opts.getOption(Constants.OPTION_CLASSIFIER_NAME), nFolds, data);

		}
		if (evaluation != null) {
			String outputFilePath = "evaluation_" + opts.getOption(Constants.OPTION_CLASSIFIER_NAME) + ".txt";
			PrintWriter writer = new PrintWriter(outputFilePath, "UTF-8");
			writer.println("======== Evaluation for classifier: " + opts.getOption(Constants.OPTION_CLASSIFIER_NAME) + " ========");

			writer.println("======== Number of folds:" + nFolds + " ========");
			writer.println(evaluation.toSummaryString(true));
			writer.close();
			LOGGER.info("Score for [" + opts.getOption(Constants.OPTION_CLASSIFIER_NAME) + "] saved at [" + outputFilePath + "]");
		}
	}

	private static Evaluation evaluateOnTestFile(JavaSparkContext context, String classifierName, String trainDataInputFilePath, String evaluationFilePath) throws Exception {
		
		Dataset trainingData = new CSVHeaderSparkJob(context, trainDataInputFilePath).createDataSet(false);
		// Train one classifier per fold
		JavaPairRDD<Integer, Classifier> classifierPerFold = trainingData.getData().mapPartitionsToPair(new ClassifierMapFunction(trainingData.getHeaderNoSummary(), classifierName, 1));
		classifierPerFold = classifierPerFold.partitionBy(new IntegerPartitioner(1));
		JavaPairRDD<Integer, Classifier> reducedByFold = classifierPerFold.mapPartitionsToPair(new ClassifierReduceFunction());
		List<Classifier> kFoldClassifiers = new ArrayList<Classifier>(1);
		List<Tuple2<Integer, Classifier>> aggregated = reducedByFold.collect();
		for (Tuple2<Integer, Classifier> t : aggregated) {
			kFoldClassifiers.add(t._1(), t._2());
		}

		Dataset evaluationData = new CSVHeaderSparkJob(context, evaluationFilePath).createDataSet(false);
		//Evaluate on file
		List<Evaluation> evaluations = evaluationData.getData().mapPartitions(new EvaluationMapFunction(evaluationData.getHeaderWithSummary(), kFoldClassifiers, 1)).collect();
		AggregateableEvaluation aggregateableEvaluation = new AggregateableEvaluation(evaluations.get(0));
		for (Evaluation eval : evaluations) {
			aggregateableEvaluation.aggregate(eval);
		}
		return aggregateableEvaluation;
	}

	private static Evaluation crossValidation(JavaSparkContext context, String classifierName, int kFolds, JavaRDD<String> data) throws Exception {
//		// Group input data by partition rather than by lines
//		JavaRDD<List<String>> trainingData = data.glom();
//		// Build Weka Header
//		Instances header = trainingData.map(new CSVHeaderMapFunction(Utils.parseCSVLine(data.first()).length)).reduce(new CSVHeaderReduceFunction());
//		// Train one classifier per fold
//		JavaPairRDD<Integer, Classifier> classifierPerFold = data.mapPartitionsToPair(new ClassifierMapFunction(header, classifierName, kFolds));
//		classifierPerFold.persist(StorageLevel.MEMORY_AND_DISK());
//		data.unpersist();
//		classifierPerFold = classifierPerFold.sortByKey();
//		classifierPerFold = classifierPerFold.partitionBy(new IntegerPartitioner(kFolds));
//		classifierPerFold.persist(StorageLevel.MEMORY_AND_DISK());
//		JavaPairRDD<Integer, Classifier> reducedByFold = classifierPerFold.mapPartitionsToPair(new ClassifierReduceFunction());
//		List<Classifier> kFoldClassifiers = new ArrayList<Classifier>(kFolds);
//		List<Tuple2<Integer, Classifier>> aggregated = reducedByFold.collect();
//		for (Tuple2<Integer, Classifier> t : aggregated) {
//			kFoldClassifiers.add(t._1(), t._2());
//		}
//		List<Evaluation> evaluations = trainingData.map(new EvaluationMapFunction(header, kFoldClassifiers, kFolds)).collect();
//		AggregateableEvaluation aggregateableEvaluation = new AggregateableEvaluation(evaluations.get(0));
//		for (Evaluation eval : evaluations) {
//			aggregateableEvaluation.aggregate(eval);
//		}
//		return aggregateableEvaluation;
		return null;
	}
}
