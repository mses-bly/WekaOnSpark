package com.integration.weka.spark.jobs;

import java.io.PrintWriter;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

import com.integration.weka.spark.classifiers.ClassifierMapFunction;
import com.integration.weka.spark.classifiers.ClassifierReduceFunction;
import com.integration.weka.spark.evaluation.EvaluationMapFunction;
import com.integration.weka.spark.evaluation.EvaluationReduceFunction;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Utils;

/**
 * Evaluation of classifiers.
 * 
 * @author Moises
 *
 */
public class EvaluationSparkJob {
	private static Logger LOGGER = Logger.getLogger(EvaluationSparkJob.class);

	public static void evaluate(SparkConf conf, JavaSparkContext context, List<String> classifierFullNameList, String trainDataInputFilePath, String evaluateDataInputFilePath, String outputFilePath) {
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(outputFilePath, "UTF-8");
			for (String classifierName : classifierFullNameList) {
				Evaluation evaluation = evaluateSingleClassifier(conf, context, classifierName, trainDataInputFilePath, evaluateDataInputFilePath);
				writer.println("======== Evaluation for classifier: " + classifierName + " ========");
				writer.println(evaluation.toSummaryString(true));
			}
		} catch (Exception ex) {
			LOGGER.error("Could not complete evaluation job. Error: [" + ex + "]");
		} finally {
			if (writer != null) {
				writer.close();
			}
		}
	}

	private static Evaluation evaluateSingleClassifier(SparkConf conf, JavaSparkContext context, String classifierName, String trainDataInputFilePath, String evaluateDataInputFilePath) {
		try {
			// Load the data file
			JavaRDD<String> csvFile = context.textFile(trainDataInputFilePath);

			// Group input data by partition rather than by lines
			JavaRDD<List<String>> trainingData = csvFile.glom();

			// Build Weka Header
			Instances header = trainingData.map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());

			// Train classifier
			Classifier classifier = trainingData.map(new ClassifierMapFunction(header, classifierName)).reduce(new ClassifierReduceFunction());

			// Evaluation of the classifier
			csvFile = context.textFile(evaluateDataInputFilePath);
			JavaRDD<List<String>> evaluationData = csvFile.glom();
			Evaluation evaluation = evaluationData.map(new EvaluationMapFunction(classifier, header)).reduce(new EvaluationReduceFunction());
			return evaluation;
		} catch (Exception ex) {
			LOGGER.error("Could not complete evaluation job for classifier [" + classifierName + "]. Error: [" + ex + "]");
		}
		return null;
	}
}
