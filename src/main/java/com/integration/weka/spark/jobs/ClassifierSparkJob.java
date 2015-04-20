package com.integration.weka.spark.jobs;

import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.classifiers.Classifier;
import weka.core.Instances;

import com.integration.weka.spark.classifiers.ClassifierMapFunction;
import com.integration.weka.spark.classifiers.ClassifierReduceFunction;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Utils;

/**
 * Wrapper for launching a classification trainer.
 * 
 * @author Moises
 *
 */
public class ClassifierSparkJob {

	private static Logger LOGGER = Logger.getLogger(ClassifierSparkJob.class);

	/**
	 * 
	 * @param conf
	 *            : Spark configuration
	 * @param context
	 *            : Spark context
	 * @param classifierName
	 *            : Classifier full qualified name
	 * @param inputFile
	 *            : Path to input data.
	 * @param outputFile
	 *            : Output folder for model.
	 */
	public static void buildClassifier(SparkConf conf, JavaSparkContext context, String classifierName, String inputFile, String outputFile) {
		LOGGER.info("Training classifier with dataset [" + inputFile + "]");
		// Load the data file
		JavaRDD<String> csvFile = context.textFile(inputFile);

		// Group input data by partition rather than by lines
		JavaRDD<List<String>> data = csvFile.glom();

		// Build Weka Header
		Instances header = data.map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());

		// Train classifier
		Classifier classifier = data.map(new ClassifierMapFunction(header, classifierName)).reduce(new ClassifierReduceFunction());
		try {
			Utils.writeModelToDisk(outputFile, classifier);
			LOGGER.info("Classifier [" + classifier.getClass().getName() + "] model saved at [" + outputFile + "]");
		} catch (Exception e) {
			LOGGER.error("Could not write model to disk. Error: [" + e + "]");
		}
	}

}
