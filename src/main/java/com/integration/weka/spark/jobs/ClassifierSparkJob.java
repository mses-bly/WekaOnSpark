package com.integration.weka.spark.jobs;

import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;

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
	 * @param classifierFullName
	 *            : Classifier full qualified name
	 * @param inputFilePath
	 *            : Path to input data.
	 * @param outputFilePath
	 *            : Output path for model.
	 */
	public static void buildClassifier(SparkConf conf, JavaSparkContext context, String classifierFullName, String inputFilePath, String outputFilePath) {
		// Load the data file
		JavaRDD<String> csvFile = context.textFile(inputFilePath);

		// Group input data by partition rather than by lines
		JavaRDD<List<String>> data = csvFile.glom();

		// Build Weka Header
		Instances header = data.map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());

		// Train classifier
		Classifier classifier = data.map(new ClassifierMapFunction(header, classifierFullName)).reduce(new ClassifierReduceFunction());
		try {
			SerializationHelper.write(outputFilePath, classifier);
			LOGGER.info("Classifier [" + classifier.getClass().getName() + "] model saved at [" + outputFilePath + "]");
		} catch (Exception e) {
			LOGGER.error("Could not write model to disk. Error: [" + e + "]");
		}
	}

}
