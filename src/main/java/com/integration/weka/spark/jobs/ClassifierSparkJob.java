package com.integration.weka.spark.jobs;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.classifiers.Classifier;
import weka.core.Instances;

import com.integration.weka.spark.classifiers.ClassifierMapFunction;
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

	/**
	 * 
	 * @param conf
	 *            Spark configuration
	 * @param context
	 *            Spark context
	 * @param inputFile
	 *            CSV Training set (without headers)
	 * @param attributesNamesFile
	 *            Attributes names
	 * @param outputFile
	 *            File to write the trained model
	 */
	public static void buildClassifier(SparkConf conf, JavaSparkContext context, String inputFile, String attributesNamesFile, String outputFile) {

		// Load the data file
		JavaRDD<String> csvFile = context.textFile(inputFile);
		// Load the attributes file
		JavaRDD<String[]> attributes = context.textFile(attributesNamesFile).map(Utils.getParseLineFunction());
		// Group input data by partition rather than by lines
		JavaRDD<List<String>> data = csvFile.glom();
		// Build Weka Header
		Instances header = data.map(new CSVHeaderMapFunction(Arrays.asList(attributes.first()))).reduce(new CSVHeaderReduceFunction());
		// Train classifier
		JavaRDD<Classifier> classifier = data.map(new ClassifierMapFunction(header));
		classifier.saveAsTextFile(outputFile);
	}

}
