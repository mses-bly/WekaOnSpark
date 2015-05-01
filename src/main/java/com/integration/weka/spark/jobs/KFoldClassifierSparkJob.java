package com.integration.weka.spark.jobs;

import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;

import com.integration.weka.spark.classifiers.KFoldClassifierMapFunction;
import com.integration.weka.spark.classifiers.KFoldClassifierReduceFunction;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Utils;

public class KFoldClassifierSparkJob {

	private static Logger LOGGER = Logger.getLogger(KFoldClassifierSparkJob.class);

	/**
	 * 
	 * @param conf: Spark configuration
	 * @param context: Spark context
	 * @param classifierFullName: Classifier full name
	 * @param kFolds: Number of folds
	 * @param inputFilePath: Input file path.
	 * @param outputFilePathPrefix: output models names prefix.
	 */
	public static void buildClassifiers(SparkConf conf, JavaSparkContext context, String classifierFullName, int kFolds, String inputFilePath, String outputFilePathPrefix) {
		// Load the data file
		JavaRDD<String> csvFile = context.textFile(inputFilePath);

		// Group input data by partition rather than by lines
		JavaRDD<List<String>> data = csvFile.glom();

		// Build Weka Header
		Instances header = data.map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());
		
		List<Classifier> classifiers = data.map(new KFoldClassifierMapFunction(header, classifierFullName, kFolds)).reduce(new KFoldClassifierReduceFunction(kFolds));
		for (int i = 0; i < kFolds; i++) {
			String outputFilePath = outputFilePathPrefix + "_fold_" + String.valueOf(i + 1) + ".model";
			try {
				SerializationHelper.write(outputFilePath, classifiers.get(i));
				LOGGER.info("Classifier [" + classifiers.get(i).getClass().getName() + "] model saved at [" + outputFilePath + "]");
			} catch (Exception e) {
				LOGGER.error("Could not write models to disk. Error: [" + e + "]");
			}
		}

	}

}
