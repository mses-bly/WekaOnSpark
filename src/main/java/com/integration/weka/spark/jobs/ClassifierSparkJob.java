package com.integration.weka.spark.jobs;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;

import com.integration.weka.spark.classifiers.ClassifierMapFunction;
import com.integration.weka.spark.classifiers.ClassifierReduceFunction;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Constants;
import com.integration.weka.spark.utils.IntegerPartitioner;
import com.integration.weka.spark.utils.Options;
import com.integration.weka.spark.utils.Utils;

/**
 * Build classifier from CSV file.
 * 
 * @author Moises
 *
 */
public class ClassifierSparkJob {

	private static Logger LOGGER = Logger.getLogger(ClassifierSparkJob.class);

	public static void buildClassifier(SparkConf conf, JavaSparkContext context, Options opts) throws Exception {

		if (!opts.hasOption(Constants.OPTION_INPUT_FILE)) {
			throw new Exception("Must provide an input file for CLASSIFY job");
		}

		if (!opts.hasOption(Constants.OPTION_CLASSIFIER_NAME)) {
			throw new Exception("Must provide a classifier to train");
		}

		String inputFilePath = opts.getOption(Constants.OPTION_INPUT_FILE);
		String classifierFullName = opts.getOption(Constants.OPTION_CLASSIFIER_NAME);
		// Load the data file
		JavaRDD<String> csvFile = context.textFile(inputFilePath);

		// Group input data by partition rather than by lines
		JavaRDD<List<String>> trainingData = csvFile.glom();

		// Build Weka Header
		Instances header = trainingData.map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());

		// Train one classifier per fold
		JavaPairRDD<Integer, Classifier> classifierPerFold = csvFile.mapPartitionsToPair(new ClassifierMapFunction(header, classifierFullName, 1));
		classifierPerFold = classifierPerFold.partitionBy(new IntegerPartitioner(1));

		JavaPairRDD<Integer, Classifier> reducedByFold = classifierPerFold.mapPartitionsToPair(new ClassifierReduceFunction());
		List<Classifier> classifier = new ArrayList<Classifier>(1);
		List<Tuple2<Integer, Classifier>> aggregated = reducedByFold.collect();
		for (Tuple2<Integer, Classifier> t : aggregated) {
			classifier.add(t._1(), t._2());
		}
		String outputFilePath = "classifier_" + classifierFullName + "_" + Utils.getDateAsStringFormat(new Date(), "YYYY-MM-dd_kk:mm:ss") + ".model";
		SerializationHelper.write(outputFilePath, classifier.get(0));
		LOGGER.info("Classifier [" + classifier.get(0).getClass().getName() + "] model saved at [" + outputFilePath + "]");
	}

}
