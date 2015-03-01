package com.integration.weka.spark.jobs;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.core.Instances;

import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.utils.Utils;

/**
 * Wrapper for launching a header build job.
 * 
 * @author Moises
 *
 */
public class CSVHeaderSparkJob {
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
	public static void loadCVSFile(SparkConf conf, JavaSparkContext context, String inputFile, String attributesNamesFile, String outputFile) {
		JavaRDD<String> csvFile = context.textFile(inputFile);
		JavaRDD<String[]> attributes = context.textFile(attributesNamesFile).map(Utils.getParseLineFunction());
		JavaRDD<Instances> instances = csvFile.glom().map(new CSVHeaderMapFunction(Arrays.asList(attributes.first())));
		// Instances inst = instances.reduce(new CSVHeaderReduceFunction());
		// System.out.println(inst);
		instances.saveAsTextFile(outputFile);
	}

}