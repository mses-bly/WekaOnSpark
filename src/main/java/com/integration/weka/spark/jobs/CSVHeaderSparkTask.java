package com.integration.weka.spark.jobs;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.core.Instances;

import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.utils.Utils;

public class CSVHeaderSparkTask {
	/**
	 * 
	 * @param conf
	 *            Spark configuration
	 * @param context
	 *            Spark context
	 * @param inputFile
	 *            File containing the data
	 * @param attributesNamesFile
	 *            File containing the attribute names.
	 * @param outputFile
	 *            file to which save the results
	 */
	public static void loadCVSFile(SparkConf conf, JavaSparkContext context, String inputFile,
			String attributesNamesFile, String outputFile) {
		JavaRDD<String> csvFile = context.textFile(inputFile);
		JavaRDD<String[]> attributes = context.textFile(attributesNamesFile).map(Utils.getParseLineFunction());
		JavaRDD<Instances> instances = csvFile.glom().map(new CSVHeaderMapFunction(Arrays.asList(attributes.first())));
		// Instances inst = instances.reduce(new CSVHeaderReduceFunction());
		// System.out.println(inst);
		instances.saveAsTextFile(outputFile);
	}

}