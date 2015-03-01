package com.integration.weka.spark.jobs;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.classifiers.functions.SimpleLogistic;
import weka.core.Instances;
import weka.distributed.DistributedWekaException;

import com.integration.weka.spark.classifiers.ClassifierMapFunction;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Utils;

public class ClassifierSparkTask {

	public static void buildClassifier(SparkConf conf, JavaSparkContext context, String inputFile,
			String attributesNamesFile, String outputFile) throws DistributedWekaException {
		// Load the data file
		JavaRDD<String> csvFile = context.textFile(inputFile);
		// Load the attributes file
		JavaRDD<String[]> attributes = context.textFile(attributesNamesFile).map(Utils.getParseLineFunction());
		// Group input data by partition rather than by lines: caching here -
		// careful with the caching strategies depending on dataset size
		// MEMORY ONLY - For the time being datasets are small so ...
		// JavaRDD<List<String>> data =
		// csvFile.glom().persist(StorageLevel.MEMORY_ONLY());
		JavaRDD<List<String>> data = csvFile.glom();
		Instances header = data.map(new CSVHeaderMapFunction(Arrays.asList(attributes.first()))).reduce(
				new CSVHeaderReduceFunction());
		JavaRDD<SimpleLogistic> classifier = data.map(new ClassifierMapFunction(header));
		classifier.saveAsTextFile(outputFile);
	}

}
