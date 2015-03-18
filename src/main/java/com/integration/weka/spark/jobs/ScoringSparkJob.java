package com.integration.weka.spark.jobs;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.core.Instance;
import weka.core.Instances;

import com.integration.weka.spark.data.InstanceBuilderFunction;
import com.integration.weka.spark.headers.HeaderUtils;
import com.integration.weka.spark.scoring.ScoringMapFunction;
import com.integration.weka.spark.utils.Utils;

public class ScoringSparkJob {
	private static Logger LOGGER = Logger.getLogger(ScoringSparkJob.class);

	public static void scoreDataSet(SparkConf conf, JavaSparkContext context, String modelPath, String dataFile, String attrFile, String outputFile) {
		LOGGER.info("Scoring dataset [" + dataFile + "]");

		// Load the data file to be predicted
		JavaRDD<String> csvFile = context.textFile(dataFile);
		// TODO Optimize here to avoid recomputing the RDD for multiple passes
		JavaRDD<List<String>> data = csvFile.glom();
		// No need for paralelism for the attributes
		String[] attributesWithClass = Utils.parseCSVLine(Utils.readLineFromFile(attrFile));
		String[] attributesNoClass = Utils.removeClassAttribute(attributesWithClass);
		// Build Predicted Data Header
		Instances dataHeader = HeaderUtils.getHeaderFromAttributes(Arrays.asList(attributesNoClass));
		// Build data Instances RDD
		JavaRDD<List<Instance>> instanceRDD = csvFile.map(new InstanceBuilderFunction(dataHeader)).glom();

		// Model Header. Contains class attribute.
		Instances modelHeader = HeaderUtils.getHeaderFromAttributes(Arrays.asList(attributesWithClass));

		


		// Read model from disc and make prediction
		Object model;
		try {
			model = Utils.readModelFromDisk(modelPath);
		} catch (Exception e) {
			LOGGER.error("Could not read model from file. Error: [" + e + "]");
			return;
		}
		JavaRDD<List<String>> predictions = instanceRDD.map(new ScoringMapFunction(model, modelHeader, dataHeader));
		// TODO change this to print the files properly
		predictions.saveAsTextFile(outputFile);

	}
}
