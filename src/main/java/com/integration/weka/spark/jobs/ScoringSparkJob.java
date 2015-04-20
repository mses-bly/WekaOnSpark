package com.integration.weka.spark.jobs;

import java.io.PrintWriter;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderReduceTask;
import cern.colt.Arrays;

import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.scoring.ScoringMapFunction;
import com.integration.weka.spark.utils.Utils;

public class ScoringSparkJob {
	private static Logger LOGGER = Logger.getLogger(ScoringSparkJob.class);

	public static void scoreDataSet(SparkConf conf, JavaSparkContext context, String modelPath, String inputFile, String outputFile) {
		PrintWriter writer = null;
		try {
			LOGGER.info("Scoring dataset [" + inputFile + "]");

			// Load the data file to be predicted
			JavaRDD<String> csvFile = context.textFile(inputFile);

			// This header contains one less attribute than the model ... in
			// theory
			// no class is set
			Instances headerFromInput = csvFile.glom().map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());
			headerFromInput = CSVToARFFHeaderReduceTask.stripSummaryAtts(headerFromInput);

			// This header contains one extra attribute.
			Instances headerFromModel = csvFile.glom().map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length + 1)).reduce(new CSVHeaderReduceFunction());
			headerFromModel = CSVToARFFHeaderReduceTask.stripSummaryAtts(headerFromInput);

			// Build data Instances RDD
			JavaRDD<List<Instance>> instanceRDD = csvFile.map(Utils.getInstanceFromLineBuilder(headerFromInput)).glom();

			// Read model from disc and make prediction
			Object model = Utils.readModelFromDisk(modelPath);

			List<List<double[]>> predictions = instanceRDD.map(new ScoringMapFunction(model, headerFromModel, headerFromInput)).collect();
			writer = new PrintWriter(outputFile, "UTF-8");
			for (List<double[]> predictionList : predictions) {
				for (double[] prediction : predictionList) {
					writer.println(Arrays.toString(prediction));
				}
			}
		} catch (Exception ex) {
			LOGGER.error("Could not complete scoring job. Error: [" + ex + "]");
		} finally {
			if (writer != null) {
				writer.close();
			}
		}
	}
}
