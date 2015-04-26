package com.integration.weka.spark.jobs;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import cern.colt.Arrays;

import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.score.ScoreMapFunction;
import com.integration.weka.spark.utils.Utils;

public class ScoreSparkJob {
	private static Logger LOGGER = Logger.getLogger(ScoreSparkJob.class);

	public static void scoreDataSet(SparkConf conf, JavaSparkContext context, String modelFilePath, String inputFilePath, String trainFilePath, String outputFilePath) {
		PrintWriter writer = null;
		try {
			// Load the data file to be predicted
			JavaRDD<String> csvInputFile = context.textFile(inputFilePath);

			int numAttr = Utils.parseCSVLine(csvInputFile.first()).length;
			List<String> attrNames = new ArrayList<String>();
			for (int i = 0; i < numAttr ; i++){
				attrNames.add("A" + i);
			}
			
			CSVToARFFHeaderMapTask csvToARFFHeaderMapTask = new CSVToARFFHeaderMapTask();
			csvToARFFHeaderMapTask.initParserOnly(attrNames);
			Instances headerFromInput = csvToARFFHeaderMapTask.getHeader();

			JavaRDD<String> csvTrainFile = context.textFile(trainFilePath);
			Instances headerFromModel = csvTrainFile.glom().map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvTrainFile.first()).length)).reduce(new CSVHeaderReduceFunction());
			headerFromModel = CSVToARFFHeaderReduceTask.stripSummaryAtts(headerFromModel);

			// Build data Instances RDD
			JavaRDD<List<Instance>> instanceRDD = csvInputFile.map(Utils.getInstanceFromLineBuilder(headerFromInput)).glom();

			// Read model from disc and make prediction
			Object model = (Classifier) SerializationHelper.read(modelFilePath);

			List<List<double[]>> predictions = instanceRDD.map(new ScoreMapFunction(model, headerFromModel, headerFromInput)).collect();
			writer = new PrintWriter(outputFilePath, "UTF-8");

			Attribute classAttribute = headerFromModel.classAttribute();
			for (List<double[]> predictionList : predictions) {
				for (int i = 0; i < predictionList.size(); i++) {
					double[] prediction = predictionList.get(i);
					if (!classAttribute.isNominal()) {
						writer.println(Arrays.toString(prediction));
					} else {
						writer.println(Arrays.toString(prediction) + " " + classAttribute.value(Utils.getBiggestElementIndex(prediction)));
					}
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
