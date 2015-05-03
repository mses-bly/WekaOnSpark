package com.integration.weka.spark.jobs;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;

import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.score.ScoreMapFunction;
import com.integration.weka.spark.utils.Constants;
import com.integration.weka.spark.utils.Options;
import com.integration.weka.spark.utils.Utils;

public class ScoreSparkJob {
	private static Logger LOGGER = Logger.getLogger(ScoreSparkJob.class);

	public static void scoreDataSet(SparkConf conf, JavaSparkContext context, Options opts) throws Exception {
		if (!opts.hasOption(Constants.OPTION_INPUT_FILE)) {
			throw new Exception("Must provide training data file for SCORE job");
		}
		if (!opts.hasOption(Constants.OPTION_SCORE_FILE)) {
			throw new Exception("Must provide file to score for SCORE job");
		}
		if (!opts.hasOption(Constants.OPTION_CLASSIFIER_MODEL_FILE)) {
			throw new Exception("Must provide classfier model for SCORE job");
		}
		String scoreFilePath = opts.getOption(Constants.OPTION_SCORE_FILE);
		String trainingDataFilePath = opts.getOption(Constants.OPTION_INPUT_FILE);
		String modelFilePath = opts.getOption(Constants.OPTION_CLASSIFIER_MODEL_FILE);

		// Load the data file to be predicted
		JavaRDD<String> csvInputFile = context.textFile(scoreFilePath);

		int numAttr = Utils.parseCSVLine(csvInputFile.first()).length;
		List<String> attrNames = new ArrayList<String>();
		for (int i = 0; i < numAttr; i++) {
			attrNames.add("A" + i);
		}

		CSVToARFFHeaderMapTask csvToARFFHeaderMapTask = new CSVToARFFHeaderMapTask();
		csvToARFFHeaderMapTask.initParserOnly(attrNames);
		Instances headerFromInput = csvToARFFHeaderMapTask.getHeader();

		JavaRDD<String> csvTrainFile = context.textFile(trainingDataFilePath);
		Instances headerFromModel = csvTrainFile.glom().map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvTrainFile.first()).length)).reduce(new CSVHeaderReduceFunction());
		headerFromModel = CSVToARFFHeaderReduceTask.stripSummaryAtts(headerFromModel);

		// Build data Instances RDD
		JavaRDD<List<Instance>> instanceRDD = csvInputFile.map(Utils.getInstanceFromLineBuilder(headerFromInput)).glom();

		// Read model from disc and make prediction
		Object model = (Classifier) SerializationHelper.read(modelFilePath);

		List<List<String>> predictions = instanceRDD.map(new ScoreMapFunction(model, headerFromModel, headerFromInput)).collect();

		String modelName = modelFilePath.split("/")[modelFilePath.split("/").length - 1];

		String outputFilePath = "score_" + modelName + ".txt";

		PrintWriter writer = new PrintWriter(outputFilePath, "UTF-8");
		for (List<String> preds : predictions) {
			for (String prediction : preds) {
				writer.println(prediction);
			}
		}

		writer.close();
		LOGGER.info("Score for " + modelName + " saved at [" + outputFilePath + "]");
	}
}
