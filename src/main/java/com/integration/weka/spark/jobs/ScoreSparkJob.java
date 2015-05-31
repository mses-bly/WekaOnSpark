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

import com.integration.weka.spark.data.Dataset;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.score.ScoreMapFunction;
import com.integration.weka.spark.utils.Constants;
import com.integration.weka.spark.utils.Options;
import com.integration.weka.spark.utils.Utils;

public class ScoreSparkJob {
	private static Logger LOGGER = Logger.getLogger(ScoreSparkJob.class);

	public static void scoreDataSet(JavaSparkContext context, Options opts) throws Exception {
		if (!opts.hasOption(Constants.OPTION_INPUT_FILE)) {
			throw new Exception("Must provide training data file for SCORE job");
		}
		if (!opts.hasOption(Constants.OPTION_SCORE_FILE)) {
			throw new Exception("Must provide file to score for SCORE job");
		}
		if (!opts.hasOption(Constants.OPTION_CLASSIFIER_MODEL_FILE)) {
			throw new Exception("Must provide classfier model for SCORE job");
		}
		
		JavaRDD<String> scoreRDD = context.textFile(opts.getOption(Constants.OPTION_SCORE_FILE));
		int numAttr = Utils.parseCSVLine(scoreRDD.first()).length;
		List<String> attrNames = new ArrayList<String>();
		for (int i = 0; i < numAttr; i++) {
			attrNames.add("A" + i);
		}
		CSVToARFFHeaderMapTask csvToARFFHeaderMapTask = new CSVToARFFHeaderMapTask();
		csvToARFFHeaderMapTask.initParserOnly(attrNames);
		Instances scoreHeader = csvToARFFHeaderMapTask.getHeader();
		JavaRDD<Instance> scoreData = new CSVHeaderSparkJob(context, opts.getOption(Constants.OPTION_SCORE_FILE)).createInstancesRDD(scoreHeader);
		
		Dataset trainData = new CSVHeaderSparkJob(context, opts).createDataSet(false);
		
		// Read model from disc and make prediction
		Object model = (Classifier) SerializationHelper.read(opts.getOption(Constants.OPTION_CLASSIFIER_MODEL_FILE));

		List<String> predictions = scoreData.map(new ScoreMapFunction(model, trainData.getHeaderWithSummary(), scoreHeader)).collect();

		String modelFilePath = opts.getOption(Constants.OPTION_CLASSIFIER_MODEL_FILE);
		String modelName = modelFilePath.split("/")[modelFilePath.split("/").length - 1];

		String outputFilePath = "score_" + modelName + ".txt";

		PrintWriter writer = new PrintWriter(outputFilePath, "UTF-8");
		for (String prediction : predictions) {
			writer.println(prediction);
		}

		writer.close();
		LOGGER.info("Score for " + modelName + " saved at [" + outputFilePath + "]");
	}
}
