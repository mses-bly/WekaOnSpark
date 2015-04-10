package com.integration.weka.spark.jobs;

import java.util.ArrayList;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.integration.weka.spark.data.KFoldsBuilder;

public class CrossValidationSparkJob {

	private static Logger LOGGER = Logger.getLogger(CrossValidationSparkJob.class);

	public static void performCrossValidation(SparkConf conf, JavaSparkContext context, String dataFile, int numFolds, String outputFolder) {
		// Load the data file to be predicted
		JavaRDD<String> csvFile = context.textFile(dataFile);

		// File needs to contain line numbers in the format
		// "ln v1,v2,v3,v4,...,vn"
		// Took care of it in the launch script
		PairFunction<String, Long, String> keyByLineNumber = new PairFunction<String, Long, String>() {
			public Tuple2<Long, String> call(String t) throws Exception {
				return new Tuple2<Long, String>(Long.valueOf(t.split(" ")[0]), t.split(" ")[1]);
			}
		};

		JavaPairRDD<Long, String> keyPairValues = csvFile.mapToPair(keyByLineNumber);

		long numRecords = csvFile.count();

		if (numRecords < numFolds) {
			LOGGER.error("Number of folds must be less or equal to the number of records");
			return;
		}
		
		KFoldsBuilder foldsBuilder = new KFoldsBuilder(numRecords, numFolds);

		ArrayList<JavaPairRDD<Long, String>> rddFolds = foldsBuilder.buildFolds(keyPairValues);
		
		if (rddFolds == null || rddFolds.size() <= 0) {
			LOGGER.error("Could not implement CrossValidation");
			return;
		}
		// TODO Continue implementing cross validation

		// For testing purposes right now. See that folds are properly
		// calculated
		int i = 1;
		for (JavaPairRDD<Long, String> fold : rddFolds){
			System.out.println("*************************FOLD "+ i + "************************");
			for (String s : fold.collectAsMap().values()){
				System.out.println(s);
			}
			i++;
		}
	}

}
