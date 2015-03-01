package com.integration.weka.spark.utils;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import weka.distributed.DistributedWekaException;

import com.integration.weka.spark.jobs.ClassifierSparkTask;

public class Launcher {
	public static void main(String[] args) {
		// SparkConf conf = new
		// SparkConf().setAppName("SparkWekaIntegration").setMaster("local[1]");
		SparkConf conf = new SparkConf().setAppName("SparkWekaIntegration");
		JavaSparkContext context = new JavaSparkContext(conf);

		String inputFile = "diabetes.csv";
		String attributesFile = "diabetes_attr.csv";
		String outputFile = "./target/output";

		// CSVHeaderSparkTask.loadCVSFile(conf, context, inputFile,
		// attributesFile, outputFile);
		//
		try {
			ClassifierSparkTask.buildClassifier(conf, context, inputFile, attributesFile, outputFile);
		} catch (DistributedWekaException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
