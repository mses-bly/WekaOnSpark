package com.integration.weka.spark.jobs;

import java.io.PrintWriter;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.core.Instances;

import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Utils;

/**
 * Wrapper for launching a header build job.
 * 
 * @author Moises
 *
 */
public class CSVHeaderSparkJob {
	private static Logger LOGGER = Logger.getLogger(CSVHeaderSparkJob.class);
	/**
	 * 
	 * @param conf
	 *            : Spark configuration
	 * @param context
	 *            : Spark context
	 * @param inputFile
	 *            : Path to input data
	 * @param outputFile
	 *            : file for reduced results
	 */
	public static void loadCVSFile(SparkConf conf, JavaSparkContext context, String inputFilePath, String outputFilePath) {
		JavaRDD<String> csvFile = context.textFile(inputFilePath);
		Instances header = csvFile.glom().map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(outputFilePath, "UTF-8");
			writer.println(header);
		} catch (Exception ex) {
			LOGGER.error("Could not write header to file " + outputFilePath + ". Error: [" + ex + "]");
		} finally {
			if (writer != null) {
				LOGGER.info("Wrote file [" + outputFilePath + "]");
				writer.close();
			}
		}
	}
}