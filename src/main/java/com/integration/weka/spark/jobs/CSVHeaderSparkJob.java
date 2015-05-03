package com.integration.weka.spark.jobs;

import java.io.PrintWriter;
import java.util.Date;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import weka.core.Instances;

import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Constants;
import com.integration.weka.spark.utils.Options;
import com.integration.weka.spark.utils.Utils;

/**
 * Create Weka ARFF Header Job
 * 
 * @author Moises
 *
 */
public class CSVHeaderSparkJob {
	private static Logger LOGGER = Logger.getLogger(CSVHeaderSparkJob.class);
	
	public static void createARFFHeader(SparkConf conf, JavaSparkContext context, Options opts) throws Exception {
		if (!opts.hasOption(Constants.OPTION_INPUT_FILE)) {
			throw new Exception("Must provide an input file for HEADER job");
		}
		JavaRDD<String> csvFile = context.textFile(opts.getOption(Constants.OPTION_INPUT_FILE));
		Instances header = csvFile.glom().map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());
		String outputFilePath = "header_" + Utils.getDateAsStringFormat(new Date(), "YYYY-MM-dd_kk:mm:ss") + ".header";
		PrintWriter writer = new PrintWriter(outputFilePath, "UTF-8");
		writer.println(header);
		LOGGER.info("Wrote file [" + outputFilePath + "]");
		writer.close();
	}
}