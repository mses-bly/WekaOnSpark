package com.integration.weka.spark.jobs;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Date;
import java.util.Iterator;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;

import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.DistributedWekaException;

import com.integration.weka.spark.data.Dataset;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.headers.InstanceBuilderMapFunction;
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
	
	private JavaSparkContext context;
	private String inputFile;
	
	
	public CSVHeaderSparkJob(JavaSparkContext context, Options opts) throws Exception {
		this.context = context;
		if (!opts.hasOption(Constants.OPTION_INPUT_FILE)) {
			throw new Exception("Must provide an input file for HEADER job");
		}
		this.inputFile = opts.getOption(Constants.OPTION_INPUT_FILE);
	}
	
	public CSVHeaderSparkJob(JavaSparkContext context, String inputFile) throws Exception {
		this.context = context;
		this.inputFile = inputFile;
	}

	public void computeHeaderAndWriteToFile() throws IOException{
		Instances header = createHeader();
		String outputFilePath = "header_" + Utils.getDateAsStringFormat(new Date(), "YYYY-MM-dd_kk:mm:ss") + ".header";
		PrintWriter writer = new PrintWriter(outputFilePath, "UTF-8");
		writer.println(header);
		LOGGER.info("Wrote file [" + outputFilePath + "]");
		writer.close();
		
	}
	
	public  Dataset createDataSet(boolean shuffle) throws DistributedWekaException, IOException{
		Instances headerWithSummary = createHeader();
		JavaRDD<Instance> data = createInstancesRDD(headerWithSummary);
		Instances headerNoSummary = new CSVToARFFHeaderReduceTask().stripSummaryAtts(headerWithSummary);
		return new Dataset(data, headerWithSummary, headerNoSummary);
	}
	
	public Instances createHeader() throws IOException{
		JavaRDD<String> csvFile = context.textFile(this.inputFile);
		
		JavaRDD<Instances> instances = csvFile.mapPartitions(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length));
		Instances header = instances.reduce(new CSVHeaderReduceFunction());
		
		return header;
	}
	
	public JavaRDD<Instance> createInstancesRDD(Instances headerWithSummary) throws DistributedWekaException{
		JavaRDD<String> csvFile = context.textFile(this.inputFile);
		JavaRDD<Instance> instanceRDD = csvFile.mapPartitions(new InstanceBuilderMapFunction(headerWithSummary));
		return instanceRDD;
	}
}