package com.integration.weka.spark.utils;

import java.io.IOException;
import java.io.StringReader;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.Function;

import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.DistributedWekaException;
import au.com.bytecode.opencsv.CSVReader;

public class Utils {
	private static Logger LOGGER = Logger.getLogger(Utils.class);

	/***********************************************
	 * Classes *
	 ***********************************************/

	/**
	 * Spark Function for Parsing a CSV line
	 */

	private static class InstanceFromLineBuilder implements Function<String, Instance> {
		private Instances strippedHeader;

		public InstanceFromLineBuilder(Instances fullDataHeader) throws DistributedWekaException {
			strippedHeader = CSVToARFFHeaderReduceTask.stripSummaryAtts(fullDataHeader);
			// Add the classifier class index
			strippedHeader.setClassIndex(fullDataHeader.classIndex());
		}

		public Instance call(String v1) throws Exception {
			CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
			rowParser.initParserOnly(CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(strippedHeader));
			// Parse the row of data
			String[] parsedRow = rowParser.parseRowOnly(v1);
			// Make an instance
			Instance instance = rowParser.makeInstance(strippedHeader, true, parsedRow);
			return instance;
		}

	}

	/***********************************************
	 * Static Accessors 
	 * @throws DistributedWekaException *
	 ***********************************************/

	public static InstanceFromLineBuilder getInstanceFromLineBuilder(Instances fullDataHeader) throws DistributedWekaException {
		return new InstanceFromLineBuilder(fullDataHeader);
	}

	/***********************************************
	 * Various utility functions *
	 ***********************************************/

	/**
	 * Same as <code>getParseLineFunction</code> but out of Spark context. No
	 * parallelism
	 * 
	 * @param line
	 * @return parsed line
	 * @throws IOException 
	 */
	public static String[] parseCSVLine(String line) throws IOException {
		CSVReader reader = new CSVReader(new StringReader(line));
		return reader.readNext();
	}

	public static String getDateAsStringFormat(Date date, String format) {
		SimpleDateFormat df = new SimpleDateFormat(format);
		return df.format(date);
	}

	/**
	 * Get the index of biggest element in the array. Not very sophisticated,
	 * but useful when values array is small enough.
	 * 
	 * @param values
	 * @return
	 */
	public static int getBiggestElementIndex(double[] values) {
		double bgValue = Double.MIN_VALUE;
		int bgIndex = -1;
		for (int i = 0; i < values.length; i++) {
			if (values[i] > bgValue) {
				bgValue = values[i];
				bgIndex = i;
			}
		}
		return bgIndex;
	}

	public static Options parseOptions(String[] options) throws Exception {
		if (options.length % 2 != 0){
			throw new Exception("Each option should have an associated value!");
		}
		Options opts = new Options();
		int i = 0;
		while (i < options.length){
			opts.addOption(options[i], options[i + 1]);
			i += 2;
		}
		return opts;
	}

}
