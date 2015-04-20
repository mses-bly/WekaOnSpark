package com.integration.weka.spark.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
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
	private static class ParseLine implements Function<String, String[]> {
		public String[] call(String line) throws Exception {
			CSVReader reader = new CSVReader(new StringReader(line));
			return reader.readNext();
		}
	}

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
	 * Static Accessors *
	 ***********************************************/

	public static ParseLine getParseLineFunction() {
		return new ParseLine();
	}

	public static InstanceFromLineBuilder getInstanceFromLineBuilder(Instances fullDataHeader) {
		try {
		return new InstanceFromLineBuilder(fullDataHeader);
		} catch (Exception ex) {
			LOGGER.error("Could not obtain Instance Builder. Error: [" + ex + "]");
		}
		return null;
	}


	/***********************************************
	 * Various utility functions *
	 ***********************************************/

	/**
	 * Saves a model to disk
	 * 
	 * @param output
	 *            Output file name - full path
	 * @param classifier
	 *            Model to write to disk
	 * @throws Exception
	 */
	public static void writeModelToDisk(String output, Classifier classifier) throws Exception {
		SerializationHelper.write(output, classifier);
	}

	/**
	 * Loads a model from file
	 * 
	 * @param input
	 *            Input file that contains the model
	 * @return Classifier instance read from file
	 * @throws Exception
	 */
	public static Classifier readModelFromDisk(String input) throws Exception {
		return (Classifier) SerializationHelper.read(input);
	}

	/**
	 * Reads the first line of a file: used to read very short files without
	 * parallelism.
	 * 
	 * @param fileName
	 * @return line read
	 */
	public static String readLineFromFile(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			return reader.readLine();
		} catch (Exception e) {
			LOGGER.error("Could not read file [" + fileName + "]. Error: [" + e + "]");
		}
		return null;
	}

	/**
	 * Same as <code>getParseLineFunction</code> but out of Spark context. No
	 * parallelism
	 * 
	 * @param line
	 * @return parsed line
	 */
	public static String[] parseCSVLine(String line) {
		CSVReader reader = new CSVReader(new StringReader(line));
		try {
			return reader.readNext();
		} catch (IOException e) {
			LOGGER.error("Could not parse CSV line [" + line + "]. Error: [" + e + "]");
		}
		return null;
	}

	public static String doubleArrayToString(double[] arr) {
		String str = "[";
		if (arr != null && arr.length > 0) {
			str += String.valueOf(arr[0]);
			for (int i = 1; i < arr.length; i++) {
				str += "," + arr[i];
			}
		}
		str += "]";
		return str;
	}

	public static <T> JavaRDD<T>[] splitRDD(double[] weights, JavaRDD<T> data) {
		return data.randomSplit(weights);
	}

}
