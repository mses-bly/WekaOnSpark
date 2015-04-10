package com.integration.weka.spark.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
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

	/**
	 * Spark function for filtering all RDDs Pairs whose key falls in a given
	 * range. Inclusive range [rangeStart, rangeEnds]
	 */
	private static class RangeFilter implements Function<Tuple2<Long, String>, Boolean> {

		private long rangeStart;
		private long rangeEnd;

		public RangeFilter(long rangeStart, long rangeEnd) throws Exception {
			if (rangeStart > rangeEnd) {
				throw new Exception("Range start must be less or equal than end");
			}
			this.rangeStart = rangeStart;
			this.rangeEnd = rangeEnd;
		}

		public Boolean call(Tuple2<Long, String> v1) throws Exception {
			if (v1._1() >= rangeStart && v1._1() <= rangeEnd) {
				return true;
			}
			return false;
		}
	}

	/***********************************************
	 * Static Accessors *
	 ***********************************************/

	public static ParseLine getParseLineFunction() {
		return new ParseLine();
	}

	public static RangeFilter getRangeFilter(long start, long end) {
		try {
			RangeFilter rangeFilter = new RangeFilter(start, end);
			return rangeFilter;
		} catch (Exception e) {
			LOGGER.error(e);
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

	/**
	 * Removes the class attribute from an array of attributes. Assume class
	 * attribute is at the end.
	 * 
	 * @param attributes
	 * @return attributes without the class attribute at the end
	 */
	public static String[] removeClassAttribute(String[] attributes) {
		String[] noClass = new String[attributes.length - 1];
		System.arraycopy(attributes, 0, noClass, 0, attributes.length - 1);
		return noClass;
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
