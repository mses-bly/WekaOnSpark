package com.integration.weka.spark.utils;

import java.io.StringReader;

import org.apache.spark.api.java.function.Function;

import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
import au.com.bytecode.opencsv.CSVReader;

public class Utils {

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

	/***********************************************
	 * Static Accessors *
	 ***********************************************/

	public static ParseLine getParseLineFunction() {
		return new ParseLine();
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
}
