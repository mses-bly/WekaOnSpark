package com.integration.weka.spark.headers;

import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.Function;

import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;

/**
 * Header building Map Function.
 * 
 * @author Moises
 *
 */
public class CSVHeaderMapFunction implements Function<List<String>, Instances> {

	private static Logger LOGGER = Logger.getLogger(CSVHeaderMapFunction.class);
	private CSVToARFFHeaderMapTask csvToARFFHeaderMapTask;
	private List<String> attributes;

	/**
	 * Instantiate map function for building header
	 * 
	 * @param attributes
	 *            Attributes for the header
	 */
	public CSVHeaderMapFunction(List<String> attributes) {
		csvToARFFHeaderMapTask = new CSVToARFFHeaderMapTask();
		this.attributes = attributes;
	}

	public Instances call(List<String> arg0) {
		try {
			for (String str : arg0) {
				csvToARFFHeaderMapTask.processRow(str, attributes);
			}
			Instances headerInstance = csvToARFFHeaderMapTask.getHeader();
			headerInstance.setClassIndex(attributes.size() - 1);
			return headerInstance;
		} catch (Exception e) {
			LOGGER.error("Could not build header for this training set. Error: [" + e + "]");
		}
		return null;
	}

}
