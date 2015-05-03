package com.integration.weka.spark.headers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.Function;

import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.DistributedWekaException;

/**
 * Header builder Map Function.
 * 
 * @author Moises
 *
 */
public class CSVHeaderMapFunction implements Function<List<String>, Instances> {

	private CSVToARFFHeaderMapTask csvToARFFHeaderMapTask;
	private List<String> attributes;

	/**
	 * Instantiate map function for building header
	 * 
	 * @param attributes
	 *            Number of attributes for the header
	 */
	public CSVHeaderMapFunction(int numAttributes) {
		csvToARFFHeaderMapTask = new CSVToARFFHeaderMapTask();
		attributes = new ArrayList<String>();
		for (int i = 0; i < numAttributes - 1; i++) {
			attributes.add("A" + i);
		}
		attributes.add("CLASS");
	}

	public Instances call(List<String> arg0) throws DistributedWekaException, IOException {
		for (String str : arg0) {
			csvToARFFHeaderMapTask.processRow(str, attributes);
		}
		Instances headerInstance = csvToARFFHeaderMapTask.getHeader();
		headerInstance.setClassIndex(attributes.size() - 1);
		headerInstance.setRelationName("RELATION");
		return headerInstance;
	}

}
