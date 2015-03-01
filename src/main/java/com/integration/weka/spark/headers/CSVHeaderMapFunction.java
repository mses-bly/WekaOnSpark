package com.integration.weka.spark.headers;


import java.util.List;

import org.apache.spark.api.java.function.Function;

import weka.distributed.CSVToARFFHeaderMapTask;
import weka.core.Instances;

public class CSVHeaderMapFunction implements Function<List<String>, Instances>{
	
	private CSVToARFFHeaderMapTask csvToARFFHeaderMapTask;
	private List<String> attributes;

	/**
	 * Instantiate Weka Distributed Map Task
	 */
	public CSVHeaderMapFunction(List<String> attributes) {
		csvToARFFHeaderMapTask = new CSVToARFFHeaderMapTask();
		this.attributes = attributes;
	}
	
	/**
	 * Process the RDD lines to produce a Header - Instances.
	 */
	public Instances call(List<String> arg0) throws Exception {
		for (String str : arg0){
			csvToARFFHeaderMapTask.processRow(str, attributes);
		}
		Instances headerInstance = csvToARFFHeaderMapTask.getHeader();
		headerInstance.setClassIndex(attributes.size()-1);
		return headerInstance;
			
	}

}
