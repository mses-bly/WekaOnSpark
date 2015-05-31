package com.integration.weka.spark.headers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.function.FlatMapFunction;
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
public class CSVHeaderMapFunction implements FlatMapFunction<Iterator<String>, Instances> {
	private List<Instances> instances;
	private List<String> attributes;
	/**
	 * Instantiate map function for building header
	 * 
	 * @param attributes
	 *            Number of attributes for the header
	 */
	public CSVHeaderMapFunction(int numAttributes) {
		instances = new ArrayList<>();
		attributes = new ArrayList<String>();
		for (int i = 0; i < numAttributes - 1; i++) {
			attributes.add("A" + i);
		}
		attributes.add("CLASS");
	}

	@Override
	public Iterable<Instances> call(Iterator<String> arg0) throws Exception {
		CSVToARFFHeaderMapTask csvToARFFHeaderMapTask = new CSVToARFFHeaderMapTask();
		while (arg0.hasNext()){
			csvToARFFHeaderMapTask.processRow(arg0.next(), attributes);
		}
		Instances innerHeader = csvToARFFHeaderMapTask.getHeader();
		innerHeader.setClassIndex(attributes.size() - 1);
		innerHeader.setRelationName("RELATION");
		instances.add(innerHeader);
		return instances;
	}
}
