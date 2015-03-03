package com.integration.weka.spark.headers;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.Function2;

import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderReduceTask;

/**
 * Header building reduce function.
 * 
 * @author Moises
 *
 */
public class CSVHeaderReduceFunction implements Function2<Instances, Instances, Instances> {

	private CSVToARFFHeaderReduceTask csvToARFFHeaderReduceTask;

	public CSVHeaderReduceFunction() {
		csvToARFFHeaderReduceTask = new CSVToARFFHeaderReduceTask();
	}

	public Instances call(Instances arg0, Instances arg1) throws Exception {
		List<Instances> lst = new ArrayList<Instances>();
		lst.add(arg0);
		lst.add(arg1);
		Instances reducedInstance = csvToARFFHeaderReduceTask.aggregate(lst);
		reducedInstance.setClassIndex(arg0.classIndex());
		return reducedInstance;
	}

}
