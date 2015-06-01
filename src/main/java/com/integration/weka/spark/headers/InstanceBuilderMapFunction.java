package com.integration.weka.spark.headers;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.function.FlatMapFunction;

import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.DistributedWekaException;

public class InstanceBuilderMapFunction implements FlatMapFunction<Iterator<String>, Instance>{
	private List<Instance> instanceList;
	private Instances headerNoSummary;
	
	public InstanceBuilderMapFunction(Instances headerWithSummary) throws DistributedWekaException {
		instanceList = new ArrayList<>();
		headerNoSummary = CSVToARFFHeaderReduceTask.stripSummaryAtts(headerWithSummary);
		// Add the classifier class index
		headerNoSummary.setClassIndex(headerWithSummary.classIndex());
	}
	
	@Override
	public Iterable<Instance> call(Iterator<String> arg0) throws Exception {
		CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
		rowParser.initParserOnly(CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(headerNoSummary));
		while (arg0.hasNext()){
			// Parse the row of data
			String[] parsedRow = rowParser.parseRowOnly(arg0.next());
			// Make an instance
			Instance instance = rowParser.makeInstance(headerNoSummary, true, parsedRow);
			instanceList.add(instance);
		}
		return instanceList;
	}

}
