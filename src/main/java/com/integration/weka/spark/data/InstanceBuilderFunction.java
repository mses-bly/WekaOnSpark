package com.integration.weka.spark.data;

import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.Function;

import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.DistributedWekaException;

/**
 * Build an instance from every data line.
 * 
 * @author Moises
 *
 */

public class InstanceBuilderFunction implements Function<String, Instance> {

	private static Logger LOGGER = Logger.getLogger(InstanceBuilderFunction.class);

	private List<String> attributesNames;
	private Instances strippedHeader;

	public InstanceBuilderFunction(Instances dataHeader) {

		try {
			strippedHeader = CSVToARFFHeaderReduceTask.stripSummaryAtts(dataHeader);

			// Add the classifier class index
			strippedHeader.setClassIndex(dataHeader.classIndex());

			// Extract dataset attributes names
			attributesNames = CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(strippedHeader);
		} catch (DistributedWekaException e) {
			LOGGER.error("Could not build Instance RDD function. Error: [" + e + "]");
		}

	}

	public Instance call(String v1) throws Exception {

		CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
		rowParser.initParserOnly(attributesNames);

		// Parse the row of data
		String[] parsedRow = rowParser.parseRowOnly(v1);

		// Make an instance
		Instance instance = rowParser.makeInstance(strippedHeader, true, parsedRow);

		return instance;

	}

}
