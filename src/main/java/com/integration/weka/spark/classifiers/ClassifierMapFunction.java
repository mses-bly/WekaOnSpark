package com.integration.weka.spark.classifiers;

import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.Function;

import weka.classifiers.functions.SimpleLogistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.DistributedWekaException;
import weka.distributed.WekaClassifierMapTask;

public class ClassifierMapFunction implements Function<List<String>, SimpleLogistic> {

	private static Logger LOGGER = Logger.getLogger(ClassifierMapFunction.class);
	private WekaClassifierMapTask classifierMapTask;
	private Instances strippedHeader;
	private List<String> attibutesNames;

	/**
	 * 
	 * @param trainingHeader
	 *            previously generated header.
	 * @throws DistributedWekaException
	 * 
	 */
	public ClassifierMapFunction(Instances trainingHeader) throws DistributedWekaException {
		// Strip summary from header
		strippedHeader = CSVToARFFHeaderReduceTask.stripSummaryAtts(trainingHeader);

		// Add the classifier class index
		strippedHeader.setClassIndex(trainingHeader.classIndex());

		// Extract dataset attributes names
		attibutesNames = CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(strippedHeader);

		// Setup the type of classifier
		classifierMapTask = new WekaClassifierMapTask();
		classifierMapTask.setClassifier(new SimpleLogistic());

		// Set our classifier with the stripped header
		classifierMapTask.setup(strippedHeader);

	}

	public SimpleLogistic call(List<String> arg0) throws Exception {
		// This is done here because of a non serializable class that weka
		// distributed package uses
		// - field (class "weka.distributed.CSVToARFFHeaderMapTask", name:
		// "m_parser", type: "class au.com.bytecode.opencsv.CSVParser")
		CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
		rowParser.initParserOnly(attibutesNames);

		for (String str : arg0) {
			// Parse the row of data
			String[] parsedRow = rowParser.parseRowOnly(str);
			// Make an instance row
			Instance currentInstance = rowParser.makeInstance(strippedHeader, true, parsedRow);
			// Add this row of data to the classifier
			classifierMapTask.processInstance(currentInstance);
		}
		classifierMapTask.finalizeTask();
		return (SimpleLogistic) classifierMapTask.getClassifier();
	}

}
