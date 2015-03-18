package com.integration.weka.spark.classifiers;

import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.Function;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.DistributedWekaException;
import weka.distributed.WekaClassifierMapTask;

/**
 * Classification Map Function. Build and train classifier.
 * 
 * @author Moises
 *
 */
public class ClassifierMapFunction implements Function<List<String>, Classifier> {

	private static Logger LOGGER = Logger.getLogger(ClassifierMapFunction.class);

	private WekaClassifierMapTask classifierMapTask;
	private Instances strippedHeader;
	private List<String> attributesNames;

	/**
	 * Creates instance of ClassifierMapFunction for classification training
	 * 
	 * @param trainingHeader
	 *            previously calculated header
	 */
	public ClassifierMapFunction(Instances trainingHeader, String classifierClassName) {
		// Strip summary from header
		try {
			strippedHeader = CSVToARFFHeaderReduceTask.stripSummaryAtts(trainingHeader);

			// Add the classifier class index
			strippedHeader.setClassIndex(trainingHeader.classIndex());

			// Extract dataset attributes names
			attributesNames = CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(strippedHeader);

			// Setup the type of classifier
			classifierMapTask = new WekaClassifierMapTask();

			// Instantiate classifier
			Class<?> clazz = Class.forName(classifierClassName);
			Object object = clazz.newInstance();
			classifierMapTask.setClassifier((Classifier) object);

			// Set our classifier with the stripped header
			classifierMapTask.setup(strippedHeader);

		} catch (DistributedWekaException e) {
			LOGGER.error("Could not instantiate ClassifierMapFunction. Error: [" + e + "]");
		} catch (Exception e) {
			LOGGER.error("Could not instantiate classifier [" + classifierClassName + "]. Error: [" + e + "]");
		}

	}

	/**
	 * Call method for this map function
	 */
	public Classifier call(List<String> arg0) {
		// This is done here because of a non serializable class that Weka
		// distributed package uses
		// - field (class "weka.distributed.CSVToARFFHeaderMapTask", name:
		// "m_parser", type: "class au.com.bytecode.opencsv.CSVParser")
		CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
		rowParser.initParserOnly(attributesNames);

		try {
			for (String str : arg0) {
				// Parse the row of data
				String[] parsedRow = rowParser.parseRowOnly(str);

				// Make an instance row
				Instance currentInstance = rowParser.makeInstance(strippedHeader, true, parsedRow);

				// Add this row of data to the classifier
				classifierMapTask.processInstance(currentInstance);
			}

			classifierMapTask.finalizeTask();
			return classifierMapTask.getClassifier();

		} catch (Exception e) {
			LOGGER.error("Could not train classifier. Error: [" + e + "]");
		}
		return null;
	}

}
