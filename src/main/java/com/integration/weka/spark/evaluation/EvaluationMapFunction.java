package com.integration.weka.spark.evaluation;

import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.Function;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.stats.ArffSummaryNumericMetric;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.WekaClassifierEvaluationMapTask;

/**
 * Evaluates a classifier on a number of examples instances.
 * 
 * @author Moises
 *
 */
public class EvaluationMapFunction implements Function<List<String>, Evaluation> {

	private static Logger LOGGER = Logger.getLogger(EvaluationMapFunction.class);

	private WekaClassifierEvaluationMapTask evaluationMapTask;
	private Instances strippedHeader;
	private Attribute classSummaryAttribute;
	private Attribute classAttribute;

	public EvaluationMapFunction(Classifier classifier, Instances dataHeader) {
		try {
			evaluationMapTask = new WekaClassifierEvaluationMapTask();

			strippedHeader = CSVToARFFHeaderReduceTask.stripSummaryAtts(dataHeader);
			evaluationMapTask.setTotalNumFolds(1);

			evaluationMapTask.setClassifier(classifier);

			classSummaryAttribute = dataHeader.attribute(CSVToARFFHeaderMapTask.ARFF_SUMMARY_ATTRIBUTE_PREFIX + dataHeader.classAttribute().name());
			classAttribute = dataHeader.classAttribute();

			evaluationMapTask.setup(strippedHeader, priors(), count(), System.currentTimeMillis(), 0);
		} catch (Exception ex) {
			LOGGER.error("Could evaluate classifier. Error: [" + ex + "]");
		}
	}

	/**
	 * Count the sum of values in the class attribute. If the class attribute is
	 * nominal, count the sum for each possible nominal value. If the class
	 * attribute is numeric, sum up all values for the class.
	 * 
	 */
	private double[] priors() {
		double[] priors = null;
		if (classAttribute.isNominal()) {
			int numValues = classAttribute.numValues();
			priors = new double[numValues];
			for (int i = 0; i < numValues; i++) {
				String valueName = classAttribute.value(i);
				// Extract the summary data for this nominal value
				// @attribute arff_summary_CLASS
				// {tested_negative_492.0,tested_positive_263.0}
				double prior = Double.valueOf(classSummaryAttribute.value(i).replace(valueName + "_", "").trim());
				priors[i] = prior;
			}
		} else {
			// @attribute arff_summary_CLASS
			// {count_200.0,sum_6358.7916108556,sumSq_257320.9770071836,min_-3.445671441,max_82.438286712,missing_0.0,mean_31.793958054278,stdDev_16.647365864600875}
			priors = new double[1];
			priors[0] = ArffSummaryNumericMetric.SUM.valueFromAttribute(classSummaryAttribute);
		}
		return priors;

	}

	/**
	 * Count the number of values in the class attribute. If the class attribute
	 * is nominal, count each possible nominal value. If the class attribute is
	 * numeric, get the count of instances (the number of records)
	 * 
	 */
	private double count() {
		if (classAttribute.isNominal()) {
			return classAttribute.numValues();
		}
		return ArffSummaryNumericMetric.COUNT.valueFromAttribute(classSummaryAttribute);
	}

	public Evaluation call(List<String> v1) {
		try {
			// This is done here because of a non serializable class that Weka distributed package uses
			// - field (class "weka.distributed.CSVToARFFHeaderMapTask", name:
			// "m_parser", type: "class au.com.bytecode.opencsv.CSVParser")
			CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
			rowParser.initParserOnly(CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(strippedHeader));
			for (String str : v1) {
				String[] parsedRow = rowParser.parseRowOnly(str);
				Instance currentInstance = rowParser.makeInstance(strippedHeader, true, parsedRow);
				evaluationMapTask.processInstance(currentInstance);
			}
			evaluationMapTask.finalizeTask();
			return evaluationMapTask.getEvaluation();
		} catch (Exception ex) {
			LOGGER.error("Could evaluate instance. Error: [" + ex + "]");
		}
		return null;
	}
}
