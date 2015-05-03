package com.integration.weka.spark.evaluation;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.Function;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.AggregateableEvaluation;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.stats.ArffSummaryNumericMetric;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.WekaClassifierEvaluationMapTask;

public class EvaluationMapFunction implements Function<List<String>, Evaluation> {

	private ArrayList<WekaClassifierEvaluationMapTask> evaluationMapTasks;
	private Instances strippedHeader;
	private Attribute classSummaryAttribute;
	private Attribute classAttribute;
	private int kFolds;

	public EvaluationMapFunction(Instances dataHeader, List<Classifier> classifiers, int kFolds) throws Exception {
		strippedHeader = CSVToARFFHeaderReduceTask.stripSummaryAtts(dataHeader);
		classSummaryAttribute = dataHeader.attribute(CSVToARFFHeaderMapTask.ARFF_SUMMARY_ATTRIBUTE_PREFIX + dataHeader.classAttribute().name());
		classAttribute = dataHeader.classAttribute();
		long seed = 1L;
		this.kFolds = kFolds;

		evaluationMapTasks = new ArrayList<WekaClassifierEvaluationMapTask>();
		for (int i = 0; i < kFolds; i++) {
			WekaClassifierEvaluationMapTask evaluationTask = new WekaClassifierEvaluationMapTask();
			evaluationTask.setTotalNumFolds(kFolds);
			evaluationTask.setFoldNumber(i + 1);
			evaluationTask.setClassifier(classifiers.get(i));
			evaluationTask.setup(strippedHeader, priors(), count(), seed, 0);
			evaluationMapTasks.add(evaluationTask);
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
	 * numeric, get the count of instances (the number
	 *  of records)
	 * 
	 */
	private double count() {
		if (classAttribute.isNominal()) {
			return classAttribute.numValues();
		}
		return ArffSummaryNumericMetric.COUNT.valueFromAttribute(classSummaryAttribute);
	}

	public Evaluation call(List<String> v1) throws Exception {
		CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
		rowParser.initParserOnly(CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(strippedHeader));
		for (int i = 0; i < v1.size(); i++) {
			for (int j = 0; j < kFolds; j++) {
				String[] parsedRow = rowParser.parseRowOnly(v1.get(i));
				Instance currentInstance = rowParser.makeInstance(strippedHeader, true, parsedRow);
				evaluationMapTasks.get(j).processInstance(currentInstance);
			}
		}

		AggregateableEvaluation aggregateableEvaluation = null;
		for (int i = 0; i < kFolds; i++) {
			evaluationMapTasks.get(i).finalizeTask();
			Evaluation iFoldEvaluation = evaluationMapTasks.get(i).getEvaluation();
			if (aggregateableEvaluation == null) {
				aggregateableEvaluation = new AggregateableEvaluation(iFoldEvaluation);
			}
			aggregateableEvaluation.aggregate(iFoldEvaluation);
		}
		return aggregateableEvaluation;
	}

}
