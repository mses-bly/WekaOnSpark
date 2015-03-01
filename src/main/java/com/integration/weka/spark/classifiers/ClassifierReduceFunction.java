package com.integration.weka.spark.classifiers;

import java.util.ArrayList;

import org.apache.spark.api.java.function.Function2;

import weka.classifiers.Classifier;
import weka.distributed.WekaClassifierReduceTask;

/**
 * Classification Reduce Function.
 * 
 * @author Moises
 *
 */
public class ClassifierReduceFunction implements Function2<Classifier, Classifier, Classifier> {

	private WekaClassifierReduceTask classifierReduceTask;

	public ClassifierReduceFunction() {
		classifierReduceTask = new WekaClassifierReduceTask();
	}

	public Classifier call(Classifier arg0, Classifier arg1) throws Exception {
		ArrayList<Classifier> classifiers = new ArrayList<Classifier>();
		classifiers.add(arg0);
		classifiers.add(arg1);
		return classifierReduceTask.aggregate(classifiers);
	}

}
