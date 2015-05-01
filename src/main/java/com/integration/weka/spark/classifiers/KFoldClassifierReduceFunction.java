package com.integration.weka.spark.classifiers;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.Function2;

import weka.classifiers.Classifier;
import weka.distributed.WekaClassifierReduceTask;

public class KFoldClassifierReduceFunction implements Function2<List<Classifier>, List<Classifier>, List<Classifier>> {

	private int kFolds;
	private WekaClassifierReduceTask classifierReduceTask;

	public KFoldClassifierReduceFunction(int kFolds) {
		this.kFolds = kFolds;
		classifierReduceTask = new WekaClassifierReduceTask();
	}
	public List<Classifier> call(List<Classifier> v1, List<Classifier> v2) throws Exception {
		ArrayList<Classifier> classifiers = new ArrayList<Classifier>();
		for (int i = 0; i < kFolds; i++) {
			ArrayList<Classifier> i_FoldClassfiers = new ArrayList<Classifier>();
			i_FoldClassfiers.add(v1.get(i));
			i_FoldClassfiers.add(v2.get(i));
			classifiers.add(classifierReduceTask.aggregate(i_FoldClassfiers));
		}
		return classifiers;
	}

}
