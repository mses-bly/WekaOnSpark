package com.integration.weka.spark.evaluation;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.Function2;

import weka.classifiers.evaluation.Evaluation;
import weka.distributed.WekaClassifierEvaluationReduceTask;

public class EvaluationReduceFunction implements Function2<Evaluation, Evaluation, Evaluation> {

	private WekaClassifierEvaluationReduceTask evaluationReduceTask = new WekaClassifierEvaluationReduceTask();

	public Evaluation call(Evaluation arg0, Evaluation arg1) throws Exception {
		List<Evaluation> lst = new ArrayList<Evaluation>();
		lst.add(arg0);
		lst.add(arg1);
		return evaluationReduceTask.aggregate(lst);
	}


}
