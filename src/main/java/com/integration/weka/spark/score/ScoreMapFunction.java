package com.integration.weka.spark.score;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.Function;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.DistributedWekaException;
import weka.distributed.WekaScoringMapTask;
import cern.colt.Arrays;

import com.integration.weka.spark.utils.Utils;

public class ScoreMapFunction implements Function<List<Instance>, List<String>> {

	private WekaScoringMapTask wekaScoringMapTask;
	private Attribute classAttribute;

	public ScoreMapFunction(Object model, Instances modelHeader, Instances dataHeader) throws DistributedWekaException {
		wekaScoringMapTask = new WekaScoringMapTask();
		wekaScoringMapTask.setModel(model, modelHeader, dataHeader);
		classAttribute = modelHeader.classAttribute();
	}

	public List<String> call(List<Instance> v1) throws Exception {
		ArrayList<String> predictions = new ArrayList<String>();
		for (Instance instance : v1) {
			String instancePrediction = "[" + instance.toString() + "] -> Prediction: ";
			double[] preds = wekaScoringMapTask.processInstance(instance);
			if (!classAttribute.isNominal()) {
				instancePrediction += "[" + Arrays.toString(preds) + "]";
			} else {
				instancePrediction += "[" + Arrays.toString(preds) + "] - " + classAttribute.value(Utils.getBiggestElementIndex(preds));
			}
			predictions.add(instancePrediction);
		}
		return predictions;
	}
}
