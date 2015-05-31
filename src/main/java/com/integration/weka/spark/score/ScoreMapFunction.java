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

public class ScoreMapFunction implements Function<Instance, String> {

	private WekaScoringMapTask wekaScoringMapTask;
	private Attribute classAttribute;

	public ScoreMapFunction(Object model, Instances modelHeader, Instances dataHeader) throws DistributedWekaException {
		wekaScoringMapTask = new WekaScoringMapTask();
		wekaScoringMapTask.setModel(model, modelHeader, dataHeader);
		classAttribute = modelHeader.classAttribute();
	}

	public String call(Instance v1) throws Exception {
		String instancePrediction = "[" + v1.toString() + "] -> Prediction: ";
		double[] preds = wekaScoringMapTask.processInstance(v1);
		if (!classAttribute.isNominal()) {
			instancePrediction += "[" + Arrays.toString(preds) + "]";
		} else {
			instancePrediction += "[" + Arrays.toString(preds) + "] - "
					+ classAttribute.value(Utils.getBiggestElementIndex(preds));
		}
		return instancePrediction;
	}
}
