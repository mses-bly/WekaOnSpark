package com.integration.weka.spark.scoring;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.Function;

import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.DistributedWekaException;
import weka.distributed.WekaScoringMapTask;

import com.integration.weka.spark.utils.Utils;

public class ScoringMapFunction implements Function<List<Instance>, List<String>> {

	private static Logger LOGGER = Logger.getLogger(ScoringMapFunction.class);

	private WekaScoringMapTask wekaScoringMapTask;

	public ScoringMapFunction(Object model, Instances modelHeader, Instances dataHeader) {
		try {
			wekaScoringMapTask = new WekaScoringMapTask();
			wekaScoringMapTask.setModel(model, modelHeader, dataHeader);
		} catch (DistributedWekaException e) {
			LOGGER.error("Could not complete scoring for this data set. Error: [" + e + "]");
		}

	}

	public List<String> call(List<Instance> v1) throws Exception {
		ArrayList<String> predictions = new ArrayList<String>();
		for (Instance instance : v1) {
			String prediction = instance.toString() + ", ";
			double[] preds = wekaScoringMapTask.processInstance(instance);
			prediction += Utils.doubleArrayToString(preds) + "\n";
			predictions.add(prediction);
		}
		return predictions;
	}
}
