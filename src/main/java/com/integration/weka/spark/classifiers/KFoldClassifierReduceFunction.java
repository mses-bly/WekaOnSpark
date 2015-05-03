package com.integration.weka.spark.classifiers;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;
import weka.classifiers.Classifier;
import weka.distributed.DistributedWekaException;
import weka.distributed.WekaClassifierReduceTask;

public class KFoldClassifierReduceFunction implements PairFlatMapFunction<Iterator<Tuple2<Integer, Classifier>>, Integer, Classifier> {

	protected List<Tuple2<Integer, Classifier>> m_reducedForFold = new ArrayList<Tuple2<Integer, Classifier>>();

	/**
	 * Before calling here all classifiers should be within the same partition.
	 */
	public Iterable<Tuple2<Integer, Classifier>> call(Iterator<Tuple2<Integer, Classifier>> t) throws Exception {
		int foldNum = -1;

		List<Classifier> classifiers = new ArrayList<Classifier>();
		while (t.hasNext()) {
			Tuple2<Integer, Classifier> partial = t.next();
			if (foldNum < 0) {
				foldNum = partial._1().intValue();
			} else {
				if (partial._1().intValue() != foldNum) {
					throw new DistributedWekaException("[WekaClassifierEvaluation] build " + "classifiers reduce phase: was not expecting fold number " + "to change within a partition!");
				}
			}
			classifiers.add(partial._2());
		}

		WekaClassifierReduceTask reduceTask = new WekaClassifierReduceTask();
		Classifier intermediateClassifier = reduceTask.aggregate(classifiers, null, false);

		m_reducedForFold.add(new Tuple2<Integer, Classifier>(foldNum, intermediateClassifier));

		return m_reducedForFold;
	}

}
