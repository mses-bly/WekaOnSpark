package com.integration.weka.spark.classifiers;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;
import weka.classifiers.Classifier;
import weka.distributed.WekaClassifierReduceTask;

public class ClassifierReduceFunction implements PairFlatMapFunction<Iterator<Tuple2<Integer, Classifier>>, Integer, Classifier> {

	protected List<Tuple2<Integer, Classifier>> reducedByFold = new ArrayList<Tuple2<Integer, Classifier>>();

	/**
	 * Before calling here, all same fold classifiers should be within the same partition.
	 * @throws Exception 
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
					throw new Exception("One partition must contains all elements of a fold!!");
				}
			}
			classifiers.add(partial._2());
		}

		WekaClassifierReduceTask reduceTask = new WekaClassifierReduceTask();
		Classifier intermediateClassifier;
		intermediateClassifier = reduceTask.aggregate(classifiers, null, false);
		reducedByFold.add(new Tuple2<Integer, Classifier>(foldNum, intermediateClassifier));
		return reducedByFold;
	}

}
