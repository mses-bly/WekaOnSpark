package com.integration.weka.spark.headers;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

public class RandomShuffleMapFunction implements PairFlatMapFunction<Iterator<String>, Integer, String> {

	private int numSlices;
	private Random random;

	public RandomShuffleMapFunction(int numSlices, Random random) {
		//don't think we will exceed Integer capacity
		this.numSlices = numSlices;
		this.random = random;
	}

	@Override
	public Iterable<Tuple2<Integer, String>> call(Iterator<String> t) throws Exception {
		ArrayList<Tuple2<Integer, String>> randomlyAssignedInstances = new ArrayList<>();
		while (t.hasNext()) {
			int rnd = random.nextInt(numSlices);
			Tuple2<Integer, String> tupl = new Tuple2<Integer, String>(rnd, t.next());
			randomlyAssignedInstances.add(tupl);
		}
		return randomlyAssignedInstances;
	}

}
