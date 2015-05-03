package com.integration.weka.spark.utils;

import org.apache.spark.Partitioner;

public class IntegerPartitioner extends Partitioner {

	private int numPartitions;

	public IntegerPartitioner(int numPartitions) {
		this.numPartitions = numPartitions;
	}

	@Override
	public int getPartition(Object arg0) {
		return ((Number) arg0).intValue();
	}

	@Override
	public int numPartitions() {
		return numPartitions;
	}

}
