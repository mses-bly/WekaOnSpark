package com.integration.weka.spark.data;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaPairRDD;

import com.integration.weka.spark.utils.Utils;

/**
 * Class for building folds from an RDD. Didn't make it static so we can have
 * some flexibility for folds in the future.
 */
public class KFoldsBuilder {
	private long numRecords;
	private long numFolds;

	public KFoldsBuilder(long numRecords, long numFolds) {
		this.numRecords = numRecords;
		this.numFolds = numFolds;
	}

	public ArrayList<JavaPairRDD<Long, String>> buildFolds(JavaPairRDD<Long, String> srcRDD) {
		ArrayList<JavaPairRDD<Long, String>> rddFolds = new ArrayList<JavaPairRDD<Long, String>>();

		long index = 0;
		long remainingRecords = numRecords;
		long remainingFolds = numFolds;

		while (index < numRecords) {
			long recordsByFold = (long) Math.ceil((double) remainingRecords / (double) remainingFolds);
			long startIndex = index;
			long endIndex = startIndex + recordsByFold - 1;

			JavaPairRDD<Long, String> fold = srcRDD.filter(Utils.getRangeFilter(startIndex, endIndex));
			rddFolds.add(fold);

			index += recordsByFold;
			remainingRecords -= recordsByFold;
			remainingFolds--;
		}

		return rddFolds;
	}

}
