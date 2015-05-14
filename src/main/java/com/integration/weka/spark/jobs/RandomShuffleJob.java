package com.integration.weka.spark.jobs;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;

import scala.Tuple2;

import com.integration.weka.spark.headers.RandomShuffleMapFunction;
import com.integration.weka.spark.utils.Constants;
import com.integration.weka.spark.utils.IntegerPartitioner;
import com.integration.weka.spark.utils.Options;

public class RandomShuffleJob {

	public static JavaRDD<String> randomlyShuffleData(SparkConf conf, JavaSparkContext context, Options opts) throws Exception {
		if (!opts.hasOption(Constants.OPTION_INPUT_FILE)) {
			throw new Exception("Must provide data file for RANDOM SHUFFLE job");
		}

		String inputFilePath = opts.getOption(Constants.OPTION_INPUT_FILE);
		Random random = new Random(System.currentTimeMillis());

		JavaRDD<String> csvFile = context.textFile(inputFilePath);

		int slices = 1;
		if (opts.hasOption(Constants.OPTION_FOLDS)) {
			slices = Integer.valueOf(opts.getOption(Constants.OPTION_FOLDS));
		} else {
			slices = (int) csvFile.count();
		}

		JavaPairRDD<Integer, String> randomized = csvFile.mapPartitionsToPair(new RandomShuffleMapFunction(slices, random));
		randomized = randomized.sortByKey();
		if (opts.hasOption(Constants.OPTION_FOLDS)) {
			//if we are using folds, group all data by assigned partition
			randomized = randomized.partitionBy(new IntegerPartitioner(slices));
		}
		JavaRDD<String> finalDataSet = randomized.mapPartitions(new FlatMapFunction<Iterator<Tuple2<Integer,String>>, String>() {
			@Override
			public Iterable<String> call(Iterator<Tuple2<Integer, String>> t) throws Exception {
				ArrayList<String> data = new ArrayList<>();
				while (t.hasNext()) {
					data.add(t.next()._2());
				}
				return data;
			}
		});

		if (opts.hasOption(Constants.OPTION_OUTPUT_FILE_NAME)) {
			finalDataSet.saveAsTextFile(opts.getOption(Constants.OPTION_OUTPUT_FILE_NAME));
		}
		return finalDataSet;

	}

}
