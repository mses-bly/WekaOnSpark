package com.integration.weka.spark.jobs;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.Partitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.AggregateableEvaluation;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

import com.integration.weka.spark.classifiers.KFoldClassifierMapFunction;
import com.integration.weka.spark.classifiers.KFoldClassifierReduceFunction;
import com.integration.weka.spark.evaluation.KFoldEvaluationMapFunction;
import com.integration.weka.spark.headers.CSVHeaderMapFunction;
import com.integration.weka.spark.headers.CSVHeaderReduceFunction;
import com.integration.weka.spark.utils.Utils;

/**
 * K-Fold evaluation of classifiers.
 * @author Moises
 *
 */

public class KFoldEvaluationSparkJob {
	private static Logger LOGGER = Logger.getLogger(KFoldEvaluationSparkJob.class);

	public static void evaluate(SparkConf conf, JavaSparkContext context, List<String> classifierFullNameList, int kFolds, String trainDataInputFilePath, String outputFilePath) {
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(outputFilePath, "UTF-8");
			for (String classifierName : classifierFullNameList) {
				Evaluation evaluation = evaluateSingleClassifier(conf, context, classifierName, kFolds, trainDataInputFilePath);
				writer.println("======== K-Fold Evaluation for classifier: " + classifierName + " ========");
				writer.println(evaluation.toSummaryString(true));
			}
		} catch (Exception ex) {
			LOGGER.error("Could not complete k-fold evaluation job. Error: [" + ex + "]");
		} finally {
			if (writer != null) {
				writer.close();
			}
		}
	}

	//	private static class KFoldClassifierMapFunction implements PairFlatMapFunction<Iterator<List<String>>, Integer, Classifier> {
	//
	//		private Instances strippedHeader;
	//		private List<String> attributesNames;
	//		private int kFolds;
	//
	//		public KFoldClassifierMapFunction(Instances trainingHeader) throws DistributedWekaException {
	//			strippedHeader = CSVToARFFHeaderReduceTask.stripSummaryAtts(trainingHeader);
	//			strippedHeader.setClassIndex(trainingHeader.classIndex());
	//			attributesNames = CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(strippedHeader);
	//			
	//			if (intermediateClassifier != null && iteration > 0) {
	//			      // continue training
	//			      task.setClassifier(intermediateClassifier);
	//			      task.setContinueTrainingUpdateableClassifier(true);
	//			    }
	//		}
	//
	//		public Iterable<Tuple2<Integer, Classifier>> call(Iterator<List<String>> t) throws Exception {
	//			// TODO Auto-generated method stub
	//			return null;
	//		}
	//
	//	}
	//	private static Evaluation evaluateClassifier(SparkConf conf, JavaSparkContext context, String classifierName, int kFolds, String trainDataInputFilePath) {
	//
	//		JavaRDD<String> trainingData = context.textFile(trainDataInputFilePath);
	//
	//		int numPartitions = trainingData.partitions().size();
	//
	//		// Build Weka Header
	//		Instances header = trainingData.glom().map(new CSVHeaderMapFunction(Utils.parseCSVLine(trainingData.first().get(0)).length)).reduce(new CSVHeaderReduceFunction());
	//
	//
	//		JavaPairRDD<Integer, Classifier> classifierFolds = trainingData.mapPartitionsToPair(new PairFlatMapFunction<Iterator<List<String>>, Integer, Classifier>() {
	//
	//			public Iterable<Tuple2<Integer, Classifier>> call(Iterator<List<String>> t) throws Exception {
	//				// TODO Auto-generated method stub
	//				return null;
	//			}
	//		});
	//		//Classifiers used for evaluation.
	//		Classifier[] classifiers = new Classifier[kFolds];
	//
	//
	//	}
	private static Evaluation evaluateSingleClassifier(SparkConf conf, JavaSparkContext context, String classifierName, final int kFolds, String trainDataInputFilePath) {
		try {
			// Load the data file
			JavaRDD<String> csvFile = context.textFile(trainDataInputFilePath);

			// Group input data by partition rather than by lines
			JavaRDD<List<String>> trainingData = csvFile.glom();

			// Build Weka Header
			Instances header = trainingData.map(new CSVHeaderMapFunction(Utils.parseCSVLine(csvFile.first()).length)).reduce(new CSVHeaderReduceFunction());

			// Train one classifier per fold
			JavaPairRDD<Integer, Classifier> classifierPerFold = csvFile.mapPartitionsToPair(new KFoldClassifierMapFunction(header, classifierName, kFolds));
			classifierPerFold = classifierPerFold.sortByKey();
			classifierPerFold = classifierPerFold.partitionBy(new Partitioner() {
				
				@Override
				public int numPartitions() {
					return kFolds;
				}
				
				@Override
				public int getPartition(Object arg0) {
					return ((Number) arg0).intValue();
				}
			});

			JavaPairRDD<Integer, Classifier> reducedByFold = classifierPerFold.mapPartitionsToPair(new KFoldClassifierReduceFunction());
			List<Classifier> kFoldClassifiers = new ArrayList<Classifier>(kFolds);
			List<Tuple2<Integer, Classifier>> aggregated = reducedByFold.collect();
		      for (Tuple2<Integer, Classifier> t : aggregated) {
		    	  kFoldClassifiers.add(t._1(), t._2());
		      }


			//Aggregated classifiers. 
			//Position 0 (fold 1) will contain an aggregated classifier trained on folds [2:end]
			//Position 1 (fold 2) will contain an aggregated classifier trained on fold 0 and folds[3:end]
			//...
			//With that we will evaluate on folds 1, 2, ... with classifiers trained in the rest of folds
			//			List<Classifier> aggregatedClassifiers = new ArrayList<Classifier>();
			//			for (int i = 0; i < kFolds; i++) {
			//				//				Classifier removedClassifier = kFoldClassifiers.remove(i);
			//				aggregatedClassifiers.add(context.parallelize(kFoldClassifiers).reduce(new ClassifierReduceFunction()));
			//				//				kFoldClassifiers.add(i, removedClassifier);
			//			}

			List<Evaluation> evaluations = trainingData.map(new KFoldEvaluationMapFunction(header, kFoldClassifiers, kFolds)).collect();
			AggregateableEvaluation aggregateableEvaluation = new AggregateableEvaluation(evaluations.get(0));
			for (Evaluation eval : evaluations) {
				aggregateableEvaluation.aggregate(eval);
			}
			return aggregateableEvaluation;

		} catch (Exception ex) {
			LOGGER.error("Could not complete k-fold evaluation job for classifier [" + classifierName + "]. Error: [" + ex + "]");
		}
		return null;
	}
}
