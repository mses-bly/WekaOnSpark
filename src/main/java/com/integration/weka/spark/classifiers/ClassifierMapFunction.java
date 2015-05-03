package com.integration.weka.spark.classifiers;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.WekaClassifierMapTask;

public class ClassifierMapFunction implements PairFlatMapFunction<Iterator<String>, Integer, Classifier> {

	private ArrayList<WekaClassifierMapTask> classifierMapTasks;
	private Instances strippedHeader;
	private List<String> attributesNames;
	private int kFolds;

	public ClassifierMapFunction(Instances trainingHeader, String classifierClassName, int kFolds) throws Exception {
		strippedHeader = CSVToARFFHeaderReduceTask.stripSummaryAtts(trainingHeader);
		strippedHeader.setClassIndex(trainingHeader.classIndex());
		attributesNames = CSVToARFFHeaderMapTask.instanceHeaderToAttributeNameList(strippedHeader);
		Class<?> clazz = Class.forName(classifierClassName);
		Object object = clazz.newInstance();
		classifierMapTasks = new ArrayList<WekaClassifierMapTask>();
		this.kFolds = kFolds;

		for (int i = 0; i < kFolds; i++) {
			WekaClassifierMapTask foldTask = new WekaClassifierMapTask();
			foldTask.setClassifier((Classifier) object);
			foldTask.setTotalNumFolds(kFolds);
			foldTask.setFoldNumber(i + 1);
			foldTask.setup(strippedHeader);
			classifierMapTasks.add(foldTask);
		}
	}

	public Iterable<Tuple2<Integer, Classifier>> call(Iterator<String> t) throws Exception {
		List<Tuple2<Integer, Classifier>> classifiers = new ArrayList<Tuple2<Integer, Classifier>>();
		CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
		rowParser.initParserOnly(attributesNames);
		while (t.hasNext()) {
			String[] parsedRow = rowParser.parseRowOnly(t.next());
			Instance currentInstance = rowParser.makeInstance(strippedHeader, true, parsedRow);
			for (int j = 0; j < kFolds; j++) {
				classifierMapTasks.get(j).processInstance(currentInstance);
			}
		}
		for (int i = 0; i < kFolds; i++) {
			classifierMapTasks.get(i).finalizeTask();
			classifiers.add(new Tuple2<Integer, Classifier>(i, classifierMapTasks.get(i).getClassifier()));
		}
		return classifiers;
	}

}
