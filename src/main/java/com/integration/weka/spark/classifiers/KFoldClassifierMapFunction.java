package com.integration.weka.spark.classifiers;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.Function;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.CSVToARFFHeaderReduceTask;
import weka.distributed.WekaClassifierMapTask;

/**
 * Builds one classifier per fold.
 *
 */
public class KFoldClassifierMapFunction implements Function<List<String>, List<Classifier>> {

	private static Logger LOGGER = Logger.getLogger(KFoldClassifierMapFunction.class);

	private ArrayList<WekaClassifierMapTask> classifierMapTasks;
	private Instances strippedHeader;
	private List<String> attributesNames;
	private int kFolds;

	/**
	 * 
	 * @param trainingHeader: training header from the data
	 * @param classifierClassName: full qualified name of the classifier
	 * @param kFolds: number of folds
	 */
	public KFoldClassifierMapFunction(Instances trainingHeader, String classifierClassName, int kFolds) {
		try {
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

		} catch (Exception ex) {
			LOGGER.error("Could not create k-folded classifier [" + classifierClassName + "]. Error: [" + ex + "]");
		}
	}

	public List<Classifier> call(List<String> v1) throws Exception {
		ArrayList<Classifier> classifiers = new ArrayList<Classifier>();
		CSVToARFFHeaderMapTask rowParser = new CSVToARFFHeaderMapTask();
		rowParser.initParserOnly(attributesNames);
		try {
			for (int i = 0; i < v1.size(); i++) {
				String[] parsedRow = rowParser.parseRowOnly(v1.get(i));
				Instance currentInstance = rowParser.makeInstance(strippedHeader, true, parsedRow);
				classifierMapTasks.get(i % kFolds).processInstance(currentInstance);
			}
			for (int i = 0; i < kFolds; i++) {
				classifierMapTasks.get(i).finalizeTask();
				classifiers.add(classifierMapTasks.get(i).getClassifier());
			}
			return classifiers;

		} catch (Exception ex) {
			LOGGER.error("Could not train k-folded classifier. Error: [" + ex + "]");
		}
		return null;
	}


}
