#!/bin/bash
STARTD=${PWD}
SELFD=$(cd $(dirname ${0}) >/dev/null 2>&1; pwd)
SELF=$(basename ${0})
SELFN=$(basename ${SELFD})
SELFU=${SELF%.*}
SELFZ=${SELFD}/${SELF}

[ -d spark-1.2.1-bin-hadoop2.4 ] || {
 	curl -sL http://d3kbcqa49mib13.cloudfront.net/spark-1.2.1-bin-hadoop2.4.tgz | tar vzx || exit ${LINENO}
}

[ -x spark-1.2.1-bin-hadoop2.4/bin ] || exit ${LINENO}

SPARK_HOME=${SELFD}/spark-1.2.1-bin-hadoop2.4/bin
LAUNCHER_CLASS=com.integration.weka.spark.utils.Launcher
WEKA_JAR_PATH=${SELFD}/target
# CLASSIFIER=weka.classifiers.trees.RandomForest
CLASSIFIER=weka.classifiers.bayes.NaiveBayes
# CLASSIFIER=weka.classifiers.functions.LinearRegression
# CLASSIFIER=weka.classifiers.functions.Logistic
INPUT_FILES_PATH=${SELFD}/testing_files
JOB=HEADER
# JOB=CLASSIFY
# JOB=SCORE
# JOB=EVALUATION
# JOB=SHUFFLE


mvn package

case ${JOB} in
	
	HEADER )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/diabetes.csv
	;;

	CLASSIFY )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/diabetes.csv \
		-classifier-name ${CLASSIFIER}
	;;	

	SCORE )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/datapolytest_reduced.csv \
		-score-file ${INPUT_FILES_PATH}/datapolytest_reduced.csv \
		-classifier-model ${INPUT_FILES_PATH}/datapolytest/classifier_weka.classifiers.functions.LinearRegression_2015-05-03_16:24:58.model
	;;

	# EVALUATION )
	# 	${SPARK_HOME}/spark-submit \
	# 	--class ${LAUNCHER_CLASS} \
	# 	${OPTIONS} \
	# 	${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
	# 	${JOB} \
	# 	-input-file ${INPUT_FILES_PATH}/diabetes.csv \
	# 	-test-file ${INPUT_FILES_PATH}/diabetes.csv \
	# 	-classifier-name ${CLASSIFIER}
	# ;;
	
	EVALUATION )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/diabetes.csv \
		-folds 10 \
		-shuffle true\
		-classifier-name ${CLASSIFIER}
	;;

	SHUFFLE )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/diabetes.csv \
		-output-file-name ${INPUT_FILES_PATH}/diabetes_random.csv\
	;;
esac
