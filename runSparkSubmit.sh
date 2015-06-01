#!/bin/bash
STARTD=${PWD}
SELFD=$(cd $(dirname ${0}) >/dev/null 2>&1; pwd)
SELF=$(basename ${0})
SELFN=$(basename ${SELFD})
SELFU=${SELF%.*}
SELFZ=${SELFD}/${SELF}
CONF=${SELFD}/conf

[ -d spark-1.3.1-bin-hadoop2.6 ] || {
	curl -sL http://www.us.apache.org/dist/spark/spark-1.3.1/spark-1.3.1-bin-hadoop2.6.tgz || exit ${LINENO}
}

[ -x spark-1.3.1-bin-hadoop2.6/bin ] || exit ${LINENO}

SPARK_HOME=${SELFD}/spark-1.3.1-bin-hadoop2.6/bin
SPARK_CONF=${SELFD}/spark-1.3.1-bin-hadoop2.6/conf

LAUNCHER_CLASS=com.integration.weka.spark.utils.Launcher
WEKA_JAR_PATH=${SELFD}/target
CLASSIFIER=weka.classifiers.trees.RandomForest
# CLASSIFIER=weka.classifiers.bayes.NaiveBayes
# CLASSIFIER=weka.classifiers.functions.LinearRegression
# CLASSIFIER=weka.classifiers.functions.Logistic
INPUT_FILES_PATH=${SELFD}/testing_files
# JOB=HEADER
# JOB=CLASSIFY
# JOB=SCORE
JOB=EVALUATION
#JOB=SHUFFLE

cp ${CONF}/log4j.properties ${SPARK_CONF}

mvn package

case ${JOB} in
	
	HEADER )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		--properties-file ${CONF}/spark-configuration.conf \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/friedman.csv
	;;

	CLASSIFY )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		--properties-file ${CONF}/spark-configuration.conf \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/friedman.csv \
		-classifier-name ${CLASSIFIER}
	;;	

	SCORE )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		--properties-file ${CONF}/spark-configuration.conf \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/datapolytest_reduced.csv \
		-score-file ${INPUT_FILES_PATH}/datapolytest_reduced.csv \
		-classifier-model ${INPUT_FILES_PATH}/datapolytest/classifier_weka.classifiers.functions.LinearRegression_2015-05-03_16:24:58.model
	;;

	EVALUATION )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
	  --properties-file ${CONF}/spark-configuration.conf \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/friedman.csv \
		-test-file ${INPUT_FILES_PATH}/friedman.csv \
		-classifier-name ${CLASSIFIER}
	;;
	
	# EVALUATION )
	# 	${SPARK_HOME}/spark-submit \
	# 	--class ${LAUNCHER_CLASS} \
	# 	--properties-file ${CONF}/spark-configuration.conf \
	# 	${OPTIONS} \
	# 	${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
	# 	${JOB} \
	# 	-input-file ${INPUT_FILES_PATH}/diabetes.csv \
	# 	-folds 10 \
	# 	-shuffle true\
	# 	-classifier-name ${CLASSIFIER}
	# ;;

	SHUFFLE )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		--properties-file ${CONF}/spark-configuration.conf \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		-input-file ${INPUT_FILES_PATH}/diabetes.csv \
		-output-file-name ${INPUT_FILES_PATH}/test_diabetes\
	;;
esac