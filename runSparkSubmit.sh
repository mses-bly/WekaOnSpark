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
CLASSIFIER=weka.classifiers.trees.RandomForest
INPUT_FILES_PATH=${SELFD}/testing_files
# JOB=HEADER
# JOB=CLASSIFY
# JOB=SCORE
JOB=EVALUATION


mvn package

case ${JOB} in
	
	HEADER )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		${INPUT_FILES_PATH}/diabetes.csv
	;;

	CLASSIFY )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		${CLASSIFIER} \
		${INPUT_FILES_PATH}/datapolytest_reduced.csv
	;;

	SCORE )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		${INPUT_FILES_PATH}/polytest/output_model_weka.classifiers.trees.RandomForest_20151126031114.model \
		${INPUT_FILES_PATH}/datapolytest_reduced_test.csv \
		${INPUT_FILES_PATH}/datapolytest_reduced.csv
	;;

	EVALUATION )
		${SPARK_HOME}/spark-submit \
		--class ${LAUNCHER_CLASS} \
		${OPTIONS} \
		${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
		${JOB} \
		2 \
		weka.classifiers.trees.RandomForest \
		weka.classifiers.functions.Logistic \
		${INPUT_FILES_PATH}/diabetes.csv \
		${INPUT_FILES_PATH}/diabetes.csv
	;;

esac
