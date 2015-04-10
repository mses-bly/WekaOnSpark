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
CLASSIFIER=weka.classifiers.bayes.NaiveBayes
INPUT_FILES_PATH=${SELFD}/testing_files
#JOB=CLASSIFY
#JOB=SCORE
JOB=CROSSVALIDATION


mvn package

#Execution CLASSIFY
# ${SPARK_HOME}/spark-submit \
# 	--class ${LAUNCHER_CLASS} \
# 	${OPTIONS} \
# 	${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
# 	${JOB} \
# 	${CLASSIFIER} \
# 	${INPUT_FILES_PATH}/diabetes_train.csv \
# 	${INPUT_FILES_PATH}/diabetes_attr.csv

#Execution SCORE
# ${SPARK_HOME}/spark-submit \
# 	--class ${LAUNCHER_CLASS} \
# 	${OPTIONS} \
# 	${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
# 	${JOB} \
# 	${INPUT_FILES_PATH}/output_weka.classifiers.bayes.NaiveBayes_1426635940080.model \
# 	${INPUT_FILES_PATH}/diabetes_test.csv \
# 	${INPUT_FILES_PATH}/diabetes_attr.csv

#Execution CROSSVALIDATION
if [ ${JOB} = "CROSSVALIDATION" ]; then
	awk '{printf("%d %s\n", NR - 1 , $0)}' ${INPUT_FILES_PATH}/test_split.txt > ${INPUT_FILES_PATH}/test_split_with_ln.txt
fi

${SPARK_HOME}/spark-submit \
	--class ${LAUNCHER_CLASS} \
	${OPTIONS} \
	${WEKA_JAR_PATH}/integration-weka-spark-0.0.1-SNAPSHOT.jar \
	${JOB} \
	${INPUT_FILES_PATH}/test_split_with_ln.txt \
	10