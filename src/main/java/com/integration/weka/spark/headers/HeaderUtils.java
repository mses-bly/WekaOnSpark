package com.integration.weka.spark.headers;

import java.util.List;

import org.apache.log4j.Logger;

import weka.core.Instances;
import weka.distributed.CSVToARFFHeaderMapTask;
import weka.distributed.DistributedWekaException;

public class HeaderUtils {
	private static Logger LOGGER = Logger.getLogger(HeaderUtils.class);
	
	/**
	 * Returns a header just from attributes, no need for data. No
	 * parallelization
	 * 
	 * @param attNames
	 *            attributes used to build this header
	 * @return Header as Instances
	 */

	public static Instances getHeaderFromAttributes(List<String> attNames) {
		CSVToARFFHeaderMapTask csvToARFFHeaderMapTask = new CSVToARFFHeaderMapTask();
		try {
			return csvToARFFHeaderMapTask.getHeader(attNames.size(), attNames);
		} catch (DistributedWekaException e) {
			LOGGER.error("Could not build header for this attributes. Error: [" + e + "]");
		}
		return null;
	}

}
