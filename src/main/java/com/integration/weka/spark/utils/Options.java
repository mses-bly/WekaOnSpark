package com.integration.weka.spark.utils;

import java.util.HashMap;

public class Options {
	private HashMap<String, String> options;

	public Options() {
		options = new HashMap<String, String>();
	}

	public boolean hasOption(String optionName) {
		return options.containsKey(optionName);
	}

	public String getOption(String optionName){
		return options.get(optionName);
	}

	public void addOption(String optionName, String optionValue) {
		options.put(optionName, optionValue);
	}

}
