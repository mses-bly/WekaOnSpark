package com.integration.weka.spark.data;

import org.apache.spark.api.java.JavaRDD;

import weka.core.Instance;
import weka.core.Instances;

public class Dataset {
	private JavaRDD<Instance> data;
	private Instances headerWithSummary;
	private Instances headerNoSummary;

	public Dataset(JavaRDD<Instance> data, Instances headerWithSummary,
			Instances headerNoSummary) {
		this.data = data;
		this.headerWithSummary = headerWithSummary;
		this.headerNoSummary = headerNoSummary;
	}

	public JavaRDD<Instance> getData() {
		return data;
	}

	public void setData(JavaRDD<Instance> data) {
		this.data = data;
	}

	public Instances getHeaderWithSummary() {
		return headerWithSummary;
	}

	public void setHeaderWithSummary(Instances headerWithSummary) {
		this.headerWithSummary = headerWithSummary;
	}

	public Instances getHeaderNoSummary() {
		return headerNoSummary;
	}

	public void setHeaderNoSummary(Instances headerNoSummary) {
		this.headerNoSummary = headerNoSummary;
	}

}
