package com.scmoreno.regression

abstract class SimpleLinearRegression {

    lateinit var coefficients: DoubleArray

    abstract fun train(features: DoubleArray, target: DoubleArray)

    fun predict(feature: Double): Double = coefficients[0] + feature * coefficients[1]
}