package com.scmoreno.regression.simple

abstract class SimpleLinearRegression {

    lateinit var coefficients: DoubleArray

    abstract fun train(features: DoubleArray, targets: DoubleArray)

    fun predict(feature: Double): Double = coefficients[0] + feature * coefficients[1]
}