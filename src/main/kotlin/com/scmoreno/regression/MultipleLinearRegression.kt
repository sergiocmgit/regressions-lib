package com.scmoreno.regression

abstract class MultipleLinearRegression {

    lateinit var coefficients: DoubleArray

    abstract fun train(features: Array<DoubleArray>, target: DoubleArray)

    fun predict(features: DoubleArray): Double =
        coefficients[0] + (1 until coefficients.size).sumOf { features[it - 1] * coefficients[it] }
}