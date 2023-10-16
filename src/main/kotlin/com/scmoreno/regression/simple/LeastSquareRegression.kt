package com.scmoreno.regression.simple

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.times

class LeastSquareRegression : SimpleLinearRegression() {

    override fun train(features: DoubleArray, target: DoubleArray) {
        val numberOfSamples = features.size

        val xArray = mk.ndarray(features)
        val yArray = mk.ndarray(target)

        val meanX = mk.stat.mean(xArray)
        val meanY = mk.stat.mean(yArray)

        val crossDeviation = mk.math.sum(yArray * xArray) - (numberOfSamples * meanY * meanX)
        val deviationX = mk.math.sum(xArray * xArray) - (numberOfSamples * meanX * meanX)

        val coefficient1 = crossDeviation / deviationX
        val coefficient0 = meanY - coefficient1 * meanX
        coefficients = doubleArrayOf(coefficient0, coefficient1)
    }
}
