package com.scmoreno.regression.multiple

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray

class NormalEquationRegression : MultipleLinearRegression() {

    override fun train(features: Array<DoubleArray>, target: DoubleArray) {
        val x = features.addIntercept()
        val xTransposed = x.transpose()
        val xTx = xTransposed dot x
        coefficients = (mk.linalg.inv(xTx) dot xTransposed dot mk.ndarray(target)).toDoubleArray()
    }

    private fun Array<DoubleArray>.addIntercept(): D2Array<Double> = mutableListOf<DoubleArray>()
        .let { result ->
            this.forEach { result += doubleArrayOf(1.0) + it }
            result
        }.let { mk.ndarray(it.toTypedArray()) }
}
