package com.scmoreno.regression

import kotlin.random.Random
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray

class RidgeRegression(private val lambda: Double) {

    lateinit var coefficients: DoubleArray

    fun train(features: Array<DoubleArray>, target: DoubleArray) {
        val numFeatures = features[0].size
        coefficients = DoubleArray(numFeatures + 1)

        val x = features.addIntercept()

        val identityMatrix = mk.identity<Double>(x[0].size)

        val xTransposed = x.transpose()
        val y = mk.ndarray(target)

        val a = (xTransposed dot x) + (identityMatrix * lambda)
        val b = xTransposed dot y

        coefficients = a.toArray().solve(b.toDoubleArray())
    }

    fun predict(features: DoubleArray): Double =
        coefficients[0] + (1 until coefficients.size).sumOf { features[it - 1] * coefficients[it] }

    private fun Array<DoubleArray>.addIntercept(): D2Array<Double> = mutableListOf<DoubleArray>()
        .let { result ->
            this.forEach { result += it + 1.0 }
            result
        }.let { mk.ndarray(it.toTypedArray()) }

    // Extension function for solving a linear system of equations using Gaussian elimination
    private fun Array<DoubleArray>.solve(y: DoubleArray): DoubleArray {
        val n = this.size
        val a = this
        val x = DoubleArray(n)

        for (i in 0 until n) {
            for (j in i + 1 until n) {
                val factor = a[j][i] / a[i][i]
                for (k in i until n) {
                    a[j][k] -= factor * a[i][k]
                }
                y[j] -= factor * y[i]
            }
        }

        for (i in n - 1 downTo 0) {
            var sum = 0.0
            for (j in i + 1 until n) {
                sum += a[i][j] * x[j]
            }
            x[i] = (y[i] - sum) / a[i][i]
        }

        return x
    }
}

fun main() {
    val regression = RidgeRegression(0.1)

    val x: Array<DoubleArray> = arrayOf(
        DoubleArray(2) { Random.nextDouble(1.0, 3.0) },
        DoubleArray(2) { Random.nextDouble(1.0, 3.0) },
        DoubleArray(2) { Random.nextDouble(1.0, 3.0) },
        DoubleArray(2) { Random.nextDouble(1.0, 3.0) }
    )
    println(mk.ndarray(x))
    val y: DoubleArray = doubleArrayOf(1.0, 2.0, 3.0, 4.0)

    val input = doubleArrayOf(1.0, 2.0)

    regression.train(x, y)
    println(mk.ndarray(regression.coefficients))

    val prediction = regression.predict(input)

    println(prediction)
}