package com.scmoreno.regression.simple

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.DynamicTest.dynamicTest
import org.junit.jupiter.api.TestFactory

class GradientDescentRegressionTest {

    private val learningRate = 0.01
    private val numIterations = 1000
    private val regression = GradientDescentRegression(learningRate, numIterations)

    @TestFactory
    fun `should calculate coefficients`() = listOf(
        TestCase(
            name = "first one",
            x = DoubleArray(10) { it.toDouble() },
            y = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 7.0, 8.0, 8.0, 9.0, 10.0, 12.0),
            expected = 2.4030736418378
        ),
        TestCase(
            name = "second one",
            x = DoubleArray(5) { it.plus(1).toDouble() },
            y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0),
            expected = 2.0125808140125887
        )
    ).map { (name, x, y, expected) ->
        dynamicTest(name) {
            // When
            regression.train(x, y)
            val result = regression.predict(1.0)
            // Then
            assertEquals(expected, result)
        }
    }

    private data class TestCase(
        val name: String,
        val x: DoubleArray,
        val y: DoubleArray,
        val expected: Double
    )
}