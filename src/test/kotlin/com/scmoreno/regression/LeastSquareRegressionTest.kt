package com.scmoreno.regression

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.DynamicTest.dynamicTest
import org.junit.jupiter.api.TestFactory

class LeastSquareRegressionTest {

    private val regression = LeastSquareRegression()

    @TestFactory
    fun `should calculate coefficients`() = listOf(
        TestCase(
            name = "simple case",
            x = DoubleArray(10) { it.toDouble() },
            y = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 7.0, 8.0, 8.0, 9.0, 10.0, 12.0),
            expected = 2.4060606060606062
        ),
        TestCase(
            name = "complex case",
            x = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0),
            y = doubleArrayOf(420.0, 365.0, 285.0, 220.0, 176.0, 117.0, 69.0, 34.0, 5.0),
            expected = 448.2422222222222
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