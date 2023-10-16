package com.scmoreno.regression.multiple

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NormalEquationRegressionTest {

    private val regression = NormalEquationRegression()

    @Test
    fun `should calculate coefficients`() {
        // Given
        val x = arrayOf(
            doubleArrayOf(1.26, 2.33),
            doubleArrayOf(2.21, 2.97),
            doubleArrayOf(2.43, 2.24),
            doubleArrayOf(2.55, 2.41)
        )
        val y = doubleArrayOf(2.112, 2.054, 2.472, 2.476)
        val input = doubleArrayOf(1.0, 1.0)
        val expected = 2.708867807550982
        // When
        regression.train(x, y)
        val result = regression.predict(input)
        // Then
        assertEquals(expected, result)
    }
}