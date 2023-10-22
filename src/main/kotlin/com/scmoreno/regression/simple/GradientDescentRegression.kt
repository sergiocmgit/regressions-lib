package com.scmoreno.regression.simple

class GradientDescentRegression(
    private val learningRate: Double,
    private val numIterations: Int
) : SimpleLinearRegression() {

    override fun train(features: DoubleArray, targets: DoubleArray) {
        var beta0 = 0.0
        var beta1 = 0.0

        for (iteration in 1..numIterations) {
            var gradient0 = 0.0
            var gradient1 = 0.0

            for (i in features.indices) {
                val prediction = beta0 + beta1 * features[i]
                val error = prediction - targets[i]

                gradient0 += (1.0 / features.size) * 2 * error
                gradient1 += (1.0 / features.size) * 2 * error * features[i]
            }

            beta0 -= learningRate * gradient0
            beta1 -= learningRate * gradient1
        }

        coefficients = doubleArrayOf(beta0, beta1)
    }
}
