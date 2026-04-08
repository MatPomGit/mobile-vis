package pl.edu.mobilecv.util

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.sqrt

/**
 * A generic Unscented Kalman Filter (UKF) implementation using OpenCV's Mat.
 *
 * This implementation uses the standard Van der Merwe's scaled unscented transform.
 */
class UnscentedKalmanFilter(
    private val stateDim: Int,
    private val measureDim: Int,
    private val alpha: Double = 1e-3,
    private val beta: Double = 2.0,
    private val kappa: Double = 0.0
) {
    var statePost = Mat.zeros(stateDim, 1, CvType.CV_64F)
    var errorCovPost = Mat.eye(stateDim, stateDim, CvType.CV_64F)
    
    var processNoiseCov = Mat.eye(stateDim, stateDim, CvType.CV_64F)
    var measurementNoiseCov = Mat.eye(measureDim, measureDim, CvType.CV_64F)

    private val lambda = alpha * alpha * (stateDim + kappa) - stateDim
    private val weightsMean = Mat(1, 2 * stateDim + 1, CvType.CV_64F)
    private val weightsCov = Mat(1, 2 * stateDim + 1, CvType.CV_64F)

    init {
        val w0m = lambda / (stateDim + lambda)
        val w0c = w0m + (1 - alpha * alpha + beta)
        val wi = 1.0 / (2.0 * (stateDim + lambda))

        weightsMean.put(0, 0, w0m)
        weightsCov.put(0, 0, w0c)
        for (i in 1 until 2 * stateDim + 1) {
            weightsMean.put(0, i, wi)
            weightsCov.put(0, i, wi)
        }
    }

    /**
     * Generates sigma points based on current state and covariance.
     */
    private fun generateSigmaPoints(x: Mat, P: Mat): List<Mat> {
        val sigmaPoints = mutableListOf<Mat>()
        sigmaPoints.add(x.clone())

        val sqrtP = Mat()
        val scaledP = Mat()
        Core.multiply(P, org.opencv.core.Scalar(stateDim + lambda), scaledP)
        
        // Use Cholesky decomposition to get sqrt(P)
        // OpenCV doesn't have a direct Cholesky in Core easily, 
        // but here P is symmetric positive definite.
        // We use SVD to get the square root: P = U * W * V^T, so sqrt(P) = U * sqrt(W)
        val w = Mat()
        val u = Mat()
        val vt = Mat()
        Core.SVDecomp(scaledP, w, u, vt)
        
        // sqrtP = u * diag(sqrt(w))
        val sqrtW = Mat.zeros(stateDim, stateDim, CvType.CV_64F)
        for (i in 0 until stateDim) {
            sqrtW.put(i, i, sqrt(w.get(i, 0)[0]))
        }
        Core.gemm(u, sqrtW, 1.0, Mat(), 0.0, sqrtP)

        for (i in 0 until stateDim) {
            val col = Mat(stateDim, 1, CvType.CV_64F)
            for (j in 0 until stateDim) {
                col.put(j, 0, sqrtP.get(j, i)[0])
            }
            
            val p1 = Mat()
            Core.add(x, col, p1)
            sigmaPoints.add(p1)

            val p2 = Mat()
            Core.subtract(x, col, p2)
            sigmaPoints.add(p2)
        }
        return sigmaPoints
    }

    /**
     * UKF Prediction step.
     * @param transitionFunc A function that transforms a sigma point to the next state.
     */
    fun predict(transitionFunc: (Mat) -> Mat) {
        // 1. Generate sigma points
        val sigmaPoints = generateSigmaPoints(statePost, errorCovPost)

        // 2. Propagate sigma points through transition function
        val propagatedSigmaPoints = sigmaPoints.map { transitionFunc(it) }

        // 3. Calculate predicted state mean
        val xPred = Mat.zeros(stateDim, 1, CvType.CV_64F)
        for (i in propagatedSigmaPoints.indices) {
            val weighted = Mat()
            Core.multiply(propagatedSigmaPoints[i], org.opencv.core.Scalar(weightsMean.get(0, i)[0]), weighted)
            Core.add(xPred, weighted, xPred)
        }

        // 4. Calculate predicted covariance
        val pPred = Mat()
        processNoiseCov.copyTo(pPred)
        for (i in propagatedSigmaPoints.indices) {
            val diff = Mat()
            Core.subtract(propagatedSigmaPoints[i], xPred, diff)
            val term = Mat()
            Core.gemm(diff, diff.t(), weightsCov.get(0, i)[0], Mat(), 0.0, term)
            Core.add(pPred, term, pPred)
        }

        statePost = xPred
        errorCovPost = pPred
    }

    /**
     * UKF Update step.
     * @param measurement Measurement vector (measureDim x 1).
     * @param measurementFunc A function that transforms a state sigma point to measurement space.
     */
    fun update(measurement: Mat, measurementFunc: (Mat) -> Mat) {
        // 1. Generate sigma points from predicted state
        val sigmaPoints = generateSigmaPoints(statePost, errorCovPost)

        // 2. Map sigma points to measurement space
        val zSigmaPoints = sigmaPoints.map { measurementFunc(it) }

        // 3. Calculate predicted measurement mean
        val zPred = Mat.zeros(measureDim, 1, CvType.CV_64F)
        for (i in zSigmaPoints.indices) {
            val weighted = Mat()
            Core.multiply(zSigmaPoints[i], org.opencv.core.Scalar(weightsMean.get(0, i)[0]), weighted)
            Core.add(zPred, weighted, zPred)
        }

        // 4. Calculate measurement covariance S
        val s = Mat()
        measurementNoiseCov.copyTo(s)
        for (i in zSigmaPoints.indices) {
            val diff = Mat()
            Core.subtract(zSigmaPoints[i], zPred, diff)
            val term = Mat()
            Core.gemm(diff, diff.t(), weightsCov.get(0, i)[0], Mat(), 0.0, term)
            Core.add(s, term, s)
        }

        // 5. Calculate cross-covariance Pxz
        val pxz = Mat.zeros(stateDim, measureDim, CvType.CV_64F)
        for (i in sigmaPoints.indices) {
            val xDiff = Mat()
            Core.subtract(sigmaPoints[i], statePost, xDiff)
            val zDiff = Mat()
            Core.subtract(zSigmaPoints[i], zPred, zDiff)
            
            val term = Mat()
            Core.gemm(xDiff, zDiff.t(), weightsCov.get(0, i)[0], Mat(), 0.0, term)
            Core.add(pxz, term, pxz)
        }

        // 6. Kalman Gain K = Pxz * S^-1
        val k = Mat()
        Core.gemm(pxz, s.inv(), 1.0, Mat(), 0.0, k)

        // 7. Update state mean
        val zDiff = Mat()
        Core.subtract(measurement, zPred, zDiff)
        val updateTerm = Mat()
        Core.gemm(k, zDiff, 1.0, Mat(), 0.0, updateTerm)
        Core.add(statePost, updateTerm, statePost)

        // 8. Update covariance P = P - K * S * K^T
        val kskT = Mat()
        val ks = Mat()
        Core.gemm(k, s, 1.0, Mat(), 0.0, ks)
        Core.gemm(ks, k.t(), 1.0, Mat(), 0.0, kskT)
        Core.subtract(errorCovPost, kskT, errorCovPost)
    }
}
