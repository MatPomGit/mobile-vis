package pl.edu.mobilecv.util

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.video.KalmanFilter as CvKalmanFilter

/**
 * A wrapper for OpenCV's Kalman Filter to track bounding boxes (x, y, w, h).
 *
 * State vector (8x1): [x, y, w, h, dx, dy, dw, dh]
 * Measurement vector (4x1): [x, y, w, h]
 */
class BBoxKalmanFilter {
    private val kf = CvKalmanFilter(8, 4, 0, CvType.CV_32F)
    private var isInitialized = false

    init {
        // Transition matrix (F)
        val transitionMatrix = Mat.eye(8, 8, CvType.CV_32F)
        val dt = 1.0
        transitionMatrix.put(0, 4, dt)
        transitionMatrix.put(1, 5, dt)
        transitionMatrix.put(2, 6, dt)
        transitionMatrix.put(3, 7, dt)
        kf._transitionMatrix = transitionMatrix

        // Measurement matrix (H)
        val measurementMatrix = Mat.zeros(4, 8, CvType.CV_32F)
        measurementMatrix.put(0, 0, 1.0)
        measurementMatrix.put(1, 1, 1.0)
        measurementMatrix.put(2, 2, 1.0)
        measurementMatrix.put(3, 3, 1.0)
        kf._measurementMatrix = measurementMatrix

        // Process noise covariance (Q)
        val processNoise = Mat.eye(8, 8, CvType.CV_32F)
        processNoise.put(0, 0, 1e-2)
        processNoise.put(1, 1, 1e-2)
        processNoise.put(2, 2, 1e-2)
        processNoise.put(3, 3, 1e-2)
        processNoise.put(4, 4, 1e-1)
        processNoise.put(5, 5, 1e-1)
        processNoise.put(6, 6, 1e-1)
        processNoise.put(7, 7, 1e-1)
        kf._processNoiseCov = processNoise

        // Measurement noise covariance (R)
        val measurementNoise = Mat.eye(4, 4, CvType.CV_32F)
        measurementNoise.put(0, 0, 1e-1)
        measurementNoise.put(1, 1, 1e-1)
        measurementNoise.put(2, 2, 1e-1)
        measurementNoise.put(3, 3, 1e-1)
        kf._measurementNoiseCov = measurementNoise

        // Posterior error covariance (P)
        val errorCovPost = Mat.eye(8, 8, CvType.CV_32F)
        errorCovPost.put(0, 0, 1.0)
        errorCovPost.put(1, 1, 1.0)
        errorCovPost.put(2, 2, 1.0)
        errorCovPost.put(3, 3, 1.0)
        errorCovPost.put(4, 4, 1.0)
        errorCovPost.put(5, 5, 1.0)
        errorCovPost.put(6, 6, 1.0)
        errorCovPost.put(7, 7, 1.0)
        kf._errorCovPost = errorCovPost
    }

    fun predict(): FloatArray {
        val prediction = kf.predict()
        val result = FloatArray(4)
        result[0] = prediction.get(0, 0)[0].toFloat()
        result[1] = prediction.get(1, 0)[0].toFloat()
        result[2] = prediction.get(2, 0)[0].toFloat()
        result[3] = prediction.get(3, 0)[0].toFloat()
        return result
    }

    fun update(x: Float, y: Float, w: Float, h: Float): FloatArray {
        if (!isInitialized) {
            val state = Mat.zeros(8, 1, CvType.CV_32F)
            state.put(0, 0, x.toDouble())
            state.put(1, 0, y.toDouble())
            state.put(2, 0, w.toDouble())
            state.put(3, 0, h.toDouble())
            kf._statePost = state
            isInitialized = true
            return floatArrayOf(x, y, w, h)
        }

        val measurement = Mat(4, 1, CvType.CV_32F)
        measurement.put(0, 0, x.toDouble())
        measurement.put(1, 0, y.toDouble())
        measurement.put(2, 0, w.toDouble())
        measurement.put(3, 0, h.toDouble())

        val corrected = kf.correct(measurement)
        val result = FloatArray(4)
        result[0] = corrected.get(0, 0)[0].toFloat()
        result[1] = corrected.get(1, 0)[0].toFloat()
        result[2] = corrected.get(2, 0)[0].toFloat()
        result[3] = corrected.get(3, 0)[0].toFloat()
        return result
    }
}
