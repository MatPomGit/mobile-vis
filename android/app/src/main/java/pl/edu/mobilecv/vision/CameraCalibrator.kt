package pl.edu.mobilecv.vision

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point3
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc

/**
 * Manages chessboard-based camera calibration state.
 *
 * Frame detection and collection run on the analysis thread; result reads
 * happen on the main thread.  All mutable fields are protected by [lock].
 *
 * Typical flow:
 * 1. For each live frame call [detectCorners].
 * 2. When the user presses "Collect", call [collectLastFrame].
 * 3. Once [frameCount] >= [MIN_FRAMES] call [calibrate].
 * 4. Retrieve [calibrationResult] to undistort images or display results.
 * 5. Call [reset] to start over.
 *
 * @param boardWidth  Number of inner corners along the horizontal axis (default 9).
 * @param boardHeight Number of inner corners along the vertical axis (default 6).
 */
class CameraCalibrator(
    val boardWidth: Int = DEFAULT_BOARD_WIDTH,
    val boardHeight: Int = DEFAULT_BOARD_HEIGHT,
) {

    companion object {
        private const val TAG = "CameraCalibrator"

        /** Default inner-corner count along the horizontal axis. */
        const val DEFAULT_BOARD_WIDTH = 9

        /** Default inner-corner count along the vertical axis. */
        const val DEFAULT_BOARD_HEIGHT = 6

        /** Minimum number of frames required to compute a calibration. */
        const val MIN_FRAMES = 10
    }

    // ------------------------------------------------------------------
    // Public state (read from any thread, writes protected by lock)
    // ------------------------------------------------------------------

    /** Calibration result computed by [calibrate], or ``null`` if not yet calibrated. */
    @Volatile
    var calibrationResult: CalibrationData? = null
        private set

    /**
     * ``true`` when a chessboard was detected in the most recent frame
     * processed by [detectCorners].
     */
    @Volatile
    var lastCornersDetected: Boolean = false
        private set

    // ------------------------------------------------------------------
    // Private state
    // ------------------------------------------------------------------

    private val lock = Object()

    /** Accumulated image-point arrays, one per collected frame. */
    private val imagePoints = mutableListOf<Mat>()

    /** Accumulated object-point arrays, one per collected frame. */
    private val objectPoints = mutableListOf<Mat>()

    /** Corners from the most recently processed frame (may be empty). */
    private val lastCornersMat: MatOfPoint2f by lazy { MatOfPoint2f() }

    /** Image size seen during the last [detectCorners] call. */
    private var lastImageSize = Size()

    /** Template object-point set (flat chessboard, Z=0). */
    private val objPtsTemplate: MatOfPoint3f by lazy {
        val pts = ArrayList<Point3>(boardWidth * boardHeight)
        for (r in 0 until boardHeight) {
            for (c in 0 until boardWidth) {
                pts.add(Point3(c.toDouble(), r.toDouble(), 0.0))
            }
        }
        MatOfPoint3f().also { it.fromList(pts) }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /** Number of frames that have been collected so far. */
    val frameCount: Int
        get() = synchronized(lock) { imagePoints.size }

    /**
     * Update the stored corner state from corners already detected externally.
     *
     * Avoids running a duplicate `findChessboardCorners` pass when the
     * detection was already performed by [ImageProcessor].
     *
     * **Must be called on the analysis (background) thread.**
     *
     * @param corners Detected corners, or ``null`` / empty if not found.
     * @param imageSize Size of the source frame (used later for calibration).
     */
    fun storeDetectedCorners(corners: MatOfPoint2f?, imageSize: Size) {
        synchronized(lock) {
            lastImageSize = imageSize
            if (corners != null && !corners.empty()) {
                corners.copyTo(lastCornersMat)
                lastCornersDetected = true
            } else {
                lastCornersDetected = false
            }
        }
    }

    /**
     * Detect inner corners of the chessboard in [gray].
     *
     * Results are stored internally and can be retrieved via [collectLastFrame].
     * Updates [lastCornersDetected].
     *
     * **Must be called on the analysis (background) thread.**
     *
     * @param gray Single-channel grayscale [Mat].
     * @param imageSize Size of the source frame (used for calibration).
     * @return Detected corners as [MatOfPoint2f], or ``null`` if not found.
     *         The caller must **not** release the returned [Mat] – it is owned
     *         by this class.
     */
    fun detectCorners(gray: Mat, imageSize: Size): MatOfPoint2f? {
        val patternSize = Size(boardWidth.toDouble(), boardHeight.toDouble())
        val corners = MatOfPoint2f()

        val found = Calib3d.findChessboardCorners(
            gray,
            patternSize,
            corners,
            Calib3d.CALIB_CB_ADAPTIVE_THRESH or Calib3d.CALIB_CB_NORMALIZE_IMAGE,
        )

        synchronized(lock) {
            lastImageSize = imageSize
            if (found && !corners.empty()) {
                val criteria = TermCriteria(
                    TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.001
                )
                Imgproc.cornerSubPix(
                    gray, corners,
                    Size(11.0, 11.0),
                    Size(-1.0, -1.0),
                    criteria,
                )
                corners.copyTo(lastCornersMat)
                lastCornersDetected = true
            } else {
                lastCornersDetected = false
            }
        }

        corners.release()
        return if (lastCornersDetected) lastCornersMat else null
    }

    /**
     * Add the corners from the most recently successful [detectCorners] call
     * to the calibration dataset.
     *
     * Thread-safe; may be called from any thread.
     *
     * @return ``true`` if corners were collected, ``false`` if no corners
     *         are available (board not visible).
     */
    fun collectLastFrame(): Boolean {
        synchronized(lock) {
            if (!lastCornersDetected || lastCornersMat.empty()) return false

            val imgPts = MatOfPoint2f()
            lastCornersMat.copyTo(imgPts)

            val objPts = MatOfPoint3f()
            objPtsTemplate.copyTo(objPts)

            imagePoints.add(imgPts)
            objectPoints.add(objPts)

            Log.d(TAG, "Collected frame ${imagePoints.size}/$MIN_FRAMES")
            return true
        }
    }

    /**
     * Compute intrinsic camera parameters from the collected frames.
     *
     * Thread-safe; may be called from any thread.
     *
     * @return [CalibrationData] on success, or ``null`` if fewer than
     *         [MIN_FRAMES] frames have been collected.
     */
    fun calibrate(): CalibrationData? {
        synchronized(lock) {
            if (imagePoints.size < MIN_FRAMES) {
                Log.w(TAG, "Not enough frames: ${imagePoints.size}/$MIN_FRAMES")
                return null
            }

            val camMat = Mat.eye(3, 3, CvType.CV_64F)
            val distCoeffs = MatOfDouble()
            val rvecs = ArrayList<Mat>()
            val tvecs = ArrayList<Mat>()

            return try {
                val rms = Calib3d.calibrateCamera(
                    objectPoints, imagePoints, lastImageSize,
                    camMat, distCoeffs, rvecs, tvecs,
                )
                val data = CalibrationData(
                    cameraMatrix = camMat,
                    distCoeffs = distCoeffs,
                    rmsError = rms,
                    frameCount = imagePoints.size,
                    calibrationImageSize = lastImageSize,
                )
                calibrationResult = data
                Log.i(TAG, "Calibration done: RMS=%.4f using %d frames".format(rms, data.frameCount))
                data
            } catch (e: Exception) {
                Log.e(TAG, "Calibration failed", e)
                null
            }
        }
    }

    /**
     * Resolve camera intrinsics for an active frame resolution.
     *
     * This is the single point of calibration-parameter retrieval for runtime
     * consumers such as visual odometry and plane detection.
     *
     * @param imageSize Active frame resolution.
     * @return [CalibrationProfile] with compatibility diagnostics, or ``null``
     *         when no calibration exists.
     */
    fun getCalibrationProfile(imageSize: Size): CalibrationProfile? {
        val current = calibrationResult ?: return null
        val calibrationSize = current.calibrationImageSize
        val isCompatible = calibrationSize.width.toInt() == imageSize.width.toInt() &&
            calibrationSize.height.toInt() == imageSize.height.toInt()
        return CalibrationProfile(
            calibration = current,
            activeImageSize = imageSize,
            isCompatible = isCompatible,
        )
    }

    /**
     * Clear all collected frames and calibration data.
     *
     * Thread-safe; may be called from any thread.
     */
    fun reset() {
        synchronized(lock) {
            imagePoints.forEach { it.release() }
            objectPoints.forEach { it.release() }
            imagePoints.clear()
            objectPoints.clear()
            lastCornersDetected = false
            calibrationResult = null
            Log.d(TAG, "Calibration state reset")
        }
    }

    // ------------------------------------------------------------------
    // Data classes
    // ------------------------------------------------------------------

    /**
     * Computed camera calibration parameters.
     *
     * @property cameraMatrix   3x3 intrinsic camera matrix.
     * @property distCoeffs     Distortion coefficients.
     * @property rmsError       Root-mean-square reprojection error in pixels.
     * @property frameCount     Number of frames used for calibration.
     * @property calibrationImageSize Resolution used during calibration.
     */
    data class CalibrationData(
        val cameraMatrix: Mat,
        val distCoeffs: MatOfDouble,
        val rmsError: Double,
        val frameCount: Int,
        val calibrationImageSize: Size,
    ) {
        /** Human-readable summary of the focal lengths and principal point. */
        fun summary(): String {
            val fx = cameraMatrix.get(0, 0)[0]
            val fy = cameraMatrix.get(1, 1)[0]
            val cx = cameraMatrix.get(0, 2)[0]
            val cy = cameraMatrix.get(1, 2)[0]
            return "fx=%.1f, fy=%.1f\ncx=%.1f, cy=%.1f\nRMS=%.4f".format(fx, fy, cx, cy, rmsError)
        }
    }

    /**
     * Runtime calibration profile for the currently active frame size.
     *
     * @property calibration Persisted intrinsic and distortion parameters.
     * @property activeImageSize Active stream resolution.
     * @property isCompatible ``true`` when active and calibration resolution match.
     */
    data class CalibrationProfile(
        val calibration: CalibrationData,
        val activeImageSize: Size,
        val isCompatible: Boolean,
    )
}
