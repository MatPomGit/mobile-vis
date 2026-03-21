package pl.edu.mobilecv

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Applies OpenCV image-processing filters to Android [Bitmap] frames.
 *
 * All input bitmaps must use [Bitmap.Config.ARGB_8888].  Internally the
 * bitmap is converted to a **BGRA** [Mat] (OpenCV's representation of
 * ARGB_8888), the chosen [OpenCvFilter] is applied, and the result is
 * converted back to an ARGB_8888 bitmap suitable for display.
 *
 * This class is **not thread-safe**; create one instance per thread or
 * synchronise access externally.
 */
class ImageProcessor {

    /**
     * Process a single [Bitmap] frame with the given [filter].
     *
     * @param bitmap ARGB_8888 bitmap to process.
     * @param filter OpenCV filter to apply.
     * @return New ARGB_8888 bitmap with the filter applied.
     */
    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        val src = Mat()
        // bitmapToMat converts ARGB_8888 → BGRA Mat (4 channels)
        Utils.bitmapToMat(bitmap, src)

        val processed: Mat = when (filter) {
            OpenCvFilter.ORIGINAL -> src.clone()
            OpenCvFilter.GRAYSCALE -> applyGrayscale(src)
            OpenCvFilter.CANNY_EDGES -> applyCanny(src)
            OpenCvFilter.GAUSSIAN_BLUR -> applyGaussianBlur(src)
            OpenCvFilter.THRESHOLD -> applyThreshold(src)
            OpenCvFilter.SOBEL -> applySobel(src)
            OpenCvFilter.LAPLACIAN -> applyLaplacian(src)
            OpenCvFilter.DILATE -> applyDilate(src)
            OpenCvFilter.ERODE -> applyErode(src)
        }

        val result = Bitmap.createBitmap(processed.cols(), processed.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(processed, result)

        src.release()
        processed.release()

        return result
    }

    // ------------------------------------------------------------------
    // Private filter implementations
    // ------------------------------------------------------------------

    /**
     * Convert the frame to grayscale and back to BGRA for display.
     *
     * Input/output: BGRA Mat (shape H × W × 4).
     */
    private fun applyGrayscale(src: Mat): Mat {
        val gray = Mat()
        val result = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.cvtColor(gray, result, Imgproc.COLOR_GRAY2BGRA)
        gray.release()
        return result
    }

    /**
     * Detect edges using the Canny algorithm.
     *
     * Pre-blurs with a 5×5 Gaussian kernel to reduce noise.
     * Thresholds: low = 50, high = 150.
     */
    private fun applyCanny(src: Mat): Mat {
        val gray = Mat()
        val blurred = Mat()
        val edges = Mat()
        val result = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)
        Imgproc.cvtColor(edges, result, Imgproc.COLOR_GRAY2BGRA)
        gray.release()
        blurred.release()
        edges.release()
        return result
    }

    /**
     * Apply a 15×15 Gaussian blur to soften the image.
     */
    private fun applyGaussianBlur(src: Mat): Mat {
        val result = Mat()
        Imgproc.GaussianBlur(src, result, Size(15.0, 15.0), 0.0)
        return result
    }

    /**
     * Apply binary threshold at pixel value 127 (range [0, 255]).
     */
    private fun applyThreshold(src: Mat): Mat {
        val gray = Mat()
        val thresh = Mat()
        val result = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.threshold(gray, thresh, 127.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.cvtColor(thresh, result, Imgproc.COLOR_GRAY2BGRA)
        gray.release()
        thresh.release()
        return result
    }

    /**
     * Compute the gradient magnitude via combined Sobel X and Y operators.
     *
     * Each derivative is computed at [CvType.CV_16S] depth to avoid
     * overflow, then scaled back to 8-bit with [Core.convertScaleAbs].
     * The two gradients are averaged with [Core.addWeighted].
     */
    private fun applySobel(src: Mat): Mat {
        val gray = Mat()
        val sobelX = Mat()
        val sobelY = Mat()
        val absX = Mat()
        val absY = Mat()
        val combined = Mat()
        val result = Mat()

        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sobelY, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sobelX, absX)
        Core.convertScaleAbs(sobelY, absY)
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, result, Imgproc.COLOR_GRAY2BGRA)

        gray.release()
        sobelX.release()
        sobelY.release()
        absX.release()
        absY.release()
        combined.release()
        return result
    }

    /**
     * Compute second-order derivative edges with the Laplacian operator.
     *
     * Computed at [CvType.CV_16S] depth then scaled back to 8-bit.
     */
    private fun applyLaplacian(src: Mat): Mat {
        val gray = Mat()
        val laplacian = Mat()
        val abs = Mat()
        val result = Mat()

        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.Laplacian(gray, laplacian, CvType.CV_16S)
        Core.convertScaleAbs(laplacian, abs)
        Imgproc.cvtColor(abs, result, Imgproc.COLOR_GRAY2BGRA)

        gray.release()
        laplacian.release()
        abs.release()
        return result
    }

    /**
     * Apply morphological dilation with a 9×9 rectangular structuring element.
     *
     * Brightens bright regions, useful for closing small dark holes.
     */
    private fun applyDilate(src: Mat): Mat {
        val result = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
        Imgproc.dilate(src, result, kernel)
        kernel.release()
        return result
    }

    /**
     * Apply morphological erosion with a 9×9 rectangular structuring element.
     *
     * Darkens dark regions, useful for removing small bright specks.
     */
    private fun applyErode(src: Mat): Mat {
        val result = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
        Imgproc.erode(src, result, kernel)
        kernel.release()
        return result
    }
}
