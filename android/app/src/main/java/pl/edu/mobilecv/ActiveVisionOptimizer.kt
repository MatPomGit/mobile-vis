package pl.edu.mobilecv

import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

/**
 * Active Vision optimizer that dynamically selects and enhances a region of interest (ROI).
 *
 * The module estimates saliency from image gradients, smooths ROI updates between frames,
 * and emphasizes the selected area while attenuating the background.
 */
class ActiveVisionOptimizer {

    companion object {
        private const val GRID_ROWS = 3
        private const val GRID_COLS = 3
        private const val ROI_SMOOTHING_FACTOR = 0.3
        private const val CONTEXT_EXPANSION = 0.15
    }

    private var previousRoi: Rect? = null

    /**
     * Optimize the analysis area and return an RGBA frame with visual focus on ROI.
     *
     * @param visualizeWork If true, draws additional overlays that explain how the module
     *   selected and stabilized the current ROI in real time.
     */
    fun optimize(input: Mat, visualizeWork: Boolean = false): Mat {
        val roi = smoothRoi(findMostInformativeRegion(input))

        val emphasized = Mat()
        Core.addWeighted(input, 1.2, input, 0.0, 8.0, emphasized)

        val output = input.clone()
        Core.addWeighted(output, 0.45, emphasized, 0.0, 0.0, output)

        val focusedRegion = emphasized.submat(roi)
        val destinationRegion = output.submat(roi)
        focusedRegion.copyTo(destinationRegion)

        if (visualizeWork) {
            Imgproc.rectangle(output, roi, Scalar(255.0, 180.0, 0.0, 255.0), 3)
            Imgproc.putText(
                output,
                "Active Vision ROI",
                org.opencv.core.Point(roi.x.toDouble(), max(25.0, (roi.y - 10).toDouble())),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar(255.0, 180.0, 0.0, 255.0),
                2,
            )
        }

        focusedRegion.release()
        destinationRegion.release()
        emphasized.release()

        return output
    }

    private fun findMostInformativeRegion(input: Mat): Rect {
        val gray = Mat()
        val laplacian = Mat()
        val absLaplace = Mat()

        Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Laplacian(gray, laplacian, org.opencv.core.CvType.CV_16S)
        Core.convertScaleAbs(laplacian, absLaplace)

        val cellWidth = max(1, input.cols() / GRID_COLS)
        val cellHeight = max(1, input.rows() / GRID_ROWS)

        var bestRect = Rect(0, 0, cellWidth, cellHeight)
        var bestScore = -1.0

        for (row in 0 until GRID_ROWS) {
            for (col in 0 until GRID_COLS) {
                val x = col * cellWidth
                val y = row * cellHeight
                val width = if (col == GRID_COLS - 1) input.cols() - x else cellWidth
                val height = if (row == GRID_ROWS - 1) input.rows() - y else cellHeight
                val cellRect = Rect(x, y, width, height)
                val cell = absLaplace.submat(cellRect)
                val score = Core.mean(cell).`val`[0]
                if (score > bestScore) {
                    bestScore = score
                    bestRect = cellRect
                }
                cell.release()
            }
        }

        gray.release()
        laplacian.release()
        absLaplace.release()

        return expandWithContext(bestRect, input.cols(), input.rows())
    }

    private fun smoothRoi(newRoi: Rect): Rect {
        val prev = previousRoi
        if (prev == null) {
            previousRoi = newRoi
            return newRoi
        }

        val smoothed = Rect(
            weightedInt(prev.x, newRoi.x),
            weightedInt(prev.y, newRoi.y),
            weightedInt(prev.width, newRoi.width),
            weightedInt(prev.height, newRoi.height),
        )
        previousRoi = smoothed
        return smoothed
    }

    private fun weightedInt(previous: Int, current: Int): Int {
        val blended = previous * (1.0 - ROI_SMOOTHING_FACTOR) + current * ROI_SMOOTHING_FACTOR
        return blended.toInt()
    }

    private fun expandWithContext(rect: Rect, maxWidth: Int, maxHeight: Int): Rect {
        val marginX = (rect.width * CONTEXT_EXPANSION).toInt()
        val marginY = (rect.height * CONTEXT_EXPANSION).toInt()

        val x1 = max(0, rect.x - marginX)
        val y1 = max(0, rect.y - marginY)
        val x2 = min(maxWidth, rect.x + rect.width + marginX)
        val y2 = min(maxHeight, rect.y + rect.height + marginY)

        return Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    }
}
