package pl.edu.mobilecv

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

/**
 * Moduł Active Tracking odpowiedzialny za cykl życia ROI:
 * detekcja celu, śledzenie, re-akwizycja i bezpieczny fallback do pełnej klatki.
 */
class ActiveVisionOptimizer {

    /** Konfiguracja pipeline aktywnego śledzenia przekazywana z UI. */
    data class TrackingConfig(
        val autoRoi: Boolean,
        val aggressiveness: Double,
        val targetLocked: Boolean,
        val fallbackToFullFrameAfter: Int,
        val visualizeOverlay: Boolean,
    )

    /** Status trackera raportowany do overlay. */
    enum class TrackerStatus {
        DETECTING,
        TRACKING,
        REACQUIRING,
        LOST,
        FALLBACK_FULL_FRAME,
    }

    /** Stan trackera aktualizowany per klatka i używany do diagnostyki UI. */
    data class TrackingState(
        val roi: Rect,
        val predictedRoi: Rect,
        val confidence: Double,
        val status: TrackerStatus,
        val lostFrames: Int,
    )

    /** Komplet danych wyjściowych pipeline: obraz i metryki. */
    data class TrackingOutput(
        val output: Mat,
        val state: TrackingState,
    )

    companion object {
        private const val GRID_ROWS = 3
        private const val GRID_COLS = 3
        private const val CONTEXT_EXPANSION = 0.15
        private const val SEARCH_EXPANSION = 0.55
        private const val MIN_TRACK_CONFIDENCE = 0.45
        private const val ALPHA_BASE = 0.20
        private const val ALPHA_RANGE = 0.70
    }

    private var previousRoi: Rect? = null
    private var previousRoiGray: Mat? = null
    private var previousVelocityX: Int = 0
    private var previousVelocityY: Int = 0
    private var lostFramesCount: Int = 0

    /**
     * Kompatybilna ścieżka dla legacy "Active Vision" (bez pełnego stanu UI trackera).
     */
    fun optimize(input: Mat, visualizeWork: Boolean = false): Mat {
        val output = process(
            input = input,
            config = TrackingConfig(
                autoRoi = true,
                aggressiveness = 0.55,
                targetLocked = false,
                fallbackToFullFrameAfter = 10,
                visualizeOverlay = visualizeWork,
            ),
        )
        return output.output
    }

    /**
     * Przetwarza klatkę i aktualizuje lifecycle ROI zgodnie z aktualnym stanem trackera.
     */
    fun process(input: Mat, config: TrackingConfig): TrackingOutput {
        val fullFrame = Rect(0, 0, input.cols(), input.rows())
        val currentGray = Mat()
        Imgproc.cvtColor(input, currentGray, Imgproc.COLOR_RGBA2GRAY)

        var status = TrackerStatus.DETECTING
        var confidence = 1.0

        val currentRoi = when {
            !config.autoRoi -> {
                status = TrackerStatus.TRACKING
                fullFrame
            }
            previousRoi == null -> {
                status = TrackerStatus.DETECTING
                detectRoi(input)
            }
            else -> {
                val tracked = trackRoi(currentGray, input)
                confidence = tracked.second
                val trackingFailed = confidence < MIN_TRACK_CONFIDENCE

                if (trackingFailed) {
                    lostFramesCount += 1
                    status = if (lostFramesCount >= config.fallbackToFullFrameAfter) {
                        TrackerStatus.FALLBACK_FULL_FRAME
                    } else {
                        TrackerStatus.REACQUIRING
                    }

                    if (status == TrackerStatus.FALLBACK_FULL_FRAME) {
                        previousRoi = fullFrame
                        fullFrame
                    } else {
                        detectRoi(input)
                    }
                } else {
                    lostFramesCount = 0
                    status = TrackerStatus.TRACKING
                    tracked.first
                }
            }
        }

        if (!config.targetLocked) {
            val detectedRoi = detectRoi(input)
            val blend = (ALPHA_BASE + config.aggressiveness * ALPHA_RANGE).coerceIn(0.15, 0.90)
            val blendedRoi = blendRoi(currentRoi, detectedRoi, blend)
            previousVelocityX = blendedRoi.x - (previousRoi?.x ?: blendedRoi.x)
            previousVelocityY = blendedRoi.y - (previousRoi?.y ?: blendedRoi.y)
            previousRoi = blendedRoi
        } else {
            previousVelocityX = currentRoi.x - (previousRoi?.x ?: currentRoi.x)
            previousVelocityY = currentRoi.y - (previousRoi?.y ?: currentRoi.y)
            previousRoi = currentRoi
        }

        previousRoiGray?.release()
        previousRoiGray = currentGray.submat(previousRoi)

        val predicted = predictRoi(previousRoi ?: fullFrame, input.cols(), input.rows())
        val outputFrame = input.clone()

        if (config.visualizeOverlay) {
            drawOverlay(outputFrame, previousRoi ?: fullFrame, predicted, confidence, status)
        }

        currentGray.release()

        return TrackingOutput(
            output = outputFrame,
            state = TrackingState(
                roi = previousRoi ?: fullFrame,
                predictedRoi = predicted,
                confidence = confidence,
                status = status,
                lostFrames = lostFramesCount,
            ),
        )
    }

    /** Czyści stan trackera przy zmianie trybu lub reset pipeline. */
    fun reset() {
        previousRoi = null
        previousRoiGray?.release()
        previousRoiGray = null
        previousVelocityX = 0
        previousVelocityY = 0
        lostFramesCount = 0
    }

    /** Wykrywa ROI o najwyższej informacji teksturalnej na bazie Laplasjanu. */
    private fun detectRoi(input: Mat): Rect {
        val gray = Mat()
        val laplacian = Mat()
        val absLaplace = Mat()

        Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Laplacian(gray, laplacian, CvType.CV_16S)
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

    /**
     * Śledzi poprzedni ROI przez template matching w lokalnym oknie przeszukiwania.
     * Zwraca nowy ROI oraz confidence dopasowania.
     */
    private fun trackRoi(currentGray: Mat, input: Mat): Pair<Rect, Double> {
        val prevRoi = previousRoi ?: return detectRoi(input) to 0.0
        val prevTemplate = previousRoiGray ?: return detectRoi(input) to 0.0
        val predicted = predictRoi(prevRoi, input.cols(), input.rows())
        val searchRect = expandRect(predicted, SEARCH_EXPANSION, input.cols(), input.rows())

        val searchRegion = currentGray.submat(searchRect)
        val resultCols = searchRegion.cols() - prevTemplate.cols() + 1
        val resultRows = searchRegion.rows() - prevTemplate.rows() + 1
        if (resultCols <= 0 || resultRows <= 0) {
            searchRegion.release()
            return detectRoi(input) to 0.0
        }

        val result = Mat(resultRows, resultCols, CvType.CV_32FC1)
        Imgproc.matchTemplate(searchRegion, prevTemplate, result, Imgproc.TM_CCOEFF_NORMED)

        val mmr = Core.minMaxLoc(result)
        val topLeft = Point(
            searchRect.x + mmr.maxLoc.x,
            searchRect.y + mmr.maxLoc.y,
        )

        val roi = clampRect(
            Rect(
                topLeft.x.toInt(),
                topLeft.y.toInt(),
                prevTemplate.cols(),
                prevTemplate.rows(),
            ),
            input.cols(),
            input.rows(),
        )

        val confidence = mmr.maxVal.coerceIn(0.0, 1.0)
        result.release()
        searchRegion.release()
        return roi to confidence
    }

    /** Oblicza ROI predykowane przez prosty model stałej prędkości 2D. */
    private fun predictRoi(roi: Rect, maxWidth: Int, maxHeight: Int): Rect = clampRect(
        Rect(
            roi.x + previousVelocityX,
            roi.y + previousVelocityY,
            roi.width,
            roi.height,
        ),
        maxWidth,
        maxHeight,
    )

    /** Rysuje komplet overlay: ROI, predicted ROI, confidence oraz status trackera. */
    private fun drawOverlay(frame: Mat, roi: Rect, predictedRoi: Rect, confidence: Double, status: TrackerStatus) {
        val roiColor = Scalar(70.0, 255.0, 90.0, 255.0)
        val predictedColor = Scalar(255.0, 200.0, 60.0, 255.0)
        val statusColor = if (status == TrackerStatus.TRACKING) {
            Scalar(70.0, 230.0, 70.0, 255.0)
        } else {
            Scalar(255.0, 120.0, 80.0, 255.0)
        }

        Imgproc.rectangle(frame, roi, roiColor, 3)
        Imgproc.rectangle(frame, predictedRoi, predictedColor, 2)

        Imgproc.putText(
            frame,
            "ROI",
            Point(roi.x.toDouble(), max(18.0, roi.y - 8.0).toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.5,
            roiColor,
            2,
        )

        Imgproc.putText(
            frame,
            "Predicted ROI",
            Point(predictedRoi.x.toDouble(), max(36.0, predictedRoi.y - 8.0).toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.5,
            predictedColor,
            2,
        )

        Imgproc.putText(
            frame,
            "Confidence: ${"%.2f".format(confidence)}",
            Point(16.0, 32.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.7,
            statusColor,
            2,
        )

        Imgproc.putText(
            frame,
            "Tracker: ${status.name}",
            Point(16.0, 62.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.7,
            statusColor,
            2,
        )
    }

    /** Łączy dwa ROI współczynnikiem alpha, aby wygładzić ruch trackera. */
    private fun blendRoi(base: Rect, update: Rect, alpha: Double): Rect {
        fun blendInt(a: Int, b: Int): Int = ((1.0 - alpha) * a + alpha * b).toInt()
        return Rect(
            blendInt(base.x, update.x),
            blendInt(base.y, update.y),
            max(1, blendInt(base.width, update.width)),
            max(1, blendInt(base.height, update.height)),
        )
    }

    /** Rozszerza ROI o kontekst sceny i ogranicza wynik do granic klatki. */
    private fun expandWithContext(rect: Rect, maxWidth: Int, maxHeight: Int): Rect {
        val marginX = (rect.width * CONTEXT_EXPANSION).toInt()
        val marginY = (rect.height * CONTEXT_EXPANSION).toInt()

        val x1 = max(0, rect.x - marginX)
        val y1 = max(0, rect.y - marginY)
        val x2 = min(maxWidth, rect.x + rect.width + marginX)
        val y2 = min(maxHeight, rect.y + rect.height + marginY)

        return Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    }

    /** Rozszerza prostokąt o zadany procent względem jego rozmiaru. */
    private fun expandRect(rect: Rect, ratio: Double, maxWidth: Int, maxHeight: Int): Rect {
        val marginX = (rect.width * ratio).toInt()
        val marginY = (rect.height * ratio).toInt()
        return clampRect(
            Rect(
                rect.x - marginX,
                rect.y - marginY,
                rect.width + (2 * marginX),
                rect.height + (2 * marginY),
            ),
            maxWidth,
            maxHeight,
        )
    }

    /** Koryguje prostokąt, aby zawsze mieścił się w obrazie i miał dodatnie wymiary. */
    private fun clampRect(rect: Rect, maxWidth: Int, maxHeight: Int): Rect {
        val x = rect.x.coerceIn(0, max(0, maxWidth - 1))
        val y = rect.y.coerceIn(0, max(0, maxHeight - 1))
        val width = rect.width.coerceIn(1, maxWidth - x)
        val height = rect.height.coerceIn(1, maxHeight - y)
        return Rect(x, y, width, height)
    }
}
