package pl.edu.mobilecv.processing

import pl.edu.mobilecv.OpenCvFilter
import pl.edu.mobilecv.isMediaPipe
import pl.edu.mobilecv.isOdometryFilter
import pl.edu.mobilecv.isTflite
import pl.edu.mobilecv.isYolo

/**
 * Odpowiada za routing wejścia/wyjścia klatki do właściwego pipeline'u wykonania.
 */
class FramePipeline {
    /**
     * Typ pipeline'u wybranego dla pojedynczej klatki.
     */
    enum class Route {
        MEDIAPIPE,
        YOLO,
        TFLITE,
        OPENCV,
    }

    /**
     * Decyzja routingu klatki wraz z informacją o konieczności resetu odometrii.
     */
    data class Decision(
        val route: Route,
        val resetOdometry: Boolean,
    )

    /**
     * Zwraca kompletną decyzję dla aktualnego filtra przetwarzania.
     */
    fun decide(filter: OpenCvFilter): Decision {
        val route = when {
            filter.isMediaPipe -> Route.MEDIAPIPE
            filter.isYolo -> Route.YOLO
            filter.isTflite -> Route.TFLITE
            else -> Route.OPENCV
        }
        return Decision(route = route, resetOdometry = !filter.isOdometryFilter)
    }
}
