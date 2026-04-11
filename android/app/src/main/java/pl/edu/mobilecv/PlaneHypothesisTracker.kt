package pl.edu.mobilecv

import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Śledzi hipotezy płaszczyzn pomiędzy klatkami i wygładza ich orientację.
 */
class PlaneHypothesisTracker {
    companion object {
        private const val NORMAL_MATCH_THRESHOLD_DEG = 25.0
        private const val CENTER_MATCH_THRESHOLD_PX = 180.0
        private const val SMOOTHING_ALPHA = 0.7
        private const val TRACK_KEEP_MISSES = 6
        private const val HISTORY_SIZE = 12
    }

    private data class TrackState(
        val id: Int,
        var smoothedNormalDeg: Double,
        var centroid: Point,
        var stability: Double,
        var confidence: Double,
        var lines: List<LineSegment>,
        var misses: Int,
        val normalHistory: ArrayDeque<Double>,
    )

    private var nextId: Int = 1
    private val tracks = mutableListOf<TrackState>()

    /**
     * Aktualizuje stan trackera nowym zestawem hipotez płaszczyzn.
     */
    fun update(hypotheses: List<PlaneHypothesis>, maxPlanes: Int): List<TrackedPlane> {
        val remainingTrackIds = tracks.indices.toMutableSet()
        val matches = mutableListOf<Pair<Int, PlaneHypothesis>>()

        for (hypothesis in hypotheses.sortedByDescending { it.confidence }.take(maxPlanes * 2)) {
            val bestMatch = remainingTrackIds
                .map { index -> index to trackMatchCost(tracks[index], hypothesis) }
                .filter { (_, cost) -> cost != null }
                .minByOrNull { (_, cost) -> cost ?: Double.MAX_VALUE }

            if (bestMatch != null) {
                val (trackIndex, _) = bestMatch
                matches.add(trackIndex to hypothesis)
                remainingTrackIds.remove(trackIndex)
            } else {
                matches.add(-1 to hypothesis)
            }
        }

        val touchedTrackIds = mutableSetOf<Int>()

        for ((trackIndex, hypothesis) in matches) {
            if (trackIndex >= 0) {
                val track = tracks[trackIndex]
                track.smoothedNormalDeg = smoothAngle(track.smoothedNormalDeg, hypothesis.normalAngleDeg)
                track.centroid = blendPoint(track.centroid, hypothesis.centroid)
                track.confidence = 0.65 * track.confidence + 0.35 * hypothesis.confidence
                track.stability = (0.78 * track.stability + 0.22).coerceAtMost(1.0)
                track.lines = hypothesis.lines
                track.misses = 0
                appendNormal(track.normalHistory, track.smoothedNormalDeg)
                touchedTrackIds.add(track.id)
            } else {
                val normalHistory = ArrayDeque<Double>()
                appendNormal(normalHistory, hypothesis.normalAngleDeg)
                tracks.add(
                    TrackState(
                        id = nextId++,
                        smoothedNormalDeg = hypothesis.normalAngleDeg,
                        centroid = hypothesis.centroid,
                        stability = 0.35,
                        confidence = hypothesis.confidence,
                        lines = hypothesis.lines,
                        misses = 0,
                        normalHistory = normalHistory,
                    ),
                )
            }
        }

        for (track in tracks) {
            if (track.id !in touchedTrackIds) {
                track.misses += 1
                track.stability *= 0.85
            }
        }

        tracks.removeAll { it.misses > TRACK_KEEP_MISSES }

        return tracks
            .sortedByDescending { it.stability * it.confidence }
            .take(maxPlanes)
            .mapIndexed { index, track ->
                TrackedPlane(
                    planeId = track.id,
                    displayIndex = index,
                    smoothedNormalDeg = track.smoothedNormalDeg,
                    jitterDeg = computeJitter(track.normalHistory),
                    stability = track.stability.coerceIn(0.0, 1.0),
                    confidence = track.confidence.coerceIn(0.0, 1.0),
                    lines = track.lines,
                    centroid = track.centroid,
                )
            }
    }

    private fun trackMatchCost(track: TrackState, hypothesis: PlaneHypothesis): Double? {
        val angleDiff = circularDiff(track.smoothedNormalDeg, hypothesis.normalAngleDeg)
        val centerDistance = hypot(
            track.centroid.x - hypothesis.centroid.x,
            track.centroid.y - hypothesis.centroid.y,
        )

        if (angleDiff > NORMAL_MATCH_THRESHOLD_DEG || centerDistance > CENTER_MATCH_THRESHOLD_PX) {
            return null
        }

        return (angleDiff / NORMAL_MATCH_THRESHOLD_DEG) + (centerDistance / CENTER_MATCH_THRESHOLD_PX)
    }

    private fun smoothAngle(previous: Double, current: Double): Double {
        val delta = signedCircularDelta(previous, current)
        return normalizeAngle(previous * SMOOTHING_ALPHA + (previous + delta) * (1.0 - SMOOTHING_ALPHA))
    }

    private fun blendPoint(previous: Point, current: Point): Point {
        return Point(
            previous.x * SMOOTHING_ALPHA + current.x * (1.0 - SMOOTHING_ALPHA),
            previous.y * SMOOTHING_ALPHA + current.y * (1.0 - SMOOTHING_ALPHA),
        )
    }

    private fun appendNormal(history: ArrayDeque<Double>, normal: Double) {
        if (history.size >= HISTORY_SIZE) {
            history.removeFirst()
        }
        history.addLast(normalizeAngle(normal))
    }

    private fun computeJitter(history: ArrayDeque<Double>): Double {
        if (history.size < 2) return 0.0
        val values = history.toList()
        val mean = values.average()
        val variance = values.sumOf { (it - mean).pow(2.0) } / values.size
        return sqrt(variance)
    }

    private fun circularDiff(a: Double, b: Double): Double {
        val diff = abs(normalizeAngle(a) - normalizeAngle(b))
        return minOf(diff, 180.0 - diff)
    }

    private fun signedCircularDelta(from: Double, to: Double): Double {
        val delta = normalizeAngle(to) - normalizeAngle(from)
        return when {
            delta > 90.0 -> delta - 180.0
            delta < -90.0 -> delta + 180.0
            else -> delta
        }
    }

    private fun normalizeAngle(angle: Double): Double {
        var normalized = angle % 180.0
        if (normalized < 0.0) normalized += 180.0
        return normalized
    }
}
