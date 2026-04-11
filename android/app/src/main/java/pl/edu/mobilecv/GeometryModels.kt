package pl.edu.mobilecv

import org.opencv.core.Point
import org.opencv.core.Rect

/**
 * Reprezentuje pojedynczy segment linii używany w analizie geometrii sceny.
 */
data class LineSegment(
    val p1: Point,
    val p2: Point,
    val length: Double,
    val angleRad: Double,
)

/**
 * Grupa linii o zbliżonym kierunku i położeniu przestrzennym.
 */
data class DirectionCluster(
    val lines: List<LineSegment>,
    val totalWeight: Double,
    val meanAngleDeg: Double,
    val boundingBox: Rect,
)

/**
 * Wynik etapu ekstrakcji linii i segmentacji kierunków.
 */
data class LineExtractionResult(
    val lines: List<LineSegment>,
    val clusters: List<DirectionCluster>,
)

/**
 * Kandydat punktu zbieżności wyestymowany z jednej grupy kierunkowej.
 */
data class VanishingPointCandidate(
    val point: Point,
    val cluster: DirectionCluster,
    val confidence: Double,
)

/**
 * Hipoteza płaszczyzny zbudowana na podstawie pary punktów zbieżności.
 */
data class PlaneHypothesis(
    val normalAngleDeg: Double,
    val centroid: Point,
    val lines: List<LineSegment>,
    val supportLineCount: Int,
    val confidence: Double,
    val vanishingPair: Pair<Point, Point>,
)

/**
 * Wynik śledzenia płaszczyzny między klatkami z metrykami stabilności.
 */
data class TrackedPlane(
    val planeId: Int,
    val displayIndex: Int,
    val smoothedNormalDeg: Double,
    val jitterDeg: Double,
    val stability: Double,
    val confidence: Double,
    val lines: List<LineSegment>,
    val centroid: Point,
)
