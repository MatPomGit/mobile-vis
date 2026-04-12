package pl.edu.mobilecv.processing

/**
 * Neutralny punkt landmarku niezależny od konkretnego frameworka CV.
 */
data class LandmarkDto(
    val x: Float,
    val y: Float,
    val z: Float,
)

/**
 * Neutralny wynik detekcji dłoni.
 */
data class HandDetectionsDto(
    val hands: List<List<LandmarkDto>>,
)

/**
 * Neutralny wynik detekcji pozy.
 */
data class PoseDetectionsDto(
    val poses: List<List<LandmarkDto>>,
)
