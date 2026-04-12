package pl.edu.mobilecv.processing

/**
 * Prosty bounding box w pikselach wyprowadzony z landmarków.
 */
data class PixelBoundingBox(
    val x1: Double,
    val y1: Double,
    val x2: Double,
    val y2: Double,
)

/**
 * Transformacje neutralnych landmarków na struktury przydatne do dalszego przetwarzania.
 */
class LandmarkTransformer {
    /**
     * Konwertuje znormalizowane landmarki do bounding boxa w pikselach.
     */
    fun toBoundingBox(landmarks: List<LandmarkDto>, width: Int, height: Int): PixelBoundingBox? {
        if (landmarks.isEmpty()) return null
        var minX = Double.POSITIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY

        for (landmark in landmarks) {
            val px = (landmark.x * width).toDouble()
            val py = (landmark.y * height).toDouble()
            minX = minOf(minX, px)
            minY = minOf(minY, py)
            maxX = maxOf(maxX, px)
            maxY = maxOf(maxY, py)
        }

        if (!minX.isFinite() || !minY.isFinite() || !maxX.isFinite() || !maxY.isFinite()) return null
        return PixelBoundingBox(minX, minY, maxX, maxY)
    }
}
