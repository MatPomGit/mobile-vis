package pl.edu.mobilecv.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class LandmarkTransformerTest {
    private val transformer = LandmarkTransformer()

    @Test
    fun `maps normalized landmarks to pixel bounding box`() {
        val result = transformer.toBoundingBox(
            landmarks = listOf(
                LandmarkDto(0.1f, 0.2f, 0.0f),
                LandmarkDto(0.3f, 0.8f, 0.0f),
            ),
            width = 200,
            height = 100,
        )

        requireNotNull(result)
        assertEquals(20.0, result.x1, 0.001)
        assertEquals(20.0, result.y1, 0.001)
        assertEquals(60.0, result.x2, 0.001)
        assertEquals(80.0, result.y2, 0.001)
    }

    @Test
    fun `returns null for empty landmarks`() {
        val result = transformer.toBoundingBox(emptyList(), width = 100, height = 100)

        assertNull(result)
    }
}
