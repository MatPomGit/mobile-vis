package pl.edu.mobilecv.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import pl.edu.mobilecv.OpenCvFilter

class FramePipelineTest {
    private val pipeline = FramePipeline()

    @Test
    fun `routes mediapipe filters and requests odometry reset`() {
        val decision = pipeline.decide(OpenCvFilter.HOLISTIC_BODY)

        assertEquals(FramePipeline.Route.MEDIAPIPE, decision.route)
        assertTrue(decision.resetOdometry)
    }

    @Test
    fun `routes yolo filters and requests odometry reset`() {
        val decision = pipeline.decide(OpenCvFilter.YOLO_DETECT)

        assertEquals(FramePipeline.Route.YOLO, decision.route)
        assertTrue(decision.resetOdometry)
    }

    @Test
    fun `keeps odometry state for odometry filters`() {
        val decision = pipeline.decide(OpenCvFilter.VISUAL_ODOMETRY)

        assertEquals(FramePipeline.Route.OPENCV, decision.route)
        assertEquals(false, decision.resetOdometry)
    }
}
