package pl.edu.mobilecv.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import pl.edu.mobilecv.AnalysisMode
import pl.edu.mobilecv.OpenCvFilter

class ModeRouterTest {
    private val router = ModeRouter()

    @Test
    fun `returns strategy for explicit mode`() {
        val strategy = router.strategyForMode(AnalysisMode.MARKERS)

        assertEquals(AnalysisMode.MARKERS, strategy.mode)
    }

    @Test
    fun `routes filter to matching mode strategy`() {
        val strategy = router.strategyForFilter(OpenCvFilter.HOLISTIC_FACE)

        assertNotNull(strategy)
        assertEquals(AnalysisMode.POSE, strategy?.mode)
    }
}
