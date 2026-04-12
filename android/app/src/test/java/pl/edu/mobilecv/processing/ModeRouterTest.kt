package pl.edu.mobilecv.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import pl.edu.mobilecv.AnalysisMode
import pl.edu.mobilecv.ModuleStatusStore
import pl.edu.mobilecv.OpenCvFilter
import pl.edu.mobilecv.ui.ModuleStatusContracts

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

    @Test
    fun `returns null for unsupported filter`() {
        val strategy = router.strategyForFilter(OpenCvFilter.ORIGINAL)

        assertNull(strategy)
    }

    @Test
    fun `routes ml dependent modes to module types`() {
        assertEquals(
            ModuleStatusStore.ModuleType.MEDIAPIPE,
            ModuleStatusContracts.moduleTypeForMode(AnalysisMode.POSE),
        )
        assertEquals(
            ModuleStatusStore.ModuleType.YOLO,
            ModuleStatusContracts.moduleTypeForMode(AnalysisMode.YOLO),
        )
        assertNull(ModuleStatusContracts.moduleTypeForMode(AnalysisMode.EDGES))
    }
}
