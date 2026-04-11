package pl.edu.mobilecv.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import pl.edu.mobilecv.AnalysisMode

/**
 * Testy rejestru trybów używanego przez menu.
 */
class ModeRegistryTest {

    /**
     * Weryfikuje, że zakładka detekcji zawiera dokładnie trzy tryby.
     */
    @Test
    fun `detection group should expose exactly three modes`() {
        val detectionModes = ModeRegistry.entriesForGroup(ModeRegistry.FunctionalGroup.DETECTION)
            .map { it.mode }

        assertEquals(3, detectionModes.size)
        assertEquals(
            listOf(AnalysisMode.MARKERS, AnalysisMode.POSE, AnalysisMode.YOLO),
            detectionModes,
        )
    }

    /**
     * Weryfikuje, że usunięty tryb ACTIVE_TRACKING nie jest już dostępny.
     */
    @Test
    fun `analysis mode entries should not include active tracking`() {
        assertFalse(AnalysisMode.entries.any { it.name == "ACTIVE_TRACKING" })
        assertTrue(AnalysisMode.entries.contains(AnalysisMode.MARKERS))
        assertTrue(AnalysisMode.entries.contains(AnalysisMode.POSE))
        assertTrue(AnalysisMode.entries.contains(AnalysisMode.YOLO))
    }
}
