package pl.edu.mobilecv.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import pl.edu.mobilecv.AnalysisMode
import pl.edu.mobilecv.ModuleStatusState
import pl.edu.mobilecv.ModuleStatusStore

/**
 * Testy rejestru trybów używanego przez menu.
 */
class ModeRegistryTest {

    /**
     * Weryfikuje, że zakładka detekcji zawiera komplet czterech trybów.
     */
    @Test
    fun `detection group should expose exactly four modes`() {
        val detectionModes = ModeRegistry.entriesForGroup(ModeRegistry.FunctionalGroup.DETECTION)
            .map { it.mode }

        assertEquals(4, detectionModes.size)
        assertEquals(
            listOf(AnalysisMode.MARKERS, AnalysisMode.POSE, AnalysisMode.YOLO, AnalysisMode.ACTIVE_TRACKING),
            detectionModes,
        )
    }

    /**
     * Weryfikuje, że tryb ACTIVE_TRACKING jest dostępny i poprawnie zarejestrowany.
     */
    @Test
    fun `analysis mode entries should include active tracking`() {
        assertTrue(AnalysisMode.entries.any { it.name == "ACTIVE_TRACKING" })
        assertFalse(AnalysisMode.entries.isEmpty())
        assertTrue(AnalysisMode.entries.contains(AnalysisMode.MARKERS))
        assertTrue(AnalysisMode.entries.contains(AnalysisMode.POSE))
        assertTrue(AnalysisMode.entries.contains(AnalysisMode.YOLO))
        assertTrue(AnalysisMode.entries.contains(AnalysisMode.ACTIVE_TRACKING))
    }

    /**
     * Weryfikuje, że nowy tryb SLAM jest widoczny w grupie ANALYSIS.
     */
    @Test
    fun `analysis group should include slam mode`() {
        val analysisModes = ModeRegistry.entriesForGroup(ModeRegistry.FunctionalGroup.ANALYSIS)
            .map { it.mode }
        assertTrue(analysisModes.contains(AnalysisMode.SLAM))
    }

    /**
     * Sprawdza spójność routingu trybu z kontraktem statusu modułów.
     */
    @Test
    fun `module status contract should expose labels and presentation`() {
        val snapshot = ModuleStatusStore.ModuleSnapshot(status = ModuleStatusState.Ready)
        val presentation = ModuleStatusContracts.toPresentation(
            ModuleStatusStore.ModuleType.MEDIAPIPE,
            snapshot,
        )

        assertEquals("READY", presentation.statusLabel)
        assertEquals(ModuleStatusStore.ModuleType.MEDIAPIPE, presentation.moduleType)
        assertNotNull(ModuleStatusContracts.moduleTypeForMode(AnalysisMode.POSE))
    }
}
