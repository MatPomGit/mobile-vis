package pl.edu.mobilecv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import pl.edu.mobilecv.ui.ModuleStatusContracts

/**
 * Testy kontraktu centralnego store statusów modułów.
 */
class ModuleStatusStoreTest {

    /**
     * Potwierdza, że aktywne są tylko oczekiwane trzy grupy modułów.
     */
    @Test
    fun `module types should contain only mediapipe yolo and tflite`() {
        val moduleTypes = ModuleStatusStore.ModuleType.entries

        assertEquals(3, moduleTypes.size)
        assertTrue(moduleTypes.contains(ModuleStatusStore.ModuleType.MEDIAPIPE))
        assertTrue(moduleTypes.contains(ModuleStatusStore.ModuleType.YOLO))
        assertTrue(moduleTypes.contains(ModuleStatusStore.ModuleType.TFLITE))
        assertFalse(moduleTypes.any { it.name == "RTMDET" })
    }

    /**
     * Weryfikuje, że status ERROR przenosi identyfikator komunikatu i etykietę UI.
     */
    @Test
    fun `error status should keep message metadata`() {
        val moduleType = ModuleStatusStore.ModuleType.YOLO
        val status = ModuleStatusState.Error(errorMessageResId = 123)

        ModuleStatusStore.setStatus(moduleType, status)

        val snapshot = ModuleStatusStore.get(moduleType)
        val uiLabel = ModuleStatusContracts.statusLabel(snapshot.status)
        assertTrue(snapshot.status is ModuleStatusState.Error)
        assertEquals(123, (snapshot.status as ModuleStatusState.Error).errorMessageResId)
        assertEquals("ERROR", uiLabel)
        assertTrue(snapshot.lastInitAttemptEpochMs != null)
    }

    /**
     * Weryfikuje kontrakt routingu statusów obiektowych.
     */
    @Test
    fun `status transitions should support downloading ready and disabled`() {
        val moduleType = ModuleStatusStore.ModuleType.TFLITE

        ModuleStatusStore.setStatus(moduleType, ModuleStatusState.Downloading)
        assertTrue(ModuleStatusStore.get(moduleType).status is ModuleStatusState.Downloading)

        ModuleStatusStore.setStatus(moduleType, ModuleStatusState.Ready)
        assertTrue(ModuleStatusStore.get(moduleType).status is ModuleStatusState.Ready)

        ModuleStatusStore.setStatus(moduleType, ModuleStatusState.Disabled)
        assertTrue(ModuleStatusStore.get(moduleType).status is ModuleStatusState.Disabled)
    }
}
