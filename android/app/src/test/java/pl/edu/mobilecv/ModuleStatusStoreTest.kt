package pl.edu.mobilecv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

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
}
