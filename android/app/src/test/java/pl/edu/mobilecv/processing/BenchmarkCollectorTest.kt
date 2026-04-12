package pl.edu.mobilecv.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import pl.edu.mobilecv.OpenCvFilter

class BenchmarkCollectorTest {
    @Test
    fun `returns snapshot only after reaching sample limit`() {
        val collector = BenchmarkCollector(setOf(OpenCvFilter.ORIGINAL), sampleLimit = 2)

        collector.addSample(OpenCvFilter.ORIGINAL, beforeNs = 1_000_000, afterNs = 2_000_000)
        assertNull(collector.consumeSnapshot(OpenCvFilter.ORIGINAL))

        collector.addSample(OpenCvFilter.ORIGINAL, beforeNs = 3_000_000, afterNs = 4_000_000)
        val snapshot = collector.consumeSnapshot(OpenCvFilter.ORIGINAL)

        assertNotNull(snapshot)
        assertEquals(2, snapshot?.samples)
        assertEquals(2.0, snapshot?.avgBeforeMs)
        assertEquals(3.0, snapshot?.avgAfterMs)
    }
}
