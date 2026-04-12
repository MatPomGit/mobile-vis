package pl.edu.mobilecv

import java.io.File
import org.junit.Assert.assertEquals
import pl.edu.mobilecv.lifecycle.MobileModelManifest
import pl.edu.mobilecv.lifecycle.MobileModelManifestEntry
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MobileModelManifestTest {

    @Test
    fun `parser should load manifest entries and find model by name`() {
        val manifestJson = """
            {
              "models": [
                {
                  "model_name": "yolov8n.pt",
                  "format": "pt",
                  "input_shape": [1, 3, 640, 640],
                  "dtype": "float32",
                  "class_map": {"0": "person"},
                  "preprocess": {"normalize": true},
                  "postprocess": {
                    "output_tensors": [
                      {"name": "dets", "rank": 3, "last_dim": 5},
                      {"name": "labels", "rank": 2}
                    ]
                  },
                  "version": "1.2.3",
                  "sha256": "abc123"
                }
              ]
            }
        """.trimIndent()

        val entries = MobileModelManifest.parse(manifestJson)
        assertEquals(1, entries.size)

        val entry = MobileModelManifest.findByModelName(entries, "yolov8n.pt")
        assertNotNull(entry)
        assertEquals("pt", entry?.format)
        assertEquals(listOf(1, 3, 640, 640), entry?.inputShape)
    }

    @Test
    fun `compute sha256 should match known value`() {
        val tempFile = File.createTempFile("manifest-test", ".bin")
        tempFile.writeText("abc")
        val hash = MobileModelManifest.computeSha256(tempFile)
        assertEquals(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            hash,
        )
        tempFile.delete()
    }

    @Test
    fun `entry compatibility should validate extension and required metadata`() {
        val file = File.createTempFile("compatibility-test", ".pt")
        val validEntry = MobileModelManifestEntry(
            modelName = file.name,
            format = "pt",
            inputShape = listOf(1, 3, 640, 640),
            dtype = "float32",
            classMap = mapOf("0" to "person"),
            preprocess = org.json.JSONObject("""{"normalize":true}"""),
            postprocess = org.json.JSONObject("""{"output_rank":3}"""),
            version = "1.0.0",
            sha256 = "dummy",
        )
        assertTrue(MobileModelManifest.isEntryCompatible(validEntry, file))

        val invalidEntry = validEntry.copy(format = "onnx")
        assertFalse(MobileModelManifest.isEntryCompatible(invalidEntry, file))
        file.delete()
    }
}
