package pl.edu.mobilecv

import android.os.Bundle
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import pl.edu.mobilecv.ui.ModuleStatusContracts
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Lekki ekran diagnostyczny prezentujący stan modułów z timestampem ostatniej próby inicjalizacji.
 */
class ModuleStatusActivity : AppCompatActivity() {

    private val timestampFormatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_module_status)
        title = getString(R.string.modules_diagnostics_title)
        renderModuleRows()
    }

    override fun onResume() {
        super.onResume()
        renderModuleRows()
    }

    /**
     * Renderuje listę modułów i ich statusów na podstawie centralnego store.
     */
    private fun renderModuleRows() {
        val container = findViewById<LinearLayout>(R.id.moduleStatusContainer)
        container.removeAllViews()

        ModuleStatusStore.snapshot().forEach { (module, snapshot) ->
            val presentation = ModuleStatusContracts.toPresentation(module, snapshot)
            val row = TextView(this).apply {
                text = getString(
                    R.string.modules_diagnostics_row,
                    presentation.moduleType.name,
                    presentation.statusLabel,
                    formatTimestamp(presentation.lastInitAttemptEpochMs),
                )
            }
            container.addView(row)
        }
    }

    private fun formatTimestamp(epochMs: Long?): String {
        if (epochMs == null) return getString(R.string.modules_diagnostics_no_attempt)
        return timestampFormatter.format(Date(epochMs))
    }
}
