package pl.edu.mobilecv

import android.content.Intent
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import pl.edu.mobilecv.databinding.ActivityMenuBinding

/**
 * Launcher activity that presents a main menu with all available analysis modes.
 *
 * The user picks a mode and is taken to [MainActivity], which opens the camera
 * and pre-selects the chosen mode tab.  Separating the launcher from the camera
 * activity prevents crashes caused by eager camera / OpenCV initialization
 * before the user has made a selection.
 *
 * An additional card for the [PointCloudViewerActivity] is appended at the bottom
 * of the list to allow loading and visualizing previously saved point clouds.
 */
class MenuActivity : AppCompatActivity() {

    companion object {
        /** Intent extra key used to pass the selected [AnalysisMode] name to [MainActivity]. */
        const val EXTRA_MODE = "extra_mode"
    }

    private lateinit var binding: ActivityMenuBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMenuBinding.inflate(layoutInflater)
        setContentView(binding.root)
        buildModeCards()
        buildPointCloudViewerCard()
    }

    /** Inflates one card per [AnalysisMode] and appends it to the container. */
    private fun buildModeCards() {
        val descriptions = mapOf(
            AnalysisMode.FILTERS to getString(R.string.mode_desc_filters),
            AnalysisMode.EDGES to getString(R.string.mode_desc_edges),
            AnalysisMode.MORPHOLOGY to getString(R.string.mode_desc_morphology),
            AnalysisMode.MARKERS to getString(R.string.mode_desc_markers),
            AnalysisMode.POSE to getString(R.string.mode_desc_pose),
            AnalysisMode.ODOMETRY to getString(R.string.mode_desc_odometry),
            AnalysisMode.GEOMETRY to getString(R.string.mode_desc_geometry),
            AnalysisMode.CALIBRATION to getString(R.string.mode_desc_calibration),
        )

        AnalysisMode.entries.forEach { mode ->
            val card = layoutInflater.inflate(R.layout.item_mode_card, binding.modeListContainer, false)
            card.findViewById<TextView>(R.id.textModeName).text = mode.displayName
            card.findViewById<TextView>(R.id.textModeDescription).text = descriptions[mode] ?: ""
            card.setOnClickListener { launchMainActivity(mode) }
            binding.modeListContainer.addView(card)
        }
    }

    /** Appends a special card to open the point cloud viewer (not a camera mode). */
    private fun buildPointCloudViewerCard() {
        val card = layoutInflater.inflate(R.layout.item_mode_card, binding.modeListContainer, false)
        card.findViewById<TextView>(R.id.textModeName).text = getString(R.string.mode_point_cloud_viewer)
        card.findViewById<TextView>(R.id.textModeDescription).text = getString(R.string.mode_desc_point_cloud_viewer)
        card.setOnClickListener {
            startActivity(Intent(this, PointCloudViewerActivity::class.java))
        }
        binding.modeListContainer.addView(card)
    }

    /** Starts [MainActivity] with the chosen [mode] pre-selected. */
    private fun launchMainActivity(mode: AnalysisMode) {
        startActivity(
            Intent(this, MainActivity::class.java).apply {
                putExtra(EXTRA_MODE, mode.name)
            }
        )
    }
}
