package pl.edu.mobilecv

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.tabs.TabLayout
import pl.edu.mobilecv.databinding.ActivityMenuBinding

/**
 * Launcher activity that presents a main menu with all available analysis modes.
 *
 * The menu is split into two tabs:
 * - **Tryby** – mode cards; tapping a card launches [MainActivity] with that mode.
 * - **O aplikacji** – general app description plus per-filter descriptions.
 *
 * Separating the launcher from the camera activity prevents crashes caused by eager
 * camera / OpenCV initialization before the user has made a selection.
 *
 * An additional card for the [PointCloudViewerActivity] is appended at the bottom of
 * the modes list to allow loading and visualizing previously saved point clouds.
 */
class MenuActivity : AppCompatActivity() {

    companion object {
        /** Intent extra key used to pass the selected [AnalysisMode] name to [MainActivity]. */
        const val EXTRA_MODE = "extra_mode"

        private const val TAG = "MenuActivity"
    }

    private lateinit var binding: ActivityMenuBinding

    /** Lazily built map of [AnalysisMode] to its Polish description string. */
    private val modeDescriptions: Map<AnalysisMode, String> by lazy {
        mapOf(
            AnalysisMode.FILTERS to getString(R.string.mode_desc_filters),
            AnalysisMode.EDGES to getString(R.string.mode_desc_edges),
            AnalysisMode.MORPHOLOGY to getString(R.string.mode_desc_morphology),
            AnalysisMode.MARKERS to getString(R.string.mode_desc_markers),
            AnalysisMode.POSE to getString(R.string.mode_desc_pose),
            AnalysisMode.ODOMETRY to getString(R.string.mode_desc_odometry),
            AnalysisMode.GEOMETRY to getString(R.string.mode_desc_geometry),
            AnalysisMode.CALIBRATION to getString(R.string.mode_desc_calibration),
            AnalysisMode.YOLO to getString(R.string.mode_desc_yolo),
        )
    }

    /** Lazily built map of [OpenCvFilter] to its Polish description string. */
    private val filterDescriptions: Map<OpenCvFilter, String> by lazy {
        mapOf(
            OpenCvFilter.ORIGINAL to getString(R.string.filter_desc_original),
            OpenCvFilter.GRAYSCALE to getString(R.string.filter_desc_grayscale),
            OpenCvFilter.GAUSSIAN_BLUR to getString(R.string.filter_desc_gaussian_blur),
            OpenCvFilter.MEDIAN_BLUR to getString(R.string.filter_desc_median_blur),
            OpenCvFilter.BILATERAL_FILTER to getString(R.string.filter_desc_bilateral),
            OpenCvFilter.BOX_FILTER to getString(R.string.filter_desc_box_filter),
            OpenCvFilter.THRESHOLD to getString(R.string.filter_desc_threshold),
            OpenCvFilter.ADAPTIVE_THRESHOLD to getString(R.string.filter_desc_adaptive_threshold),
            OpenCvFilter.HISTOGRAM_EQUALIZATION to getString(R.string.filter_desc_hist_eq),
            OpenCvFilter.CANNY_EDGES to getString(R.string.filter_desc_canny),
            OpenCvFilter.SOBEL to getString(R.string.filter_desc_sobel),
            OpenCvFilter.SCHARR to getString(R.string.filter_desc_scharr),
            OpenCvFilter.LAPLACIAN to getString(R.string.filter_desc_laplacian),
            OpenCvFilter.PREWITT to getString(R.string.filter_desc_prewitt),
            OpenCvFilter.ROBERTS to getString(R.string.filter_desc_roberts),
            OpenCvFilter.DILATE to getString(R.string.filter_desc_dilate),
            OpenCvFilter.ERODE to getString(R.string.filter_desc_erode),
            OpenCvFilter.OPEN to getString(R.string.filter_desc_open),
            OpenCvFilter.CLOSE to getString(R.string.filter_desc_close),
            OpenCvFilter.GRADIENT to getString(R.string.filter_desc_gradient),
            OpenCvFilter.TOP_HAT to getString(R.string.filter_desc_top_hat),
            OpenCvFilter.BLACK_HAT to getString(R.string.filter_desc_black_hat),
            OpenCvFilter.APRIL_TAGS to getString(R.string.filter_desc_apriltag),
            OpenCvFilter.ARUCO to getString(R.string.filter_desc_aruco),
            OpenCvFilter.QR_CODE to getString(R.string.filter_desc_qr),
            OpenCvFilter.CCTAG to getString(R.string.filter_desc_cctag),
            OpenCvFilter.HOLISTIC_BODY to getString(R.string.filter_desc_body),
            OpenCvFilter.HOLISTIC_HANDS to getString(R.string.filter_desc_hands),
            OpenCvFilter.HOLISTIC_FACE to getString(R.string.filter_desc_face),
            OpenCvFilter.IRIS to getString(R.string.filter_desc_iris),
            OpenCvFilter.VISUAL_ODOMETRY to getString(R.string.filter_desc_visual_odometry),
            OpenCvFilter.POINT_CLOUD to getString(R.string.filter_desc_point_cloud),
            OpenCvFilter.PLANE_DETECTION to getString(R.string.filter_desc_plane_detection),
            OpenCvFilter.VANISHING_POINTS to getString(R.string.filter_desc_vanishing_points),
            OpenCvFilter.CHESSBOARD_CALIBRATION to getString(R.string.filter_desc_chessboard),
            OpenCvFilter.UNDISTORT to getString(R.string.filter_desc_undistort),
            OpenCvFilter.YOLO_DETECT to getString(R.string.filter_desc_yolo_detect),
            OpenCvFilter.YOLO_SEGMENT to getString(R.string.filter_desc_yolo_segment),
            OpenCvFilter.YOLO_POSE to getString(R.string.filter_desc_yolo_pose),
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMenuBinding.inflate(layoutInflater)
        setContentView(binding.root)
        showAppVersion()
        buildInstructionsCard()
        buildOdometryTutorialCard()
        buildModeCards()
        buildPointCloudViewerCard()
        buildAboutContent()
        setupMenuTabs()
    }

    // ------------------------------------------------------------------
    // Tab setup
    // ------------------------------------------------------------------

    /** Configures the two-tab layout and switches the visible scroll view. */
    private fun setupMenuTabs() {
        binding.tabLayoutMenu.addTab(binding.tabLayoutMenu.newTab().setText(R.string.tab_modes))
        binding.tabLayoutMenu.addTab(binding.tabLayoutMenu.newTab().setText(R.string.tab_about))

        binding.tabLayoutMenu.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                val showAbout = tab.position == 1
                binding.scrollViewModes.visibility = if (showAbout) View.GONE else View.VISIBLE
                binding.scrollViewAbout.visibility = if (showAbout) View.VISIBLE else View.GONE
            }

            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) {}
        })
    }

    // ------------------------------------------------------------------
    // Modes tab content
    // ------------------------------------------------------------------

    /**
     * Inflates an instructions card and inserts it as the first item in the container.
     * Tapping the card opens a dialog with a step-by-step guide on how to use the app.
     */
    private fun buildInstructionsCard() {
        val card = layoutInflater.inflate(R.layout.item_mode_card, binding.modeListContainer, false)
        card.findViewById<TextView>(R.id.textModeName).text = getString(R.string.instructions_card_title)
        card.findViewById<TextView>(R.id.textModeDescription).text = getString(R.string.instructions_card_description)
        card.setOnClickListener { showInstructionsDialog() }
        binding.modeListContainer.addView(card)
    }

    /**
     * Inflates a card that opens the [OdometryTutorialActivity].
     * The card is placed in the modes list so the user can learn about visual odometry
     * before selecting the corresponding camera mode.
     */
    private fun buildOdometryTutorialCard() {
        val card = layoutInflater.inflate(R.layout.item_mode_card, binding.modeListContainer, false)
        card.findViewById<TextView>(R.id.textModeName).text = getString(R.string.tutorial_card_title)
        card.findViewById<TextView>(R.id.textModeDescription).text = getString(R.string.tutorial_card_description)
        card.setOnClickListener {
            startActivity(Intent(this, OdometryTutorialActivity::class.java))
        }
        binding.modeListContainer.addView(card)
    }

    /** Displays a scrollable dialog with the app's usage instructions. */
    private fun showInstructionsDialog() {
        AlertDialog.Builder(this)
            .setTitle(R.string.instructions_title)
            .setMessage(R.string.instructions_content)
            .setPositiveButton(R.string.instructions_close, null)
            .show()
    }

    /** Displays the current app version below the app name. */
    private fun showAppVersion() {
        val versionName = try {
            packageManager.getPackageInfo(packageName, 0).versionName ?: ""
        } catch (e: android.content.pm.PackageManager.NameNotFoundException) {
            android.util.Log.e(TAG, "Could not read app version", e)
            ""
        }
        binding.textAppVersion.text = getString(R.string.app_version_format, versionName)
    }

    /** Inflates one card per [AnalysisMode] and appends it to the container. */
    private fun buildModeCards() {
        AnalysisMode.entries.forEach { mode ->
            val card = layoutInflater.inflate(R.layout.item_mode_card, binding.modeListContainer, false)
            card.findViewById<TextView>(R.id.textModeName).text = mode.displayName
            card.findViewById<TextView>(R.id.textModeDescription).text = modeDescriptions[mode] ?: ""
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

    // ------------------------------------------------------------------
    // About tab content
    // ------------------------------------------------------------------

    /**
     * Populates the About tab with a general app description card followed by
     * one section per [AnalysisMode], each showing its mode description and
     * a list of individual filter descriptions.
     */
    private fun buildAboutContent() {
        addAppDescriptionCard()

        AnalysisMode.entries.forEach { mode ->
            addModeSectionCard(mode)
            mode.filters.forEach { filter ->
                addFilterDescriptionRow(filter.displayName, filterDescriptions[filter] ?: "")
            }
        }
    }

    /** Adds a card at the top of the About tab describing the application. */
    private fun addAppDescriptionCard() {
        val card = layoutInflater.inflate(R.layout.item_mode_card, binding.aboutContainer, false)
        card.findViewById<TextView>(R.id.textModeName).text = getString(R.string.app_name)
        card.findViewById<TextView>(R.id.textModeDescription).text =
            getString(R.string.about_app_description)
        card.isClickable = false
        binding.aboutContainer.addView(card)
    }

    /**
     * Adds a section-header card for [mode] in the About tab.
     * The card shows the mode name and its high-level description.
     */
    private fun addModeSectionCard(mode: AnalysisMode) {
        val card = layoutInflater.inflate(R.layout.item_mode_card, binding.aboutContainer, false)
        card.findViewById<TextView>(R.id.textModeName).text = mode.displayName
        card.findViewById<TextView>(R.id.textModeDescription).text = modeDescriptions[mode] ?: ""
        card.isClickable = false
        binding.aboutContainer.addView(card)
    }

    /**
     * Inflates a filter description row and appends it to the About container.
     *
     * @param name        Human-readable filter name shown as a bullet item.
     * @param description Polish description of what the filter does.
     */
    private fun addFilterDescriptionRow(name: String, description: String) {
        val row = layoutInflater.inflate(
            R.layout.item_filter_description, binding.aboutContainer, false
        )
        row.findViewById<TextView>(R.id.textFilterName).text = "• $name"
        row.findViewById<TextView>(R.id.textFilterDescription).text = description
        binding.aboutContainer.addView(row)
    }
}


