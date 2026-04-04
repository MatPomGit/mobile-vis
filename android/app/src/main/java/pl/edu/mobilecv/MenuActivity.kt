package pl.edu.mobilecv

import android.content.Intent
import android.content.res.ColorStateList
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.tabs.TabLayout
import pl.edu.mobilecv.databinding.ActivityMenuBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Launcher activity that presents a main menu with all available analysis modes.
 *
 * The menu is split into five tabs:
 * - **Przetwarzanie** – basic preprocessing modes (FILTERS / EDGES / MORPHOLOGY / EFFECTS).
 * - **Detekcja**      – detection modes (MARKERS / YOLO / POSE).
 * - **Analiza 3D**    – spatial-analysis modes (ODOMETRY / GEOMETRY / CALIBRATION).
 * - **Modele**        – model download status and manual-download buttons.
 * - **O aplikacji**   – general app description plus per-filter descriptions.
 *
 * Separating the launcher from the camera activity prevents crashes caused by eager
 * camera / OpenCV initialization before the user has made a selection.
 */
class MenuActivity : AppCompatActivity() {

    companion object {
        /** Intent extra key used to pass the selected [AnalysisMode] name to [MainActivity]. */
        const val EXTRA_MODE = "extra_mode"

        private const val TAG = "MenuActivity"

        // Tab positions in the TabLayout
        private const val TAB_PROCESSING = 0
        private const val TAB_DETECTION  = 1
        private const val TAB_ANALYSIS   = 2
        private const val TAB_MODELS     = 3
        private const val TAB_ABOUT      = 4

        /** Multiplier applied to [R.dimen.group_stroke_width] to derive button margins. */
        private const val BUTTON_MARGIN_MULTIPLIER = 8
    }

    private lateinit var binding: ActivityMenuBinding

    /** Single-thread executor used for background model downloads. Lazily created on first use. */
    private val downloadExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }

    /** Holds references to the UI elements of each model status row, plus the model group flag. */
    private data class ModelStatusViews(
        val dotView: View,
        val textModelSize: TextView,
        val btnAction: MaterialButton,
        val isYolo: Boolean,
    )

    /** Keyed by model filename (e.g. [MediaPipeProcessor.MODEL_POSE]). */
    private val modelStatusViews = mutableMapOf<String, ModelStatusViews>()

    /** Lazily built map of [AnalysisMode] to its Polish description string. */
    private val modeDescriptions: Map<AnalysisMode, String> by lazy {
        mapOf(
            AnalysisMode.FILTERS    to getString(R.string.mode_desc_filters),
            AnalysisMode.EDGES      to getString(R.string.mode_desc_edges),
            AnalysisMode.MORPHOLOGY to getString(R.string.mode_desc_morphology),
            AnalysisMode.MARKERS    to getString(R.string.mode_desc_markers),
            AnalysisMode.POSE       to getString(R.string.mode_desc_pose),
            AnalysisMode.ODOMETRY   to getString(R.string.mode_desc_odometry),
            AnalysisMode.GEOMETRY   to getString(R.string.mode_desc_geometry),
            AnalysisMode.CALIBRATION to getString(R.string.mode_desc_calibration),
            AnalysisMode.YOLO       to getString(R.string.mode_desc_yolo),
            AnalysisMode.EFFECTS    to getString(R.string.mode_desc_effects),
        )
    }

    /** Lazily built map of [OpenCvFilter] to its Polish description string. */
    private val filterDescriptions: Map<OpenCvFilter, String> by lazy {
        mapOf(
            OpenCvFilter.ORIGINAL              to getString(R.string.filter_desc_original),
            OpenCvFilter.GRAYSCALE             to getString(R.string.filter_desc_grayscale),
            OpenCvFilter.GAUSSIAN_BLUR         to getString(R.string.filter_desc_gaussian_blur),
            OpenCvFilter.MEDIAN_BLUR           to getString(R.string.filter_desc_median_blur),
            OpenCvFilter.BILATERAL_FILTER      to getString(R.string.filter_desc_bilateral),
            OpenCvFilter.BOX_FILTER            to getString(R.string.filter_desc_box_filter),
            OpenCvFilter.THRESHOLD             to getString(R.string.filter_desc_threshold),
            OpenCvFilter.ADAPTIVE_THRESHOLD    to getString(R.string.filter_desc_adaptive_threshold),
            OpenCvFilter.HISTOGRAM_EQUALIZATION to getString(R.string.filter_desc_hist_eq),
            OpenCvFilter.CANNY_EDGES           to getString(R.string.filter_desc_canny),
            OpenCvFilter.SOBEL                 to getString(R.string.filter_desc_sobel),
            OpenCvFilter.SCHARR                to getString(R.string.filter_desc_scharr),
            OpenCvFilter.LAPLACIAN             to getString(R.string.filter_desc_laplacian),
            OpenCvFilter.PREWITT               to getString(R.string.filter_desc_prewitt),
            OpenCvFilter.ROBERTS               to getString(R.string.filter_desc_roberts),
            OpenCvFilter.DILATE                to getString(R.string.filter_desc_dilate),
            OpenCvFilter.ERODE                 to getString(R.string.filter_desc_erode),
            OpenCvFilter.OPEN                  to getString(R.string.filter_desc_open),
            OpenCvFilter.CLOSE                 to getString(R.string.filter_desc_close),
            OpenCvFilter.GRADIENT              to getString(R.string.filter_desc_gradient),
            OpenCvFilter.TOP_HAT               to getString(R.string.filter_desc_top_hat),
            OpenCvFilter.BLACK_HAT             to getString(R.string.filter_desc_black_hat),
            OpenCvFilter.APRIL_TAGS            to getString(R.string.filter_desc_apriltag),
            OpenCvFilter.ARUCO                 to getString(R.string.filter_desc_aruco),
            OpenCvFilter.QR_CODE               to getString(R.string.filter_desc_qr),
            OpenCvFilter.CCTAG                 to getString(R.string.filter_desc_cctag),
            OpenCvFilter.HOLISTIC_BODY         to getString(R.string.filter_desc_body),
            OpenCvFilter.HOLISTIC_HANDS        to getString(R.string.filter_desc_hands),
            OpenCvFilter.HOLISTIC_FACE         to getString(R.string.filter_desc_face),
            OpenCvFilter.IRIS                  to getString(R.string.filter_desc_iris),
            OpenCvFilter.VISUAL_ODOMETRY       to getString(R.string.filter_desc_visual_odometry),
            OpenCvFilter.POINT_CLOUD           to getString(R.string.filter_desc_point_cloud),
            OpenCvFilter.PLANE_DETECTION       to getString(R.string.filter_desc_plane_detection),
            OpenCvFilter.VANISHING_POINTS      to getString(R.string.filter_desc_vanishing_points),
            OpenCvFilter.CHESSBOARD_CALIBRATION to getString(R.string.filter_desc_chessboard),
            OpenCvFilter.UNDISTORT             to getString(R.string.filter_desc_undistort),
            OpenCvFilter.YOLO_DETECT           to getString(R.string.filter_desc_yolo_detect),
            OpenCvFilter.YOLO_SEGMENT          to getString(R.string.filter_desc_yolo_segment),
            OpenCvFilter.YOLO_POSE             to getString(R.string.filter_desc_yolo_pose),
            OpenCvFilter.INVERT                to getString(R.string.filter_desc_invert),
            OpenCvFilter.SEPIA                 to getString(R.string.filter_desc_sepia),
            OpenCvFilter.EMBOSS                to getString(R.string.filter_desc_emboss),
            OpenCvFilter.PIXELATE              to getString(R.string.filter_desc_pixelate),
            OpenCvFilter.CARTOON               to getString(R.string.filter_desc_cartoon),
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMenuBinding.inflate(layoutInflater)
        setContentView(binding.root)
        showAppVersion()
        buildInstructionsCard()
        buildProcessingCards()
        buildDetectionCards()
        buildAnalysisCards()
        buildModelsTab()
        buildAboutContent()
        setupMenuTabs()
    }

    override fun onDestroy() {
        super.onDestroy()
        downloadExecutor.shutdownNow()
    }

    // ------------------------------------------------------------------
    // Tab setup
    // ------------------------------------------------------------------

    /** Configures the five-tab layout and switches the visible scroll view. */
    private fun setupMenuTabs() {
        with(binding.tabLayoutMenu) {
            addTab(newTab().setText(R.string.tab_processing))
            addTab(newTab().setText(R.string.tab_detection))
            addTab(newTab().setText(R.string.tab_analysis))
            addTab(newTab().setText(R.string.tab_models))
            addTab(newTab().setText(R.string.tab_about))
        }

        binding.tabLayoutMenu.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                updateTabVisibility(tab.position)
                if (tab.position == TAB_MODELS) refreshModelStatus()
            }

            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) {}
        })
    }

    /** Shows the scroll view that corresponds to [position] and hides all others. */
    private fun updateTabVisibility(position: Int) {
        binding.scrollViewModes.visibility      = if (position == TAB_PROCESSING) View.VISIBLE else View.GONE
        binding.scrollViewDetection.visibility  = if (position == TAB_DETECTION)  View.VISIBLE else View.GONE
        binding.scrollViewAnalysis.visibility   = if (position == TAB_ANALYSIS)   View.VISIBLE else View.GONE
        binding.scrollViewModels.visibility     = if (position == TAB_MODELS)     View.VISIBLE else View.GONE
        binding.scrollViewAbout.visibility      = if (position == TAB_ABOUT)      View.VISIBLE else View.GONE
    }

    // ------------------------------------------------------------------
    // "Przetwarzanie" tab – FILTERS, EDGES, MORPHOLOGY, EFFECTS
    // ------------------------------------------------------------------

    /**
     * Inflates an instructions card and inserts it as the first item in the processing
     * container. Tapping the card opens a dialog with a step-by-step usage guide.
     */
    private fun buildInstructionsCard() {
        val card = layoutInflater.inflate(R.layout.item_mode_card, binding.modeListContainer, false)
        card.findViewById<TextView>(R.id.textModeName).text = getString(R.string.instructions_card_title)
        card.findViewById<TextView>(R.id.textModeDescription).text = getString(R.string.instructions_card_description)
        card.setOnClickListener { showInstructionsDialog() }
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

    /**
     * Adds a short group description and then one card per mode to the preprocessing tab.
     * Cards are stroked with [R.color.group_processing] (teal).
     */
    private fun buildProcessingCards() {
        addGroupDescription(binding.modeListContainer, getString(R.string.group_desc_processing))
        val modes = listOf(AnalysisMode.FILTERS, AnalysisMode.EDGES, AnalysisMode.MORPHOLOGY, AnalysisMode.EFFECTS)
        modes.forEach { mode ->
            addModeCard(binding.modeListContainer, mode, R.color.group_processing)
        }
    }

    // ------------------------------------------------------------------
    // "Detekcja" tab – MARKERS, YOLO, POSE
    // ------------------------------------------------------------------

    /**
     * Adds a short group description and then one card per mode to the detection tab.
     * Cards are stroked with [R.color.group_detection] (deep orange).
     */
    private fun buildDetectionCards() {
        addGroupDescription(binding.detectionContainer, getString(R.string.group_desc_detection))
        val modes = listOf(AnalysisMode.MARKERS, AnalysisMode.YOLO, AnalysisMode.POSE)
        modes.forEach { mode ->
            addModeCard(binding.detectionContainer, mode, R.color.group_detection)
        }
    }

    // ------------------------------------------------------------------
    // "Analiza 3D" tab – ODOMETRY, GEOMETRY, CALIBRATION + PointCloudViewer
    // ------------------------------------------------------------------

    /**
     * Adds a short group description, one card per 3-D-analysis mode and a special card
     * for the [PointCloudViewerActivity] to the analysis tab.
     * Cards are stroked with [R.color.group_analysis] (purple).
     */
    private fun buildAnalysisCards() {
        addGroupDescription(binding.analysisContainer, getString(R.string.group_desc_analysis))
        val modes = listOf(AnalysisMode.ODOMETRY, AnalysisMode.GEOMETRY, AnalysisMode.CALIBRATION)
        modes.forEach { mode ->
            addModeCard(binding.analysisContainer, mode, R.color.group_analysis)
        }
        buildPointCloudViewerCard()
    }

    /** Appends a special card to open the point cloud viewer (not a live-camera mode). */
    private fun buildPointCloudViewerCard() {
        val card = layoutInflater.inflate(R.layout.item_mode_card, binding.analysisContainer, false)
        card.findViewById<TextView>(R.id.textModeName).text = getString(R.string.mode_point_cloud_viewer)
        card.findViewById<TextView>(R.id.textModeDescription).text = getString(R.string.mode_desc_point_cloud_viewer)
        applyGroupStroke(card, R.color.group_analysis)
        card.setOnClickListener {
            startActivity(Intent(this, PointCloudViewerActivity::class.java))
        }
        binding.analysisContainer.addView(card)
    }

    // ------------------------------------------------------------------
    // "Modele" tab
    // ------------------------------------------------------------------

    /** Builds the Models tab with MediaPipe and YOLO model status rows. */
    private fun buildModelsTab() {
        val container = binding.modelsContainer

        addGroupDescription(container, getString(R.string.models_tab_description))

        // MediaPipe section
        addSectionHeader(container, getString(R.string.models_mediapipe_title))
        addGroupDescription(container, getString(R.string.models_mediapipe_description))

        addModelRow(container, MediaPipeProcessor.MODEL_POSE,  getString(R.string.model_name_pose_landmarker),  isYolo = false)
        addModelRow(container, MediaPipeProcessor.MODEL_HAND,  getString(R.string.model_name_hand_landmarker),  isYolo = false)
        addModelRow(container, MediaPipeProcessor.MODEL_FACE,  getString(R.string.model_name_face_landmarker),  isYolo = false)
        addDownloadAllButton(container, isYolo = false)

        // YOLO section
        addSectionHeader(container, getString(R.string.models_yolo_title))
        addGroupDescription(container, getString(R.string.models_yolo_description))

        addModelRow(container, YoloProcessor.MODEL_DETECT,  getString(R.string.model_name_yolo_detect),  isYolo = true)
        addModelRow(container, YoloProcessor.MODEL_SEGMENT, getString(R.string.model_name_yolo_segment), isYolo = true)
        addModelRow(container, YoloProcessor.MODEL_POSE,    getString(R.string.model_name_yolo_pose),    isYolo = true)
        addDownloadAllButton(container, isYolo = true)
    }

    /**
     * Inflates an [R.layout.item_model_status] row and appends it to [container].
     * Stores references to its views in [modelStatusViews] for later status updates.
     */
    private fun addModelRow(container: LinearLayout, filename: String, displayName: String, isYolo: Boolean) {
        val row = layoutInflater.inflate(R.layout.item_model_status, container, false)
        val dotView    = row.findViewById<View>(R.id.viewModelStatusDot)
        val textName   = row.findViewById<TextView>(R.id.textModelName)
        val textSize   = row.findViewById<TextView>(R.id.textModelSize)
        val btnAction  = row.findViewById<MaterialButton>(R.id.btnModelAction)

        textName.text = displayName
        modelStatusViews[filename] = ModelStatusViews(dotView, textSize, btnAction, isYolo)

        btnAction.setOnClickListener { downloadSingleModel(filename, isYolo) }
        container.addView(row)
        updateModelRowUi(filename, isYolo)
    }

    /**
     * Adds a "Download all missing" [MaterialButton] at the bottom of the current section.
     *
     * @param isYolo ``true`` for the YOLO section, ``false`` for the MediaPipe section.
     */
    private fun addDownloadAllButton(container: LinearLayout, isYolo: Boolean) {
        val margin = resources.getDimensionPixelSize(R.dimen.group_stroke_width) * BUTTON_MARGIN_MULTIPLIER
        val btn = MaterialButton(this).apply {
            text = getString(R.string.model_download_all)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).also { it.setMargins(margin, margin / 2, margin, margin) }
            setOnClickListener { downloadAllModels(isYolo) }
        }
        container.addView(btn)
    }

    /**
     * Updates the status dot colour, status text and button state for a single model row.
     *
     * Reads the model's file from disk (fast local I/O) and reflects the result in the UI.
     */
    private fun updateModelRowUi(filename: String, isYolo: Boolean) {
        val path = if (isYolo) ModelDownloadManager.getYoloModelPath(this, filename)
                   else        ModelDownloadManager.getModelPath(this, filename)
        val views = modelStatusViews[filename] ?: return

        if (path != null) {
            val sizeStr = formatFileSize(java.io.File(path).length())
            ViewCompat.setBackgroundTintList(
                views.dotView,
                ColorStateList.valueOf(ContextCompat.getColor(this, R.color.model_status_ready))
            )
            views.textModelSize.text = getString(R.string.model_status_available, sizeStr)
            views.btnAction.text     = getString(R.string.model_btn_downloaded)
            views.btnAction.isEnabled = false
        } else {
            ViewCompat.setBackgroundTintList(
                views.dotView,
                ColorStateList.valueOf(ContextCompat.getColor(this, R.color.model_status_missing))
            )
            views.textModelSize.text  = getString(R.string.model_status_missing)
            views.btnAction.text      = getString(R.string.model_btn_download)
            views.btnAction.isEnabled = true
        }
    }

    /** Refreshes all model status rows; called whenever the Models tab is selected. */
    private fun refreshModelStatus() {
        modelStatusViews.entries.forEach { (filename, views) ->
            updateModelRowUi(filename, views.isYolo)
        }
    }

    /**
     * Downloads a single model on the background thread and updates its row on completion.
     *
     * @param filename Model filename key, e.g. [MediaPipeProcessor.MODEL_POSE].
     * @param isYolo   ``true`` if the model belongs to the YOLO group.
     */
    private fun downloadSingleModel(filename: String, isYolo: Boolean) {
        val views = modelStatusViews[filename] ?: return
        val url   = if (isYolo) ModelDownloadManager.YOLO_MODEL_URLS[filename]
                    else        ModelDownloadManager.MODEL_URLS[filename]
        if (url == null) return

        views.btnAction.isEnabled = false
        views.btnAction.text      = getString(R.string.model_btn_downloading)

        downloadExecutor.execute {
            val success = ModelDownloadManager.downloadModel(this, filename, url)
            runOnUiThread {
                updateModelRowUi(filename, isYolo)
                val msgRes = if (success) R.string.model_download_success else R.string.model_download_failed
                Toast.makeText(this, getString(msgRes, filename), Toast.LENGTH_SHORT).show()
                if (!success) {
                    views.btnAction.isEnabled = true
                    views.btnAction.text      = getString(R.string.model_btn_download)
                }
            }
        }
    }

    /**
     * Downloads all missing models for a group (MediaPipe or YOLO) in the background.
     *
     * @param isYolo ``true`` to download all missing YOLO models,
     *               ``false`` to download all missing MediaPipe models.
     */
    private fun downloadAllModels(isYolo: Boolean) {
        downloadExecutor.execute {
            val success = if (isYolo) {
                ModelDownloadManager.downloadMissingYoloModels(this)
            } else {
                ModelDownloadManager.downloadMissingModels(this)
            }
            runOnUiThread {
                refreshModelStatus()
                val msgRes = if (success) R.string.model_download_all_success
                             else         R.string.model_download_all_failed
                Toast.makeText(this, msgRes, Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ------------------------------------------------------------------
    // "O aplikacji" tab
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

    // ------------------------------------------------------------------
    // Navigation
    // ------------------------------------------------------------------

    /** Starts [MainActivity] with the chosen [mode] pre-selected. */
    private fun launchMainActivity(mode: AnalysisMode) {
        startActivity(
            Intent(this, MainActivity::class.java).apply {
                putExtra(EXTRA_MODE, mode.name)
            }
        )
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

    // ------------------------------------------------------------------
    // Shared card / UI helpers
    // ------------------------------------------------------------------

    /**
     * Inflates an [R.layout.item_mode_card], sets mode name + description,
     * applies a coloured stroke for the given processing group, wires the click
     * listener to launch [MainActivity] and appends it to [container].
     *
     * @param container  Parent [LinearLayout] to append the card to.
     * @param mode       [AnalysisMode] represented by this card.
     * @param strokeColorRes Colour resource ID for the card stroke.
     */
    private fun addModeCard(container: LinearLayout, mode: AnalysisMode, strokeColorRes: Int) {
        val card = layoutInflater.inflate(R.layout.item_mode_card, container, false)
        card.findViewById<TextView>(R.id.textModeName).text = mode.displayName
        card.findViewById<TextView>(R.id.textModeDescription).text = modeDescriptions[mode] ?: ""
        applyGroupStroke(card, strokeColorRes)
        card.setOnClickListener { launchMainActivity(mode) }
        container.addView(card)
    }

    /**
     * Applies a visible coloured stroke to a [com.google.android.material.card.MaterialCardView]
     * to visually differentiate cards belonging to different processing groups.
     *
     * @param cardView    The inflated [R.layout.item_mode_card] root view.
     * @param colorRes    Colour resource ID for the stroke.
     */
    private fun applyGroupStroke(cardView: View, colorRes: Int) {
        if (cardView is com.google.android.material.card.MaterialCardView) {
            cardView.strokeColor = ContextCompat.getColor(this, colorRes)
            cardView.strokeWidth = resources.getDimensionPixelSize(R.dimen.group_stroke_width)
        }
    }

    /**
     * Adds a descriptive [TextView] above the mode cards in a group container.
     *
     * @param container The [LinearLayout] to prepend the description to.
     * @param text      Short Polish description of the processing stage.
     */
    private fun addGroupDescription(container: LinearLayout, text: String) {
        val padding = resources.getDimensionPixelSize(R.dimen.group_stroke_width) * 4
        val secondaryTextColor = com.google.android.material.color.MaterialColors.getColor(
            this, com.google.android.material.R.attr.colorOnSurfaceVariant, android.graphics.Color.GRAY
        )
        val tv = TextView(this).apply {
            this.text = text
            setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium)
            setTextColor(secondaryTextColor)
            setPadding(padding, padding, padding, padding)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).also { it.setMargins(0, 0, 0, padding / 2) }
        }
        container.addView(tv)
    }

    /**
     * Adds a bold section-header [TextView] to [container].
     * Used to separate the MediaPipe and YOLO sub-sections in the Models tab.
     *
     * @param container The [LinearLayout] to append the header to.
     * @param text      Header title text.
     */
    private fun addSectionHeader(container: LinearLayout, text: String) {
        val padding = resources.getDimensionPixelSize(R.dimen.group_stroke_width) * 4
        val tv = TextView(this).apply {
            this.text = text
            setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleMedium)
            setPadding(padding, padding * 2, padding, 0)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        container.addView(tv)
    }

    /**
     * Formats a file size given in bytes to a human-readable string (B / KB / MB).
     *
     * @param bytes File size in bytes.
     * @return Formatted string such as "4.2 MB".
     */
    private fun formatFileSize(bytes: Long): String = when {
        bytes < 1_024L              -> "$bytes B"
        bytes < 1_024L * 1_024L     -> "${bytes / 1_024L} KB"
        else                        -> String.format("%.1f MB", bytes / (1_024.0 * 1_024.0))
    }
}
