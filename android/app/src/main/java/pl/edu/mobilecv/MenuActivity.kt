package pl.edu.mobilecv

import android.content.Intent
import android.content.res.ColorStateList
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
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
import kotlin.jvm.java

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

    /** Groups of downloadable models shown in the Models tab. */
    private enum class ModelGroup { MEDIAPIPE, YOLO, RTMDET, MOBILINT }

    private lateinit var binding: ActivityMenuBinding

    /** Single-thread executor used for background model downloads. Lazily created on first use. */
    private val downloadExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }

    /** Holds references to the UI elements of each model status row, plus the model group flag. */
    private data class ModelStatusViews(
        val dotView: View,
        val textModelSize: TextView,
        val btnAction: MaterialButton,
        val progressBar: ProgressBar,
        val group: ModelGroup,
    )

    /** Keyed by model filename (e.g. [MediaPipeProcessor.MODEL_POSE]). */
    private val modelStatusViews = mutableMapOf<String, ModelStatusViews>()

    /** Lazily built map of [AnalysisMode] to its Polish description string. */
    private val modeDescriptions: Map<AnalysisMode, String> by lazy {
        mapOf(
            AnalysisMode.FILTERS         to getString(R.string.mode_desc_filters),
            AnalysisMode.EDGES           to getString(R.string.mode_desc_edges),
            AnalysisMode.MORPHOLOGY      to getString(R.string.mode_desc_morphology),
            AnalysisMode.MARKERS         to getString(R.string.mode_desc_markers),
            AnalysisMode.POSE            to getString(R.string.mode_desc_pose),
            AnalysisMode.ODOMETRY        to getString(R.string.mode_desc_odometry),
            AnalysisMode.FULL_ODOMETRY_3D to getString(R.string.mode_desc_full_odometry_3d),
            AnalysisMode.GEOMETRY        to getString(R.string.mode_desc_geometry),
            AnalysisMode.CALIBRATION     to getString(R.string.mode_desc_calibration),
            AnalysisMode.YOLO            to getString(R.string.mode_desc_yolo),
            AnalysisMode.RTMDET          to getString(R.string.mode_desc_rtmdet),
            AnalysisMode.MOBILINT        to getString(R.string.mode_desc_mobilint),
            AnalysisMode.EFFECTS         to getString(R.string.mode_desc_effects),
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
            OpenCvFilter.YOLO_CLASSIFY         to getString(R.string.filter_desc_yolo_classify),
            OpenCvFilter.YOLO_OBB              to getString(R.string.filter_desc_yolo_obb),
            OpenCvFilter.RTMDET_DETECT         to getString(R.string.filter_desc_rtmdet_detect),
            OpenCvFilter.RTMDET_ROTATED        to getString(R.string.filter_desc_rtmdet_rotated),
            OpenCvFilter.FULL_ODOMETRY         to getString(R.string.filter_desc_full_odometry),
            OpenCvFilter.ODOMETRY_TRAJECTORY   to getString(R.string.filter_desc_odometry_trajectory),
            OpenCvFilter.ODOMETRY_MAP          to getString(R.string.filter_desc_odometry_map),
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
        suggestModelsTabIfNeeded()
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
     * Inflates an [R.layout.item_mode_card], sets title + description,
     * applies a coloured stroke and wires the click listener.
     */
    private fun addCustomCard(
        container: LinearLayout,
        title: String,
        description: String,
        strokeColorRes: Int,
        onClick: () -> Unit
    ): View {
        val card = layoutInflater.inflate(R.layout.item_mode_card, container, false)
        card.findViewById<TextView>(R.id.textModeName).text = title
        card.findViewById<TextView>(R.id.textModeDescription).text = description
        if (strokeColorRes != 0) {
            applyGroupStroke(card, strokeColorRes)
        }
        card.setOnClickListener { onClick() }
        container.addView(card)
        return card
    }

    /**
     * Inflates an instructions card and inserts it as the first item in the processing
     * container. Tapping the card opens a dialog with a step-by-step usage guide.
     */
    private fun buildInstructionsCard() {
        addCustomCard(
            container = binding.modeListContainer,
            title = getString(R.string.instructions_card_title),
            description = getString(R.string.instructions_card_description),
            strokeColorRes = R.color.group_processing // Default for first tab
        ) {
            showInstructionsDialog()
        }
    }

    /**
     * Inflates a tutorial card that launches [OdometryTutorialActivity].
     *
     * This card is added to the 3D Analysis container to provide users with
     * educational context regarding visual odometry and SLAM (Simultaneous
     * Localization and Mapping) before they use the live tracking modes.
     */
    private fun buildOdometryTutorialCard() {
        addCustomCard(
            container = binding.analysisContainer,
            title = getString(R.string.tutorial_card_title),
            description = getString(R.string.tutorial_card_description),
            strokeColorRes = R.color.group_analysis
        ) {
            startActivity(Intent(this, OdometryTutorialActivity::class.java))
        }
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
        val modes = listOf(
            AnalysisMode.MARKERS,
            AnalysisMode.YOLO,
            AnalysisMode.RTMDET,
            AnalysisMode.MOBILINT,
            AnalysisMode.POSE
        )
        modes.forEach { mode ->
            addModeCard(binding.detectionContainer, mode, R.color.group_detection)
        }
    }

    // ------------------------------------------------------------------
    // "Analiza 3D" tab – ODOMETRY, FULL_ODOMETRY_3D, GEOMETRY, CALIBRATION + PointCloudViewer
    // ------------------------------------------------------------------

    /**
     * Adds a short group description, one card per 3-D-analysis mode and a special card
     * for the [PointCloudViewerActivity] to the analysis tab.
     * Cards are stroked with [R.color.group_analysis] (purple).
     */
    private fun buildAnalysisCards() {
        addGroupDescription(binding.analysisContainer, getString(R.string.group_desc_analysis))
        val modes = listOf(AnalysisMode.ODOMETRY, AnalysisMode.FULL_ODOMETRY_3D, AnalysisMode.GEOMETRY, AnalysisMode.CALIBRATION)
        modes.forEach { mode ->
            addModeCard(binding.analysisContainer, mode, R.color.group_analysis)
        }
        buildOdometryTutorialCard()
        buildPointCloudViewerCard()
    }

    /** Appends a special card to open the point cloud viewer (not a live-camera mode). */
    private fun buildPointCloudViewerCard() {
        addCustomCard(
            container = binding.analysisContainer,
            title = getString(R.string.mode_point_cloud_viewer),
            description = getString(R.string.mode_desc_point_cloud_viewer),
            strokeColorRes = R.color.group_analysis
        ) {
            startActivity(Intent(this, PointCloudViewerActivity::class.java))
        }
    }

    // ------------------------------------------------------------------
    // "Modele" tab
    // ------------------------------------------------------------------

    /** Builds the Models tab with MediaPipe, YOLO and RTMDet model status rows. */
    private fun buildModelsTab() {
        val container = binding.modelsContainer

        addGroupDescription(container, getString(R.string.models_tab_description))

        // MediaPipe section
        addSectionHeader(container, getString(R.string.models_mediapipe_title))
        addGroupDescription(container, getString(R.string.models_mediapipe_description))

        addModelRow(container, MediaPipeProcessor.MODEL_POSE,  getString(R.string.model_name_pose_landmarker),  ModelGroup.MEDIAPIPE)
        addModelRow(container, MediaPipeProcessor.MODEL_HAND,  getString(R.string.model_name_hand_landmarker),  ModelGroup.MEDIAPIPE)
        addModelRow(container, MediaPipeProcessor.MODEL_FACE,  getString(R.string.model_name_face_landmarker),  ModelGroup.MEDIAPIPE)
        addModelRow(container, MediaPipeProcessor.MODEL_FACE_DETECTOR, getString(R.string.model_name_face_detector), ModelGroup.MEDIAPIPE)
        addDownloadAllButton(container, ModelGroup.MEDIAPIPE)

        // YOLO section -- rows keyed by the *.pt filenames from YOLO_MODEL_URLS
        addSectionHeader(container, getString(R.string.models_yolo_title))
        addGroupDescription(container, getString(R.string.models_yolo_description))

        addModelRow(container, "yolov8n.pt",       getString(R.string.model_name_yolo_detect),   ModelGroup.YOLO)
        addModelRow(container, "yolov8n-seg.pt",   getString(R.string.model_name_yolo_segment),  ModelGroup.YOLO)
        addModelRow(container, "yolov8n-pose.pt",  getString(R.string.model_name_yolo_pose),     ModelGroup.YOLO)
        addModelRow(container, "yolov8n-cls.pt",   getString(R.string.model_name_yolo_classify), ModelGroup.YOLO)
        addModelRow(container, "yolov8n-obb.pt",   getString(R.string.model_name_yolo_obb),      ModelGroup.YOLO)
        addDownloadAllButton(container, ModelGroup.YOLO)

        // RTMDet section
        addSectionHeader(container, getString(R.string.models_rtmdet_title))
        addGroupDescription(container, getString(R.string.models_rtmdet_description))

        addModelRow(container, RtmDetProcessor.MODEL_DETECT,   getString(R.string.model_name_rtmdet_detect),   ModelGroup.RTMDET)
        addModelRow(container, RtmDetProcessor.MODEL_ROTATED,  getString(R.string.model_name_rtmdet_rotated),  ModelGroup.RTMDET)
        addDownloadAllButton(container, ModelGroup.RTMDET)

        // Mobilint section
        addSectionHeader(container, getString(R.string.models_mobilint_title))
        addGroupDescription(container, getString(R.string.models_mobilint_description))

        addModelRow(container, "mobilint_detect.mbl", getString(R.string.model_name_mobilint_detect), ModelGroup.MOBILINT)
        addDownloadAllButton(container, ModelGroup.MOBILINT)
    }

    /**
     * Inflates an [R.layout.item_model_status] row and appends it to [container].
     * Stores references to its views in [modelStatusViews] for later status updates.
     */
    private fun addModelRow(container: LinearLayout, filename: String, displayName: String, group: ModelGroup) {
        val row = layoutInflater.inflate(R.layout.item_model_status, container, false)
        val dotView    = row.findViewById<View>(R.id.viewModelStatusDot)
        val textName   = row.findViewById<TextView>(R.id.textModelName)
        val textSize   = row.findViewById<TextView>(R.id.textModelSize)
        val btnAction  = row.findViewById<MaterialButton>(R.id.btnModelAction)
        val progressBar = row.findViewById<ProgressBar>(R.id.progressBarModelDownload)

        textName.text = displayName
        modelStatusViews[filename] = ModelStatusViews(dotView, textSize, btnAction, progressBar, group)

        btnAction.setOnClickListener { downloadSingleModel(filename, group) }
        container.addView(row)
        updateModelRowUi(filename, group)
    }

    /**
     * Adds a "Download all missing" [MaterialButton] at the bottom of the current section.
     *
     * @param group Model group whose missing files this button should download.
     */
    private fun addDownloadAllButton(container: LinearLayout, group: ModelGroup) {
        val margin = resources.getDimensionPixelSize(R.dimen.group_stroke_width) * BUTTON_MARGIN_MULTIPLIER
        val btn = MaterialButton(this).apply {
            text = getString(R.string.model_download_all)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).also { it.setMargins(margin, margin / 2, margin, margin) }
            setOnClickListener { downloadAllModels(group) }
        }
        container.addView(btn)
    }

    /**
     * Updates the status dot colour, status text and button state for a single model row.
     *
     * Reads the model's file from disk (fast local I/O) and reflects the result in the UI.
     */
    private fun updateModelRowUi(filename: String, group: ModelGroup) {
        val path = when (group) {
            ModelGroup.YOLO    -> ModelDownloadManager.getYoloModelPath(this, filename)
            ModelGroup.RTMDET  -> ModelDownloadManager.getRtmDetModelPath(this, filename)
            ModelGroup.MEDIAPIPE -> ModelDownloadManager.getModelPath(this, filename)
            ModelGroup.MOBILINT  -> ModelDownloadManager.getMobilintModelPath(this, filename)
        }
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
            updateModelRowUi(filename, views.group)
        }
    }

    /**
     * Downloads a single model on the background thread and updates its row on completion.
     * Shows an indeterminate progress bar in the row while the download is in progress.
     *
     * @param filename Model filename key, e.g. [MediaPipeProcessor.MODEL_POSE].
     * @param group    Model group this file belongs to.
     */
    private fun downloadSingleModel(filename: String, group: ModelGroup) {
        val views = modelStatusViews[filename] ?: return
        val url = when (group) {
            ModelGroup.YOLO    -> ModelDownloadManager.YOLO_MODEL_URLS[filename]
            ModelGroup.RTMDET  -> ModelDownloadManager.RTMDET_MODEL_URLS[filename]
            ModelGroup.MEDIAPIPE -> ModelDownloadManager.MODEL_URLS[filename]
            ModelGroup.MOBILINT  -> ModelDownloadManager.MOBILINT_MODEL_URLS[filename]
        }
        if (url == null) return

        views.btnAction.isEnabled = false
        views.btnAction.text      = getString(R.string.model_btn_downloading)
        views.progressBar.visibility = View.VISIBLE

        downloadExecutor.execute {
            val success = when (group) {
                ModelGroup.YOLO    -> ModelDownloadManager.downloadYoloModel(this, filename, url)
                ModelGroup.RTMDET  -> ModelDownloadManager.downloadRtmDetModel(this, filename, url)
                ModelGroup.MEDIAPIPE -> ModelDownloadManager.downloadModel(this, filename, url)
                ModelGroup.MOBILINT  -> ModelDownloadManager.downloadMobilintModel(this, filename, url)
            }
            runOnUiThread {
                views.progressBar.visibility = View.GONE
                updateModelRowUi(filename, group)
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
     * Downloads all missing models for a group (MediaPipe, YOLO or RTMDet) in the background.
     * Shows indeterminate progress bars for all missing rows while the download runs.
     *
     * @param group Model group to download.
     */
    private fun downloadAllModels(group: ModelGroup) {
        val groupFilenames = modelStatusViews.entries
            .filter { it.value.group == group }
            .map { it.key }

        groupFilenames.forEach { filename ->
            val views = modelStatusViews[filename] ?: return@forEach
            val alreadyPresent = when (group) {
                ModelGroup.YOLO    -> ModelDownloadManager.getYoloModelPath(this, filename) != null
                ModelGroup.RTMDET  -> ModelDownloadManager.getRtmDetModelPath(this, filename) != null
                ModelGroup.MEDIAPIPE -> ModelDownloadManager.getModelPath(this, filename) != null
                ModelGroup.MOBILINT  -> ModelDownloadManager.getMobilintModelPath(this, filename) != null
            }
            if (!alreadyPresent) {
                views.progressBar.visibility = View.VISIBLE
                views.btnAction.isEnabled = false
                views.btnAction.text = getString(R.string.model_btn_downloading)
            }
        }

        downloadExecutor.execute {
            val success = when (group) {
                ModelGroup.YOLO    -> ModelDownloadManager.downloadMissingYoloModels(this)
                ModelGroup.RTMDET  -> ModelDownloadManager.downloadMissingRtmDetModels(this)
                ModelGroup.MEDIAPIPE -> ModelDownloadManager.downloadMissingModels(this)
                ModelGroup.MOBILINT  -> ModelDownloadManager.downloadMissingMobilintModels(this)
            }
            runOnUiThread {
                groupFilenames.forEach { filename ->
                    modelStatusViews[filename]?.progressBar?.visibility = View.GONE
                }
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
     * Shows a one-time suggestion dialog at startup when any downloadable model is absent.
     * Tapping the positive button navigates directly to the Models tab.
     */
    private fun suggestModelsTabIfNeeded() {
        val mediapipeMissing = !ModelDownloadManager.areAllModelsReady(this)
        val yoloMissing = ModelDownloadManager.YOLO_MODEL_URLS.keys.any {
            ModelDownloadManager.getYoloModelPath(this, it) == null
        }
        val rtmdetMissing = !ModelDownloadManager.areRtmDetModelsReady(this)

        if (!mediapipeMissing && !yoloMissing && !rtmdetMissing) return

        AlertDialog.Builder(this)
            .setTitle(R.string.models_missing_title)
            .setMessage(R.string.models_missing_message)
            .setPositiveButton(R.string.models_go_to_tab) { _, _ ->
                binding.tabLayoutMenu.getTabAt(TAB_MODELS)?.select()
                updateTabVisibility(TAB_MODELS)
                refreshModelStatus()
            }
            .setNegativeButton(R.string.models_later, null)
            .show()
    }

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
        addCustomCard(
            container = binding.aboutContainer,
            title = getString(R.string.app_name),
            description = getString(R.string.about_app_description),
            strokeColorRes = 0 // No stroke for general info
        ) {
            // No action
        }.apply {
            isClickable = false
            isFocusable = false
        }
    }

    /**
     * Adds a section-header card for [mode] in the About tab.
     * The card shows the mode name and its high-level description.
     */
    private fun addModeSectionCard(mode: AnalysisMode) {
        addCustomCard(
            container = binding.aboutContainer,
            title = mode.displayName,
            description = modeDescriptions[mode] ?: "",
            strokeColorRes = 0
        ) {
            // No action
        }.apply {
            isClickable = false
            isFocusable = false
        }
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
        addCustomCard(
            container = container,
            title = mode.displayName,
            description = modeDescriptions[mode] ?: "",
            strokeColorRes = strokeColorRes
        ) {
            launchMainActivity(mode)
        }
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
     * Used to separate the MediaPipe and YOLO subsections in the Models tab.
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
