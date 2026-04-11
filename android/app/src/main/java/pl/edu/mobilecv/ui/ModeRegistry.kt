package pl.edu.mobilecv.ui

import androidx.annotation.ColorRes
import androidx.annotation.StringRes
import pl.edu.mobilecv.AnalysisMode
import pl.edu.mobilecv.OpenCvFilter
import pl.edu.mobilecv.R

/**
 * Centralny rejestr metadanych trybów i filtrów wykorzystywanych przez menu.
 *
 * Rejestr trzyma:
 * - przynależność trybu do grupy funkcjonalnej,
 * - etykietę i opis trybu,
 * - listę kontrolek (filtrów) dostępnych w danym trybie.
 */
object ModeRegistry {

    /** Grupy funkcjonalne mapowane na zakładki w [pl.edu.mobilecv.MenuActivity]. */
    enum class FunctionalGroup(
        @StringRes val descriptionResId: Int,
        @ColorRes val strokeColorResId: Int,
    ) {
        PROCESSING(
            descriptionResId = R.string.group_desc_processing,
            strokeColorResId = R.color.group_processing
        ),
        DETECTION(
            descriptionResId = R.string.group_desc_detection,
            strokeColorResId = R.color.group_detection
        ),
        ANALYSIS(
            descriptionResId = R.string.group_desc_analysis,
            strokeColorResId = R.color.group_analysis
        ),
    }

    /** Definicja pojedynczego trybu pokazywanego w menu. */
    data class ModeEntry(
        val mode: AnalysisMode,
        val group: FunctionalGroup,
        @StringRes val descriptionResId: Int,
        val controls: List<OpenCvFilter>,
    ) {
        val label: String = mode.displayName
    }

    private val modeEntries: Map<AnalysisMode, ModeEntry> = listOf(
        ModeEntry(AnalysisMode.FILTERS, FunctionalGroup.PROCESSING, R.string.mode_desc_filters, AnalysisMode.FILTERS.filters),
        ModeEntry(AnalysisMode.EDGES, FunctionalGroup.PROCESSING, R.string.mode_desc_edges, AnalysisMode.EDGES.filters),
        ModeEntry(AnalysisMode.MORPHOLOGY, FunctionalGroup.PROCESSING, R.string.mode_desc_morphology, AnalysisMode.MORPHOLOGY.filters),
        ModeEntry(AnalysisMode.EFFECTS, FunctionalGroup.PROCESSING, R.string.mode_desc_effects, AnalysisMode.EFFECTS.filters),
        ModeEntry(AnalysisMode.MARKERS, FunctionalGroup.DETECTION, R.string.mode_desc_markers, AnalysisMode.MARKERS.filters),
        ModeEntry(AnalysisMode.POSE, FunctionalGroup.DETECTION, R.string.mode_desc_pose, AnalysisMode.POSE.filters),
        ModeEntry(AnalysisMode.YOLO, FunctionalGroup.DETECTION, R.string.mode_desc_yolo, AnalysisMode.YOLO.filters),
        ModeEntry(
            AnalysisMode.ACTIVE_TRACKING,
            FunctionalGroup.DETECTION,
            R.string.mode_desc_active_tracking,
            AnalysisMode.ACTIVE_TRACKING.filters
        ),
        ModeEntry(
            AnalysisMode.ODOMETRY_UNIFIED,
            FunctionalGroup.ANALYSIS,
            R.string.mode_desc_odometry,
            AnalysisMode.ODOMETRY_UNIFIED.filters
        ),
        ModeEntry(AnalysisMode.SLAM, FunctionalGroup.ANALYSIS, R.string.mode_desc_slam, AnalysisMode.SLAM.filters),
        ModeEntry(AnalysisMode.CALIBRATION, FunctionalGroup.ANALYSIS, R.string.mode_desc_calibration, AnalysisMode.CALIBRATION.filters),
    ).associateBy { it.mode }

    private val filterDescriptionResIds: Map<OpenCvFilter, Int> = mapOf(
        OpenCvFilter.ORIGINAL to R.string.filter_desc_original,
        OpenCvFilter.GRAYSCALE to R.string.filter_desc_grayscale,
        OpenCvFilter.GAUSSIAN_BLUR to R.string.filter_desc_gaussian_blur,
        OpenCvFilter.MEDIAN_BLUR to R.string.filter_desc_median_blur,
        OpenCvFilter.BILATERAL_FILTER to R.string.filter_desc_bilateral,
        OpenCvFilter.BOX_FILTER to R.string.filter_desc_box_filter,
        OpenCvFilter.THRESHOLD to R.string.filter_desc_threshold,
        OpenCvFilter.ADAPTIVE_THRESHOLD to R.string.filter_desc_adaptive_threshold,
        OpenCvFilter.HISTOGRAM_EQUALIZATION to R.string.filter_desc_hist_eq,
        OpenCvFilter.CANNY_EDGES to R.string.filter_desc_canny,
        OpenCvFilter.SOBEL to R.string.filter_desc_sobel,
        OpenCvFilter.SCHARR to R.string.filter_desc_scharr,
        OpenCvFilter.LAPLACIAN to R.string.filter_desc_laplacian,
        OpenCvFilter.PREWITT to R.string.filter_desc_prewitt,
        OpenCvFilter.ROBERTS to R.string.filter_desc_roberts,
        OpenCvFilter.DILATE to R.string.filter_desc_dilate,
        OpenCvFilter.ERODE to R.string.filter_desc_erode,
        OpenCvFilter.OPEN to R.string.filter_desc_open,
        OpenCvFilter.CLOSE to R.string.filter_desc_close,
        OpenCvFilter.GRADIENT to R.string.filter_desc_gradient,
        OpenCvFilter.TOP_HAT to R.string.filter_desc_top_hat,
        OpenCvFilter.BLACK_HAT to R.string.filter_desc_black_hat,
        OpenCvFilter.APRIL_TAGS to R.string.filter_desc_apriltag,
        OpenCvFilter.ARUCO to R.string.filter_desc_aruco,
        OpenCvFilter.QR_CODE to R.string.filter_desc_qr,
        OpenCvFilter.CCTAG to R.string.filter_desc_cctag,
        OpenCvFilter.ACTIVE_TRACKING to R.string.filter_desc_active_tracking,
        OpenCvFilter.HOLISTIC_BODY to R.string.filter_desc_body,
        OpenCvFilter.HOLISTIC_HANDS to R.string.filter_desc_hands,
        OpenCvFilter.HOLISTIC_FACE to R.string.filter_desc_face,
        OpenCvFilter.IRIS to R.string.filter_desc_iris,
        OpenCvFilter.FACE_DETECTION_BLAZE to R.string.filter_desc_face_blaze,
        OpenCvFilter.HOLOGRAM_3D to R.string.filter_desc_hologram_3d,
        OpenCvFilter.EMOTION_RECOGNITION to R.string.filter_desc_emotion,
        OpenCvFilter.VISUAL_ODOMETRY to R.string.filter_desc_visual_odometry,
        OpenCvFilter.CHESSBOARD_CALIBRATION to R.string.filter_desc_chessboard,
        OpenCvFilter.UNDISTORT to R.string.filter_desc_undistort,
        OpenCvFilter.YOLO_DETECT to R.string.filter_desc_yolo_detect,
        OpenCvFilter.YOLO_KALMAN to R.string.filter_desc_yolo_kalman,
        OpenCvFilter.YOLO_SEGMENT to R.string.filter_desc_yolo_segment,
        OpenCvFilter.YOLO_POSE to R.string.filter_desc_yolo_pose,
        OpenCvFilter.YOLO_CLASSIFY to R.string.filter_desc_yolo_classify,
        OpenCvFilter.YOLO_OBB to R.string.filter_desc_yolo_obb,
        OpenCvFilter.MARKER_UKF to R.string.filter_desc_marker_ukf,
        OpenCvFilter.FULL_ODOMETRY to R.string.filter_desc_full_odometry,
        OpenCvFilter.ODOMETRY_TRAJECTORY to R.string.filter_desc_odometry_trajectory,
        OpenCvFilter.ODOMETRY_MAP to R.string.filter_desc_odometry_map,
        OpenCvFilter.SLAM_MARKERS to R.string.filter_desc_slam_markers,
        OpenCvFilter.SLAM_POINTS to R.string.filter_desc_slam_points,
        OpenCvFilter.SLAM_MARKERS_FUSED to R.string.filter_desc_slam_markers_fused,
        OpenCvFilter.INVERT to R.string.filter_desc_invert,
        OpenCvFilter.SEPIA to R.string.filter_desc_sepia,
        OpenCvFilter.EMBOSS to R.string.filter_desc_emboss,
        OpenCvFilter.PIXELATE to R.string.filter_desc_pixelate,
        OpenCvFilter.CARTOON to R.string.filter_desc_cartoon,
    )

    /** Zwraca wszystkie wpisy dla podanej grupy w stałej, deklaratywnej kolejności. */
    fun entriesForGroup(group: FunctionalGroup): List<ModeEntry> =
        modeEntries.values.filter { it.group == group }

    /** Zwraca wpis rejestru dla trybu albo rzuca czytelny błąd developerski. */
    fun requireEntry(mode: AnalysisMode): ModeEntry =
        modeEntries[mode] ?: error(
            "ModeRegistry error: missing entry for AnalysisMode.${mode.name}. " +
                "Please add it to ModeRegistry.modeEntries."
        )

    /** Zwraca identyfikator zasobu opisu filtra albo `null`, jeśli brak opisu. */
    fun filterDescriptionResId(filter: OpenCvFilter): Int? = filterDescriptionResIds[filter]

    /** Weryfikuje kompletność rejestru względem [AnalysisMode]. */
    fun validateConsistency() {
        val missingModes = AnalysisMode.entries.filterNot { modeEntries.containsKey(it) }
        check(missingModes.isEmpty()) {
            "ModeRegistry consistency error: missing entries for modes: " +
                missingModes.joinToString { it.name }
        }
    }
}
