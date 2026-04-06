package pl.edu.mobilecv

/**
 * Enumeration of all available image-processing filters and detectors.
 *
 * Each entry bundles a human-readable [displayName] shown in the UI.
 */
enum class OpenCvFilter(val displayName: String) {
    /** No processing – raw camera feed passed through unchanged. */
    ORIGINAL("Original"),

    /** Converts the frame to grayscale. */
    GRAYSCALE("Grayscale"),

    /** Detects edges using the Canny algorithm. */
    CANNY_EDGES("Canny Edges"),

    /** Applies a 15×15 Gaussian blur. */
    GAUSSIAN_BLUR("Gaussian Blur"),

    /** Binary threshold at 127 (Otsu-style midpoint). */
    THRESHOLD("Threshold"),

    /** Gradient magnitude via combined Sobel X and Y operators. */
    SOBEL("Sobel Edges"),

    /** Second-order derivative edges via the Laplacian operator. */
    LAPLACIAN("Laplacian"),

    /** Morphological dilation with a configurable rectangular kernel. */
    DILATE("Dilate"),

    /** Morphological erosion with a configurable rectangular kernel. */
    ERODE("Erode"),

    /** Morphological opening (erosion then dilation) – removes small bright spots. */
    OPEN("Open"),

    /** Morphological closing (dilation then erosion) – fills small dark holes. */
    CLOSE("Close"),

    /** Morphological gradient (dilation minus erosion) – highlights object boundaries. */
    GRADIENT("Gradient"),

    /** White top-hat transform (original minus opening) – extracts bright fine details. */
    TOP_HAT("Top-Hat"),

    /** Black top-hat transform (closing minus original) – extracts dark fine details. */
    BLACK_HAT("Black-Hat"),

    /** Detects AprilTag fiducial markers and overlays position info. */
    APRIL_TAGS("AprilTag"),

    /** Detects ArUco markers (4×4 dictionary) and overlays position info. */
    ARUCO("ArUco"),

    /** Detects QR codes and overlays position info. */
    QR_CODE("QR Code"),

    /** Detects CCTag (Circular Concentric Tag) markers and overlays ring count and position. */
    CCTAG("CCTag"),

    /** Detects chessboard corners for camera calibration. */
    CHESSBOARD_CALIBRATION("Szachownica"),

    /** Shows the undistorted camera feed using stored calibration data. */
    UNDISTORT("Korekcja"),

    // ------------------------------------------------------------------
    // MediaPipe Holistic & Iris
    // ------------------------------------------------------------------

    /**
     * Detects and tracks full-body pose landmarks (33 points) using
     * MediaPipe Pose Landmarker.
     */
    HOLISTIC_BODY("Poza ciała"),

    /**
     * Detects and tracks hand landmarks (21 points per hand) for both
     * hands using MediaPipe Hand Landmarker.
     */
    HOLISTIC_HANDS("Ręce"),

    /**
     * Detects and tracks face-mesh landmarks (468 points) using
     * MediaPipe Face Landmarker.
     */
    HOLISTIC_FACE("Twarz"),

    /**
     * Tracks iris and eye landmarks (478 refined face points including
     * 10 iris-specific landmarks) using MediaPipe Face Landmarker with
     * iris refinement enabled.
     */
    IRIS("Wzrok (Iris)"),

    /**
     * Renders a 3D hologram object (wireframe cube) whose rotation is driven
     * by the viewer's face position relative to the screen centre.
     *
     * Uses MediaPipe Face Landmarker to detect the nose-tip landmark and maps
     * its normalised X/Y offset from the frame centre to yaw and pitch angles.
     * The resulting rotation is applied to a cube rendered via perspective
     * projection, creating the illusion of a holographic display that responds
     * to the viewer's gaze.
     */
    HOLOGRAM_3D("Hologram 3D"),

    /** Estimates sparse monocular visual odometry from tracked feature points. */
    VISUAL_ODOMETRY("Odometria wizyjna"),

    /** Builds a pseudo 3D point cloud view from frame-to-frame parallax. */
    POINT_CLOUD("Chmura punktów"),

    /**
     * Detects planar surfaces using vanishing-point analysis and RANSAC-based
     * line clustering.  Detected planes are highlighted with semi-transparent
     * colour overlays and normal-direction arrows.
     */
    PLANE_DETECTION("Detekcja płaszczyzn"),

    /**
     * Detects vanishing points from groups of parallel line segments and
     * visualizes them with converging-line bundles and point markers.
     */
    VANISHING_POINTS("Punkty zbieżności"),

    /** Median blur filter. */
    MEDIAN_BLUR("Median Blur"),

    /** Bilateral filter. */
    BILATERAL_FILTER("Bilateral Filter"),

    /** Box blur filter. */
    BOX_FILTER("Box Filter"),

    /** Adaptive thresholding. */
    ADAPTIVE_THRESHOLD("Adaptive Threshold"),

    /** Histogram equalization. */
    HISTOGRAM_EQUALIZATION("Histogram Equalization"),

    /** Scharr operator for edge detection. */
    SCHARR("Scharr Edges"),

    /** Prewitt operator (simulated via custom kernel). */
    PREWITT("Prewitt Edges"),

    /** Roberts cross operator (simulated via custom kernel). */
    ROBERTS("Roberts Edges"),

    // ------------------------------------------------------------------
    // Artistic / visual effects
    // ------------------------------------------------------------------

    /** Inverts all colour channels (bitwise NOT). */
    INVERT("Inwersja"),

    /**
     * Applies a warm sepia-tone look by blending the grayscale image with
     * predefined brown-tinted channel weights.
     */
    SEPIA("Sepia"),

    /**
     * Emboss / relief effect that accentuates surface texture using a
     * directional convolution kernel.
     */
    EMBOSS("Emboss"),

    /**
     * Reduces apparent resolution by downsampling and then upsampling the
     * frame, creating a blocky pixel-art appearance.
     */
    PIXELATE("Pikselizacja"),

    /**
     * Cartoon / comic-book effect produced by combining a colour-smoothed
     * image (bilateral filter) with thick Canny edges drawn on top.
     */
    CARTOON("Kreskówka"),

    // ------------------------------------------------------------------
    // YOLO object detection
    // ------------------------------------------------------------------

    /**
     * Detects objects in the current frame using YOLOv8-nano (detect variant).
     * Draws bounding boxes with class labels and confidence scores for all
     * 80 COCO classes.
     */
    YOLO_DETECT("YOLO Detect"),

    /**
     * YOLOv8-nano with Kalman filter tracking to improve bounding box stability.
     */
    YOLO_KALMAN("YOLO + Kalman"),

    /**
     * Runs YOLOv8-nano instance segmentation, overlaying bounding boxes for
     * detected objects.  Full pixel-level mask rendering is approximated with
     * the bounding box due to CPU performance constraints.
     */
    YOLO_SEGMENT("YOLO Segment"),

    /**
     * Estimates 17-keypoint human body poses using YOLOv8-nano pose variant.
     * Draws bounding boxes for each detected person and connects the skeleton
     * keypoints with coloured lines.
     */
    YOLO_POSE("YOLO Pose"),

    /**
     * Classifies the entire camera frame into one of 1 000 ImageNet categories
     * using YOLOv8-nano classify variant.
     * Displays the top-5 predicted classes with their softmax confidence scores.
     */
    YOLO_CLASSIFY("YOLO Classify"),

    /**
     * Detects objects with oriented (rotated) bounding boxes using YOLOv8-nano
     * OBB variant trained on the DOTAv1 dataset (15 aerial-imagery classes).
     * Draws rotated rectangles aligned to each detected object's principal axis.
     */
    YOLO_OBB("YOLO OBB"),

    // ------------------------------------------------------------------
    // RTMDet object detection
    // ------------------------------------------------------------------

    /**
     * Detects objects using RTMDet-nano (OpenMMLab) running on the OpenCV DNN
     * backend.  Draws axis-aligned bounding boxes with class labels and
     * confidence scores for all 80 COCO classes.
     */
    RTMDET_DETECT("RTMDet Detect"),

    /**
     * Detects objects with oriented (rotated) bounding boxes using
     * RTMDet-nano-r (RTMDet-Rotated) running on the OpenCV DNN backend.
     * Each detection is rendered as a rotated rectangle together with its
     * class label and rotation angle.
     */
    RTMDET_ROTATED("RTMDet Rotated"),

    // ------------------------------------------------------------------
    // Full 3-D odometry
    // ------------------------------------------------------------------

    /**
     * Full monocular visual odometry: shows live optical-flow tracks overlaid
     * on the camera image together with a HUD displaying the accumulated frame
     * count, RANSAC inlier ratio, and current world-frame pose coordinates.
     */
    FULL_ODOMETRY("Pełna odometria"),

    /**
     * Accumulated camera trajectory visualised as a top-down (X-Z plane)
     * bird's-eye view on a dark canvas.  Each dot represents a camera position
     * estimate; the current position is highlighted in red with a cross-hair.
     */
    ODOMETRY_TRAJECTORY("Trajektoria 3D"),

    /**
     * Sparse 3-D map built by triangulating inlier feature correspondences
     * across frames.  Points are projected onto the X-Z (top-down) plane
     * together with the current camera position marker.
     */
    ODOMETRY_MAP("Mapa 3D"),
}

/**
 * Returns ``true`` if this filter requires the MediaPipe processing pipeline
 * rather than the OpenCV pipeline.
 */
val OpenCvFilter.isMediaPipe: Boolean
    get() = this == OpenCvFilter.HOLISTIC_BODY ||
        this == OpenCvFilter.HOLISTIC_HANDS ||
        this == OpenCvFilter.HOLISTIC_FACE ||
        this == OpenCvFilter.IRIS ||
        this == OpenCvFilter.HOLOGRAM_3D

/**
 * Returns ``true`` if this filter requires the YOLO processing pipeline
 * backed by the OpenCV DNN module.
 */
val OpenCvFilter.isYolo: Boolean
    get() = this == OpenCvFilter.YOLO_DETECT ||
        this == OpenCvFilter.YOLO_KALMAN ||
        this == OpenCvFilter.YOLO_SEGMENT ||
        this == OpenCvFilter.YOLO_POSE ||
        this == OpenCvFilter.YOLO_CLASSIFY ||
        this == OpenCvFilter.YOLO_OBB

/**
 * Returns ``true`` if this filter requires the RTMDet processing pipeline
 * backed by the PyTorch Mobile library.
 */
val OpenCvFilter.isRtmDet: Boolean
    get() = this == OpenCvFilter.RTMDET_DETECT ||
        this == OpenCvFilter.RTMDET_ROTATED

/**
 * Returns ``true`` if this filter belongs to the full 3-D odometry pipeline
 * and should keep the [FullOdometryEngine] alive between frames.
 */
val OpenCvFilter.isFullOdometry: Boolean
    get() = this == OpenCvFilter.FULL_ODOMETRY ||
        this == OpenCvFilter.ODOMETRY_TRAJECTORY ||
        this == OpenCvFilter.ODOMETRY_MAP
