package pl.edu.mobilecv

import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import pl.edu.mobilecv.databinding.BottomSheetCalibrationBinding

/**
 * Bottom-sheet menu for chessboard camera calibration.
 *
 * Provides real-time feedback (chessboard detection, frame count) and
 * exposes three user actions via lambdas wired by [MainActivity]:
 *
 * - [onCollectFrame]  – save the current set of detected corners.
 * - [onCalibrate]     – compute calibration from collected frames.
 * - [onReset]         – clear all collected frames and calibration data.
 *
 * The [updateStatus] method should be called periodically to refresh the
 * detection indicator and frame counter while the sheet is open.
 */
class CalibrationBottomSheet : BottomSheetDialogFragment() {

    private var _binding: BottomSheetCalibrationBinding? = null
    private val binding get() = _binding!!

    private val handler = Handler(Looper.getMainLooper())

    // Callbacks wired by the host activity.
    var onCollectFrame: (() -> Boolean)? = null
    var onCalibrate: (() -> CameraCalibrator.CalibrationData?)? = null
    var onReset: (() -> Unit)? = null

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View {
        _binding = BottomSheetCalibrationBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupButtons()
        scheduleStatusRefresh()
    }

    override fun onDestroyView() {
        handler.removeCallbacksAndMessages(null)
        _binding = null
        super.onDestroyView()
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Refresh the UI to reflect the current calibration state.
     *
     * Must be called on the main thread.
     *
     * @param cornersDetected Whether the chessboard is currently visible.
     * @param frameCount      Number of collected frames.
     * @param calibrationData Non-null when calibration has been computed.
     */
    fun updateStatus(
        cornersDetected: Boolean,
        frameCount: Int,
        calibrationData: CameraCalibrator.CalibrationData?,
    ) {
        val b = _binding ?: return

        // Detection indicator (green = found, grey = not found)
        b.viewDetectionIndicator.setBackgroundColor(
            if (cornersDetected) Color.parseColor("#4CAF50") else Color.parseColor("#9E9E9E")
        )
        b.tvDetectionStatus.setText(
            if (cornersDetected) R.string.calibration_status_found
            else R.string.calibration_status_searching
        )

        // Frame progress
        val max = CameraCalibrator.MIN_FRAMES
        b.progressFrames.max = max
        b.progressFrames.progress = frameCount.coerceAtMost(max)
        b.tvFrameCount.text = getString(R.string.calibration_frames_count, frameCount, max)

        // Button states
        b.btnCollectFrame.isEnabled = cornersDetected
        b.btnComputeCalibration.isEnabled = frameCount >= max

        // Results section
        if (calibrationData != null) {
            b.layoutCalibrationResults.visibility = View.VISIBLE
            b.tvCalibrationResults.text = calibrationData.summary()
        } else {
            b.layoutCalibrationResults.visibility = View.GONE
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    private fun setupButtons() {
        binding.btnCollectFrame.setOnClickListener {
            val collected = onCollectFrame?.invoke() ?: false
            if (!collected) {
                binding.btnCollectFrame.isEnabled = false
            }
        }

        binding.btnComputeCalibration.setOnClickListener {
            val result = onCalibrate?.invoke()
            if (result != null) {
                binding.layoutCalibrationResults.visibility = View.VISIBLE
                binding.tvCalibrationResults.text = result.summary()
            }
        }

        binding.btnResetCalibration.setOnClickListener {
            onReset?.invoke()
            binding.layoutCalibrationResults.visibility = View.GONE
            binding.progressFrames.progress = 0
            binding.tvFrameCount.text =
                getString(R.string.calibration_frames_count, 0, CameraCalibrator.MIN_FRAMES)
            binding.btnComputeCalibration.isEnabled = false
        }
    }

    /**
     * Poll the calibrator state every [REFRESH_INTERVAL_MS] milliseconds
     * while the sheet is visible.  Stopped in [onDestroyView].
     */
    private fun scheduleStatusRefresh() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                if (_binding == null) return
                val activity = requireActivity() as? MainActivity ?: return
                val cal = activity.cameraCalibrator
                updateStatus(
                    cornersDetected = cal.lastCornersDetected,
                    frameCount = cal.frameCount,
                    calibrationData = cal.calibrationResult,
                )
                handler.postDelayed(this, REFRESH_INTERVAL_MS)
            }
        }, REFRESH_INTERVAL_MS)
    }

    companion object {
        const val TAG = "CalibrationBottomSheet"

        /** UI refresh interval in milliseconds. */
        private const val REFRESH_INTERVAL_MS = 250L
    }
}
