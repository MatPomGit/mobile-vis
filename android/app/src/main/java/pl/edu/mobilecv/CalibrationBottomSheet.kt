package pl.edu.mobilecv

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import pl.edu.mobilecv.databinding.BottomSheetCalibrationBinding
import androidx.core.graphics.toColorInt
import pl.edu.mobilecv.vision.CameraCalibrator

/**
 * Menu (bottom-sheet) do kalibracji kamery za pomocą szachownicy.
 *
 * Zapewnia informację zwrotną w czasie rzeczywistym (wykrycie szachownicy, liczba klatek)
 * i udostępnia trzy akcje użytkownika poprzez lambdy podpięte przez [MainActivity]:
 *
 * - [onCollectFrame]  – zapisanie bieżącego zestawu wykrytych narożników.
 * - [onCalibrate]     – obliczenie parametrów kalibracji z zebranych klatek.
 * - [onReset]         – wyczyszczenie wszystkich zebranych klatek i danych kalibracji.
 */
class CalibrationBottomSheet : BottomSheetDialogFragment() {

    private var _binding: BottomSheetCalibrationBinding? = null
    private val binding get() = _binding!!

    private val handler = Handler(Looper.getMainLooper())

    // Callbacki ustawiane przez aktywność hostującą.
    var onCollectFrame: (() -> Boolean)? = null
    var onCalibrate: (() -> CameraCalibrator.CalibrationData?)? = null
    var onReset: (() -> Unit)? = null

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

    /**
     * Odświeża interfejs użytkownika, aby odzwierciedlić bieżący stan kalibracji.
     */
    fun updateStatus(
        cornersDetected: Boolean,
        frameCount: Int,
        calibrationData: CameraCalibrator.CalibrationData?,
    ) {
        val b = _binding ?: return

        // Wskaźnik wykrywania (zielony = znaleziono, szary = szukanie)
        b.viewDetectionIndicator.setBackgroundColor(
            if (cornersDetected) "#4CAF50".toColorInt() else "#9E9E9E".toColorInt()
        )
        b.tvDetectionStatus.setText(
            if (cornersDetected) R.string.calibration_status_found
            else R.string.calibration_status_searching
        )

        // Postęp zbierania klatek
        val max = CameraCalibrator.MIN_FRAMES
        b.progressFrames.max = max
        b.progressFrames.progress = frameCount.coerceAtMost(max)
        b.tvFrameCount.text = getString(R.string.calibration_frames_count, frameCount, max)

        // Stany przycisków
        b.btnCollectFrame.isEnabled = cornersDetected
        b.btnComputeCalibration.isEnabled = frameCount >= max

        // Sekcja wyników
        if (calibrationData != null) {
            b.layoutCalibrationResults.visibility = View.VISIBLE
            b.tvCalibrationResults.text = calibrationData.summary()
        } else {
            b.layoutCalibrationResults.visibility = View.GONE
        }
    }

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

    private fun scheduleStatusRefresh() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                _binding ?: return
                // Bezpieczny dostęp do aktywności - unikanie requireActivity(), które rzuca wyjątek po odpięciu
                val act = activity as? MainActivity ?: return
                val cal = act.cameraCalibrator
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

        /** Interwał odświeżania UI w milisekundach. */
        private const val REFRESH_INTERVAL_MS = 250L
    }
}
