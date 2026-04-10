package pl.edu.mobilecv

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import android.view.View
import androidx.core.content.ContextCompat
import pl.edu.mobilecv.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService

/**
 * Kontroler zarządzający nagrywaniem oraz timerem UI.
 */
class RecordingController(
    private val context: Context,
    private val binding: ActivityMainBinding,
    private val recorder: ProcessedVideoRecorder,
    private val backgroundExecutor: ExecutorService,
    private val callbacks: Callbacks,
) {
    interface Callbacks {
        /** Zwraca informację, czy Activity nadal może aktualizować UI. */
        fun canUpdateUi(): Boolean

        /** Zwraca string z zasobów dla podanego id. */
        fun stringRes(id: Int): String

        /** Pokazuje krótką wiadomość użytkownikowi. */
        fun showToast(messageRes: Int)
    }

    companion object {
        private const val RECORDING_TIMER_FORMAT = "%02d:%02d"
    }

    private val recordingTimerHandler = Handler(Looper.getMainLooper())
    private var recordingTimerRunnable: Runnable? = null
    private var recordingStartTimeMs: Long = 0L

    @Volatile
    var isRecording: Boolean = false
        private set

    fun onDestroy() {
        if (isRecording) {
            isRecording = false
            backgroundExecutor.execute {
                recorder.finalize { }
            }
        }
        stopRecordingTimer()
    }

    fun writeFrame(frame: android.graphics.Bitmap) {
        if (isRecording) {
            recorder.writeFrame(frame)
        }
    }

    @SuppressLint("MissingPermission")
    fun startRecording() {
        val hasAudio = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        val started = recorder.start(withAudio = hasAudio)
        if (!started) {
            callbacks.showToast(R.string.video_error)
            return
        }
        isRecording = true
        updateCaptureButtonState()
    }

    fun stopRecording() {
        if (!isRecording) {
            return
        }
        isRecording = false
        updateCaptureButtonState()
        backgroundExecutor.submit {
            recorder.finalize { success ->
                if (callbacks.canUpdateUi()) {
                    callbacks.showToast(if (success) R.string.video_saved else R.string.video_error)
                }
            }
        }
    }

    fun bindCaptureButton(onShortClickCapture: () -> Unit) {
        binding.btnCapture.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                onShortClickCapture()
            }
        }
        binding.btnCapture.setOnLongClickListener {
            if (!isRecording) {
                startRecording()
            }
            true
        }
    }

    private fun startRecordingTimer() {
        recordingStartTimeMs = System.currentTimeMillis()
        val runnable = object : Runnable {
            override fun run() {
                val elapsed = (System.currentTimeMillis() - recordingStartTimeMs) / 1000L
                binding.textViewRecordingTimer.text = RECORDING_TIMER_FORMAT.format(
                    elapsed / 60,
                    elapsed % 60,
                )
                recordingTimerHandler.postDelayed(this, 1000)
            }
        }
        recordingTimerRunnable = runnable
        recordingTimerHandler.post(runnable)
    }

    private fun stopRecordingTimer() {
        recordingTimerRunnable?.let { recordingTimerHandler.removeCallbacks(it) }
        recordingTimerRunnable = null
        binding.textViewRecordingTimer.text = RECORDING_TIMER_FORMAT.format(0, 0)
    }

    private fun updateCaptureButtonState() {
        binding.btnCapture.isActivated = isRecording
        binding.btnCapture.contentDescription = callbacks.stringRes(
            if (isRecording) R.string.stop_recording_description else R.string.capture_button_description,
        )
        if (isRecording) {
            binding.layoutRecordingIndicator.visibility = View.VISIBLE
            startRecordingTimer()
        } else {
            binding.layoutRecordingIndicator.visibility = View.GONE
            stopRecordingTimer()
        }
    }
}
