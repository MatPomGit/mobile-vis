package pl.edu.mobilecv

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.media.MediaRecorder
import android.os.Build
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.util.Log
import java.util.concurrent.locks.ReentrantLock

/**
 * Records processed (annotated) camera frames to an MP4 video file in MediaStore.
 *
 * Thread-safety contract:
 * - [start] – call from the **main thread** to prepare a new recording session.
 * - [writeFrame] – call from the **analysis thread** for every processed frame.
 * - [finalize] – call from the **analysis thread** (via executor.submit) to finish recording.
 *
 * Audio is captured from the microphone in a dedicated background thread.
 * Video and audio muxer writes are serialised with [muxerLock].
 */
class ProcessedVideoRecorder(private val context: Context) {

    companion object {
        private const val TAG = "ProcessedVideoRecorder"
        private const val MIME_TYPE_VIDEO = MediaFormat.MIMETYPE_VIDEO_AVC
        private const val MIME_TYPE_AUDIO = MediaFormat.MIMETYPE_AUDIO_AAC
        private const val VIDEO_BIT_RATE = 4_000_000
        private const val VIDEO_FRAME_RATE = 30
        private const val VIDEO_I_FRAME_INTERVAL = 1
        private const val AUDIO_SAMPLE_RATE = 44_100
        private const val AUDIO_BIT_RATE = 128_000
        private const val AUDIO_CHANNELS = 1
        private const val CODEC_TIMEOUT_US = 10_000L
    }

    /** True while a recording session is active (set to false on [finalize]). */
    @Volatile var active = false
        private set

    private var videoEncoder: MediaCodec? = null
    private var audioEncoder: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var parcelFd: ParcelFileDescriptor? = null
    private var pendingUri: android.net.Uri? = null
    private var audioRecord: AudioRecord? = null
    private var audioThread: Thread? = null

    private var videoTrackIndex = -1
    private var audioTrackIndex = -1
    @Volatile private var muxerStarted = false
    @Volatile private var videoFormatReady = false
    @Volatile private var audioFormatReady = false // set to true immediately when audio is disabled

    @Volatile private var startTimeUs = 0L
    private var withAudio = false

    private val videoBufferInfo = MediaCodec.BufferInfo()
    private val audioBufferInfo = MediaCodec.BufferInfo()
    private val muxerLock = ReentrantLock()

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Prepares a new recording session. Call from the **main thread** before the
     * first [writeFrame] call. Returns false if the MediaStore entry could not be created.
     */
    fun start(withAudio: Boolean): Boolean {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "VID_${System.currentTimeMillis()}")
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/MobileCV")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            } else {
                @Suppress("DEPRECATION")
                put(
                    MediaStore.Video.Media.DATA,
                    "${android.os.Environment.getExternalStoragePublicDirectory(
                        android.os.Environment.DIRECTORY_MOVIES
                    )}/MobileCV/VID_${System.currentTimeMillis()}.mp4",
                )
            }
        }
        val uri = context.contentResolver.insert(
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            values,
        ) ?: run {
            Log.e(TAG, "Failed to create MediaStore entry")
            return false
        }
        pendingUri = uri
        this.withAudio = withAudio
        active = true
        muxerStarted = false
        videoFormatReady = false
        audioFormatReady = !withAudio
        videoTrackIndex = -1
        audioTrackIndex = -1
        startTimeUs = 0L
        return true
    }

    /**
     * Encodes a processed [bitmap] frame. Call from the **analysis thread**.
     * Initialises the encoders lazily on the first call.
     */
    fun writeFrame(bitmap: Bitmap) {
        if (!active) return
        if (videoEncoder == null) {
            if (!initVideoEncoder(bitmap.width, bitmap.height)) {
                Log.e(TAG, "Failed to initialize video encoder; aborting recording")
                active = false
                return
            }
            if (withAudio) {
                if (initAudioEncoder()) startAudioCapture() else audioFormatReady = true
            }
        }
        encodeVideoFrame(bitmap)
        drainVideoEncoder(endOfStream = false)
    }

    /**
     * Finalises the recording: signals end-of-stream, drains remaining data,
     * publishes the file to MediaStore, and invokes [onComplete] with a success flag.
     * Must be called from the **analysis thread** (via executor.submit) so that it
     * runs after the last in-progress [writeFrame].
     */
    fun finalize(onComplete: (success: Boolean) -> Unit) {
        if (!active) { onComplete(false); return }
        active = false
        var success = false
        try {
            drainVideoEncoder(endOfStream = true)
            audioThread?.join(3_000)
            muxerLock.lock()
            try {
                if (muxerStarted) muxer?.stop()
            } finally {
                muxerLock.unlock()
            }
            publishMediaStoreEntry()
            success = true
        } catch (e: Exception) {
            Log.e(TAG, "finalize failed", e)
        } finally {
            release()
        }
        onComplete(success)
    }

    // -------------------------------------------------------------------------
    // Encoder init
    // -------------------------------------------------------------------------

    private fun initVideoEncoder(width: Int, height: Int): Boolean {
        // YUV encoders require even dimensions
        val encWidth = if (width % 2 == 0) width else width - 1
        val encHeight = if (height % 2 == 0) height else height - 1
        return try {
            val format = MediaFormat.createVideoFormat(MIME_TYPE_VIDEO, encWidth, encHeight).apply {
                setInteger(
                    MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible,
                )
                setInteger(MediaFormat.KEY_BIT_RATE, VIDEO_BIT_RATE)
                setInteger(MediaFormat.KEY_FRAME_RATE, VIDEO_FRAME_RATE)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, VIDEO_I_FRAME_INTERVAL)
            }
            val encoder = MediaCodec.createEncoderByType(MIME_TYPE_VIDEO)
            encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder.start()
            videoEncoder = encoder

            val pfd = context.contentResolver.openFileDescriptor(pendingUri!!, "w")
                ?: throw IllegalStateException("Cannot open output FD")
            parcelFd = pfd
            muxer = MediaMuxer(pfd.fileDescriptor, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            true
        } catch (e: Exception) {
            Log.e(TAG, "initVideoEncoder failed", e)
            false
        }
    }

    private fun initAudioEncoder(): Boolean {
        return try {
            val format = MediaFormat.createAudioFormat(
                MIME_TYPE_AUDIO, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS,
            ).apply {
                setInteger(MediaFormat.KEY_BIT_RATE, AUDIO_BIT_RATE)
                setInteger(
                    MediaFormat.KEY_AAC_PROFILE,
                    MediaCodecInfo.CodecProfileLevel.AACObjectLC,
                )
                setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16384)
            }
            val encoder = MediaCodec.createEncoderByType(MIME_TYPE_AUDIO)
            encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder.start()
            audioEncoder = encoder
            true
        } catch (e: Exception) {
            Log.e(TAG, "initAudioEncoder failed", e)
            false
        }
    }

    // -------------------------------------------------------------------------
    // Audio capture thread
    // -------------------------------------------------------------------------

    @SuppressLint("MissingPermission")
    private fun startAudioCapture() {
        val encoder = audioEncoder ?: return
        val minBufSize = AudioRecord.getMinBufferSize(
            AUDIO_SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                AUDIO_SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBufSize, 4096),
            )
        } catch (e: Exception) {
            Log.e(TAG, "AudioRecord init failed", e)
            audioFormatReady = true
            return
        }
        audioRecord = record
        record.startRecording()

        val thread = Thread {
            val pcm = ByteArray(4096)
            try {
                while (active) {
                    val read = record.read(pcm, 0, pcm.size)
                    if (read > 0) {
                        feedAudioPcm(encoder, pcm, read)
                        drainAudioEncoder(endOfStream = false)
                        tryStartMuxer()
                    }
                }
            } finally {
                feedAudioEos(encoder)
                drainAudioEncoder(endOfStream = true)
                record.stop()
                record.release()
                audioRecord = null
            }
        }
        thread.name = "AudioCapture"
        audioThread = thread
        thread.start()
    }

    private fun feedAudioPcm(encoder: MediaCodec, pcm: ByteArray, size: Int) {
        val idx = encoder.dequeueInputBuffer(CODEC_TIMEOUT_US)
        if (idx >= 0) {
            val buf = encoder.getInputBuffer(idx) ?: return
            val ts = timestampUs()
            buf.clear()
            buf.put(pcm, 0, size)
            encoder.queueInputBuffer(idx, 0, size, ts, 0)
        }
    }

    private fun feedAudioEos(encoder: MediaCodec) {
        val idx = encoder.dequeueInputBuffer(CODEC_TIMEOUT_US)
        if (idx >= 0) {
            encoder.queueInputBuffer(idx, 0, 0, timestampUs(), MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        }
    }

    // -------------------------------------------------------------------------
    // Video frame encoding
    // -------------------------------------------------------------------------

    private fun encodeVideoFrame(bitmap: Bitmap) {
        val encoder = videoEncoder ?: return
        val idx = encoder.dequeueInputBuffer(CODEC_TIMEOUT_US)
        if (idx < 0) return // encoder busy; drop frame

        val image = encoder.getInputImage(idx)
        val ts = timestampUs()
        if (image != null) {
            writeBitmapToImage(bitmap, image)
            encoder.queueInputBuffer(idx, 0, 0, ts, 0)
        } else {
            // Fallback: ByteBuffer with NV12
            val buf = encoder.getInputBuffer(idx) ?: return
            val nv12 = bitmapToNv12(bitmap)
            buf.clear()
            buf.put(nv12)
            encoder.queueInputBuffer(idx, 0, nv12.size, ts, 0)
        }
    }

    /**
     * Writes [bitmap] pixel data into the YUV [android.media.Image] planes.
     * Handles both NV12 (pixelStride=2 for UV) and I420 (pixelStride=1) layouts.
     */
    private fun writeBitmapToImage(bitmap: Bitmap, image: android.media.Image) {
        val w = bitmap.width.coerceAtMost(image.width)
        val h = bitmap.height.coerceAtMost(image.height)
        val argb = IntArray(w * h)
        bitmap.getPixels(argb, 0, w, 0, 0, w, h)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val yStride = yPlane.rowStride
        val uvStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        for (j in 0 until h) {
            val srcRow = j * w
            val dstRow = j * yStride
            for (i in 0 until w) {
                val p = argb[srcRow + i]
                val r = (p shr 16) and 0xff
                val g = (p shr 8) and 0xff
                val b = p and 0xff
                yBuf.put(dstRow + i, (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).toByte())
            }
        }

        for (j in 0 until h / 2) {
            val srcRow = j * 2 * w
            val dstRow = j * uvStride
            for (i in 0 until w / 2) {
                val p = argb[srcRow + i * 2]
                val r = (p shr 16) and 0xff
                val g = (p shr 8) and 0xff
                val b = p and 0xff
                val u = (((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128).coerceIn(0, 255)
                val v = (((112 * r - 94 * g - 18 * b + 128) shr 8) + 128).coerceIn(0, 255)
                val pos = dstRow + i * uvPixelStride
                uBuf.put(pos, u.toByte())
                vBuf.put(pos, v.toByte())
            }
        }
    }

    /** Converts [bitmap] to NV12 byte array (Y plane followed by interleaved UV). */
    private fun bitmapToNv12(bitmap: Bitmap): ByteArray {
        val w = bitmap.width
        val h = bitmap.height
        val argb = IntArray(w * h)
        bitmap.getPixels(argb, 0, w, 0, 0, w, h)
        val nv12 = ByteArray(w * h * 3 / 2)
        for (j in 0 until h) {
            val srcRow = j * w
            val dstRow = j * w
            for (i in 0 until w) {
                val p = argb[srcRow + i]
                val r = (p shr 16) and 0xff
                val g = (p shr 8) and 0xff
                val b = p and 0xff
                nv12[dstRow + i] = (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).toByte()
            }
        }
        val uvOffset = w * h
        var uvIdx = 0
        for (j in 0 until h / 2) {
            val srcRow = j * 2 * w
            for (i in 0 until w / 2) {
                val p = argb[srcRow + i * 2]
                val r = (p shr 16) and 0xff
                val g = (p shr 8) and 0xff
                val b = p and 0xff
                nv12[uvOffset + uvIdx] =
                    (((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128).coerceIn(0, 255).toByte()
                nv12[uvOffset + uvIdx + 1] =
                    (((112 * r - 94 * g - 18 * b + 128) shr 8) + 128).coerceIn(0, 255).toByte()
                uvIdx += 2
            }
        }
        return nv12
    }

    // -------------------------------------------------------------------------
    // Encoder draining
    // -------------------------------------------------------------------------

    private fun drainVideoEncoder(endOfStream: Boolean) {
        val encoder = videoEncoder ?: return
        if (endOfStream) {
            // Signal EOS via the input buffer (ByteBuffer mode)
            val idx = encoder.dequeueInputBuffer(CODEC_TIMEOUT_US)
            if (idx >= 0) {
                encoder.queueInputBuffer(idx, 0, 0, timestampUs(), MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            }
        }
        while (true) {
            val idx = encoder.dequeueOutputBuffer(videoBufferInfo, CODEC_TIMEOUT_US)
            when {
                idx == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (!videoFormatReady) {
                        muxerLock.lock()
                        try {
                            videoTrackIndex = muxer!!.addTrack(encoder.outputFormat)
                            videoFormatReady = true
                        } finally {
                            muxerLock.unlock()
                        }
                        tryStartMuxer()
                    }
                }
                idx >= 0 -> {
                    val buf = encoder.getOutputBuffer(idx)
                    if (buf != null &&
                        videoBufferInfo.size > 0 &&
                        videoBufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0
                    ) {
                        if (muxerStarted) {
                            buf.position(videoBufferInfo.offset)
                            buf.limit(videoBufferInfo.offset + videoBufferInfo.size)
                            muxerLock.lock()
                            try {
                                muxer!!.writeSampleData(videoTrackIndex, buf, videoBufferInfo)
                            } finally {
                                muxerLock.unlock()
                            }
                        }
                    }
                    encoder.releaseOutputBuffer(idx, false)
                    if (videoBufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
        }
    }

    private fun drainAudioEncoder(endOfStream: Boolean) {
        val encoder = audioEncoder ?: return
        while (true) {
            val idx = encoder.dequeueOutputBuffer(audioBufferInfo, CODEC_TIMEOUT_US)
            when {
                idx == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (!audioFormatReady) {
                        muxerLock.lock()
                        try {
                            audioTrackIndex = muxer!!.addTrack(encoder.outputFormat)
                            audioFormatReady = true
                        } finally {
                            muxerLock.unlock()
                        }
                        tryStartMuxer()
                    }
                }
                idx >= 0 -> {
                    val buf = encoder.getOutputBuffer(idx)
                    if (buf != null &&
                        audioBufferInfo.size > 0 &&
                        audioBufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0
                    ) {
                        if (muxerStarted) {
                            buf.position(audioBufferInfo.offset)
                            buf.limit(audioBufferInfo.offset + audioBufferInfo.size)
                            muxerLock.lock()
                            try {
                                muxer!!.writeSampleData(audioTrackIndex, buf, audioBufferInfo)
                            } finally {
                                muxerLock.unlock()
                            }
                        }
                    }
                    encoder.releaseOutputBuffer(idx, false)
                    if (audioBufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private fun tryStartMuxer() {
        if (muxerStarted || !videoFormatReady || !audioFormatReady) return
        muxerLock.lock()
        try {
            if (muxerStarted) return
            muxer!!.start()
            muxerStarted = true
        } finally {
            muxerLock.unlock()
        }
    }

    private fun timestampUs(): Long {
        val now = System.nanoTime() / 1000L
        if (startTimeUs == 0L) startTimeUs = now
        return now - startTimeUs
    }

    private fun publishMediaStoreEntry() {
        parcelFd?.close()
        parcelFd = null
        val uri = pendingUri ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val update = ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) }
            context.contentResolver.update(uri, update, null, null)
        }
    }

    private fun release() {
        try { videoEncoder?.stop() } catch (_: Exception) {}
        try { audioEncoder?.stop() } catch (_: Exception) {}
        videoEncoder?.release()
        audioEncoder?.release()
        audioRecord?.release()
        try { muxer?.release() } catch (_: Exception) {}
        parcelFd?.close()
        videoEncoder = null
        audioEncoder = null
        audioRecord = null
        muxer = null
        parcelFd = null
        pendingUri = null
    }
}
