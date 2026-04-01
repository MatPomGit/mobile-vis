package pl.edu.mobilecv

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * WebSocket client for the ROSBridge v2 protocol (rosbridge_suite).
 *
 * Connects to a `rosbridge_server` running on the robot and publishes
 * marker detection results to a ROS2 topic using JSON messages.
 *
 * The standard rosbridge port is **9090**.  Messages follow the ROSBridge
 * protocol specification:
 * ```json
 * {"op": "advertise", "topic": "/mobilecv/detections", "type": "std_msgs/String"}
 * {"op": "publish",   "topic": "/mobilecv/detections", "msg": {"data": "…"}}
 * ```
 *
 * Usage:
 * ```kotlin
 * val client = RosBridgeClient()
 * client.onStateChanged = { state -> updateUi(state) }
 * client.connect("192.168.1.100", 9090)
 * // … later …
 * client.publishMarkers(detections)
 * client.disconnect()
 * ```
 *
 * All [onStateChanged] callbacks are invoked on OkHttp internal threads;
 * callers must switch to the main thread if they update UI elements.
 */
class RosBridgeClient {

    /** Connection state reported to [onStateChanged]. */
    enum class State { DISCONNECTED, CONNECTING, CONNECTED, ERROR }

    companion object {
        private const val TAG = "RosBridgeClient"

        /** Default ROSBridge WebSocket port. */
        const val DEFAULT_PORT = 9090

        /** ROS2 topic for publishing marker detections. */
        private const val TOPIC_DETECTIONS = "/mobilecv/detections"

        /** ROS2 message type used for detections (JSON payload in data field). */
        private const val MSG_TYPE_STRING = "std_msgs/String"

        private const val CONNECT_TIMEOUT_SECONDS = 10L

        /** Minimum interval between successive publish calls: 100 ms = max 10 Hz. */
        private const val MIN_PUBLISH_INTERVAL_MS = 100L
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS) // no read timeout for persistent WebSocket
        .build()

    private var webSocket: WebSocket? = null

    /**
     * Minimum interval between successive [publishMarkers] calls in milliseconds.
     * Limits publish rate to ~10 Hz to avoid saturating the network link.
     */
    private var lastPublishMs: Long = 0

    /** Called on connection state changes.  Invoked on OkHttp threads. */
    var onStateChanged: ((State) -> Unit)? = null

    /** Current connection state. */
    @Volatile
    var state: State = State.DISCONNECTED
        private set

    /**
     * Open a WebSocket connection to the rosbridge_server.
     *
     * Does nothing when already [State.CONNECTING] or [State.CONNECTED].
     *
     * @param host Robot IP address or hostname (e.g. "192.168.1.100").
     * @param port rosbridge_server port (default [DEFAULT_PORT] = 9090).
     */
    fun connect(host: String, port: Int = DEFAULT_PORT) {
        if (state == State.CONNECTING || state == State.CONNECTED) {
            Log.w(TAG, "connect() called while already in state $state – ignoring")
            return
        }
        updateState(State.CONNECTING)

        val url = "ws://$host:$port"
        Log.i(TAG, "Connecting to $url")
        val request = Request.Builder().url(url).build()
        webSocket = httpClient.newWebSocket(request, RosBridgeWebSocketListener())
    }

    /**
     * Close the active WebSocket connection gracefully.
     *
     * After this call [state] transitions to [State.DISCONNECTED].
     * The client can be reconnected by calling [connect] again.
     */
    fun disconnect() {
        webSocket?.close(1000, "User disconnect")
        webSocket = null
        updateState(State.DISCONNECTED)
    }

    /**
     * Release all resources held by this client.
     *
     * Must be called when the client is permanently no longer needed
     * (e.g. in [android.app.Activity.onDestroy]).  After this call the
     * client cannot be reused.
     */
    fun shutdown() {
        disconnect()
        httpClient.dispatcher.executorService.shutdown()
        httpClient.connectionPool.evictAll()
    }

    /**
     * Publish a list of marker detections to the [TOPIC_DETECTIONS] ROS2 topic.
     *
     * Does nothing when [state] is not [State.CONNECTED], the list is empty,
     * or the minimum publish interval has not elapsed (rate-limited to ~10 Hz).
     *
     * The payload is a JSON array serialised into the `data` field of a
     * `std_msgs/String` message so that any ROS2 node can parse it:
     * ```json
     * [{"type":"apriltag","id":0,"corners":[…],"timestamp_ms":…}, …]
     * ```
     *
     * @param detections Marker detections collected during the current frame.
     */
    fun publishMarkers(detections: List<MarkerDetection>) {
        val ws = webSocket
        if (ws == null) {
            Log.d(TAG, "publishMarkers: skipped – no active WebSocket")
            return
        }
        if (state != State.CONNECTED) {
            Log.d(TAG, "publishMarkers: skipped – state is $state")
            return
        }
        if (detections.isEmpty()) {
            return
        }

        val now = System.currentTimeMillis()
        if (now - lastPublishMs < MIN_PUBLISH_INTERVAL_MS) {
            return
        }
        lastPublishMs = now

        val payload = buildDetectionsPayload(detections)
        val msg = JSONObject().apply {
            put("op", "publish")
            put("topic", TOPIC_DETECTIONS)
            put("msg", JSONObject().apply { put("data", payload) })
        }
        ws.send(msg.toString())
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private fun advertise() {
        webSocket?.send(
            JSONObject().apply {
                put("op", "advertise")
                put("topic", TOPIC_DETECTIONS)
                put("type", MSG_TYPE_STRING)
            }.toString()
        )
    }

    private fun buildDetectionsPayload(detections: List<MarkerDetection>): String {
        val array = JSONArray()
        for (detection in detections) {
            val obj = JSONObject()
            when (detection) {
                is MarkerDetection.AprilTag -> {
                    obj.put("type", "apriltag")
                    obj.put("id", detection.id)
                    obj.put("corners", cornersToJsonArray(detection.corners))
                    obj.put("timestamp_ms", detection.timestampMs)
                }
                is MarkerDetection.Aruco -> {
                    obj.put("type", "aruco")
                    obj.put("id", detection.id)
                    obj.put("corners", cornersToJsonArray(detection.corners))
                    obj.put("timestamp_ms", detection.timestampMs)
                }
                is MarkerDetection.QrCode -> {
                    obj.put("type", "qr")
                    obj.put("text", detection.text)
                    obj.put("corners", cornersToJsonArray(detection.corners))
                    obj.put("timestamp_ms", detection.timestampMs)
                }
                is MarkerDetection.CCTag -> {
                    obj.put("type", "cctag")
                    obj.put("id", detection.id)
                    obj.put("center_x", detection.center.first.toDouble())
                    obj.put("center_y", detection.center.second.toDouble())
                    obj.put("radius", detection.radius.toDouble())
                    obj.put("corners", cornersToJsonArray(detection.corners))
                    obj.put("timestamp_ms", detection.timestampMs)
                }
            }
            array.put(obj)
        }
        return array.toString()
    }

    private fun cornersToJsonArray(corners: List<Pair<Float, Float>>): JSONArray {
        val array = JSONArray()
        for ((x, y) in corners) {
            array.put(JSONObject().apply {
                put("x", x.toDouble())
                put("y", y.toDouble())
            })
        }
        return array
    }

    private fun updateState(newState: State) {
        state = newState
        onStateChanged?.invoke(newState)
    }

    // -------------------------------------------------------------------------
    // Inner WebSocket listener
    // -------------------------------------------------------------------------

    private inner class RosBridgeWebSocketListener : WebSocketListener() {
        override fun onOpen(ws: WebSocket, response: Response) {
            Log.i(TAG, "WebSocket opened")
            advertise()
            updateState(State.CONNECTED)
        }

        override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
            Log.e(TAG, "WebSocket failure", t)
            webSocket = null
            updateState(State.ERROR)
        }

        override fun onClosing(ws: WebSocket, code: Int, reason: String) {
            Log.i(TAG, "WebSocket closing: $code $reason")
            ws.close(1000, null)
            webSocket = null
            updateState(State.DISCONNECTED)
        }

        override fun onClosed(ws: WebSocket, code: Int, reason: String) {
            Log.i(TAG, "WebSocket closed: $code $reason")
            webSocket = null
            updateState(State.DISCONNECTED)
        }
    }
}
