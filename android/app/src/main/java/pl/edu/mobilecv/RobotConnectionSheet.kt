package pl.edu.mobilecv

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText

/**
 * Bottom sheet that allows the user to configure and manage the ROSBridge
 * WebSocket connection to the robot.
 *
 * The fragment communicates back to [MainActivity] through the [onConnect]
 * and [onDisconnect] lambdas rather than directly holding a reference to
 * [RosBridgeClient], which keeps the fragment decoupled from the activity.
 *
 * Usage:
 * ```kotlin
 * RobotConnectionSheet().apply {
 *     currentState = rosBridgeClient.state
 *     onConnect = { host, port -> rosBridgeClient.connect(host, port) }
 *     onDisconnect = { rosBridgeClient.disconnect() }
 * }.show(supportFragmentManager, RobotConnectionSheet.TAG)
 * ```
 */
class RobotConnectionSheet : BottomSheetDialogFragment() {

    companion object {
        const val TAG = "RobotConnectionSheet"
        private const val DEFAULT_HOST = "192.168.1.100"
    }

    /** Current client state to pre-populate the UI when the sheet opens. */
    var currentState: RosBridgeClient.State = RosBridgeClient.State.DISCONNECTED

    /** Host and port last used for connection, pre-filled in the fields. */
    var lastHost: String = DEFAULT_HOST
    var lastPort: Int = RosBridgeClient.DEFAULT_PORT

    /** Called when the user presses Connect.  Receives (host, port). */
    var onConnect: ((host: String, port: Int) -> Unit)? = null

    /** Called when the user presses Disconnect. */
    var onDisconnect: (() -> Unit)? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View = inflater.inflate(R.layout.bottom_sheet_robot_connection, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val statusText = view.findViewById<TextView>(R.id.textViewConnectionStatus)
        val hostEdit = view.findViewById<TextInputEditText>(R.id.editTextRobotHost)
        val portEdit = view.findViewById<TextInputEditText>(R.id.editTextRobotPort)
        val connectBtn = view.findViewById<MaterialButton>(R.id.btnConnectRobot)

        // Pre-fill fields with last used values.
        hostEdit.setText(lastHost)
        portEdit.setText(lastPort.toString())

        updateUi(statusText, connectBtn, currentState)

        connectBtn.setOnClickListener {
            when (currentState) {
                RosBridgeClient.State.CONNECTED,
                RosBridgeClient.State.CONNECTING -> {
                    onDisconnect?.invoke()
                    dismiss()
                }
                RosBridgeClient.State.DISCONNECTED,
                RosBridgeClient.State.ERROR -> {
                    val host = hostEdit.text?.toString()?.trim().orEmpty()
                    val portStr = portEdit.text?.toString()?.trim().orEmpty()
                    val port = portStr.toIntOrNull()

                    when {
                        host.isEmpty() -> {
                            view.findViewById<com.google.android.material.textfield.TextInputLayout>(
                                R.id.tilRobotHost
                            ).error = requireContext().getString(R.string.robot_host_empty_error)
                        }
                        port == null || port !in 1..65535 -> {
                            view.findViewById<com.google.android.material.textfield.TextInputLayout>(
                                R.id.tilRobotPort
                            ).error = requireContext().getString(R.string.robot_port_invalid_error)
                        }
                        else -> {
                            onConnect?.invoke(host, port)
                            dismiss()
                        }
                    }
                }
            }
        }
    }

    private fun updateUi(
        statusText: TextView,
        connectBtn: MaterialButton,
        state: RosBridgeClient.State,
    ) {
        val ctx = requireContext()
        when (state) {
            RosBridgeClient.State.DISCONNECTED -> {
                statusText.text = ctx.getString(R.string.robot_status_disconnected)
                connectBtn.text = ctx.getString(R.string.robot_btn_connect)
            }
            RosBridgeClient.State.CONNECTING -> {
                statusText.text = ctx.getString(R.string.robot_status_connecting)
                connectBtn.text = ctx.getString(R.string.robot_btn_disconnect)
            }
            RosBridgeClient.State.CONNECTED -> {
                statusText.text = ctx.getString(R.string.robot_status_connected)
                connectBtn.text = ctx.getString(R.string.robot_btn_disconnect)
            }
            RosBridgeClient.State.ERROR -> {
                statusText.text = ctx.getString(R.string.robot_status_error)
                connectBtn.text = ctx.getString(R.string.robot_btn_connect)
            }
        }
    }
}
