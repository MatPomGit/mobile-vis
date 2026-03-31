package pl.edu.mobilecv

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.RadioButton
import android.widget.RadioGroup
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.button.MaterialButton

/**
 * Bottom sheet that lets the user choose the camera resolution for live analysis.
 *
 * The sheet presents one [RadioButton] per [CameraResolution] entry and notifies
 * [MainActivity] via [onResolutionSelected] when the user confirms the choice.
 *
 * Usage:
 * ```kotlin
 * ResolutionBottomSheet().apply {
 *     currentResolution = selectedResolution
 *     onResolutionSelected = { resolution -> applyResolution(resolution) }
 * }.show(supportFragmentManager, ResolutionBottomSheet.TAG)
 * ```
 */
class ResolutionBottomSheet : BottomSheetDialogFragment() {

    companion object {
        const val TAG = "ResolutionBottomSheet"
    }

    /** The resolution that should appear pre-selected when the sheet opens. */
    var currentResolution: CameraResolution = CameraResolution.DEFAULT

    /**
     * Invoked on the main thread when the user taps **Apply** with a valid selection.
     * Receives the chosen [CameraResolution].
     */
    var onResolutionSelected: ((CameraResolution) -> Unit)? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View = inflater.inflate(R.layout.bottom_sheet_resolution, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val radioGroup = view.findViewById<RadioGroup>(R.id.radioGroupResolution)
        val applyBtn = view.findViewById<MaterialButton>(R.id.btnApplyResolution)

        // Populate one RadioButton per resolution preset.
        CameraResolution.entries.forEach { resolution ->
            val radio = RadioButton(requireContext()).apply {
                id = View.generateViewId()
                text = resolution.displayName
                tag = resolution
                isChecked = (resolution == currentResolution)
            }
            radioGroup.addView(radio)
        }

        applyBtn.setOnClickListener {
            val checkedId = radioGroup.checkedRadioButtonId
            if (checkedId != View.NO_ID) {
                val selected = view.findViewById<RadioButton>(checkedId).tag as CameraResolution
                onResolutionSelected?.invoke(selected)
            }
            dismiss()
        }
    }
}
