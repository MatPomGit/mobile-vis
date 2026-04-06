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
        private const val ARG_CURRENT_RESOLUTION = "current_resolution"

        /**
         * Creates a new instance of [ResolutionBottomSheet] with the specified [current] resolution.
         */
        fun newInstance(current: CameraResolution): ResolutionBottomSheet {
            return ResolutionBottomSheet().apply {
                arguments = Bundle().apply {
                    putString(ARG_CURRENT_RESOLUTION, current.name)
                }
            }
        }
    }

    /** The resolution that should appear pre-selected when the sheet opens. */
    private val initialResolution: CameraResolution by lazy {
        val name = arguments?.getString(ARG_CURRENT_RESOLUTION)
        CameraResolution.entries.find { it.name == name } ?: CameraResolution.DEFAULT
    }

    /**
     * Deprecated: Use Fragment Result API with request key "resolution_request"
     * and result key "selected_resolution" (String - name of [CameraResolution]).
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
                isChecked = (resolution == initialResolution)
            }
            radioGroup.addView(radio)
        }

        applyBtn.setOnClickListener {
            val checkedId = radioGroup.checkedRadioButtonId
            if (checkedId != View.NO_ID) {
                val selected = view.findViewById<RadioButton>(checkedId).tag as CameraResolution
                
                // Invoke legacy callback if set
                onResolutionSelected?.invoke(selected)
                
                // Modern Fragment Result API
                val result = Bundle().apply {
                    putString("selected_resolution", selected.name)
                }
                parentFragmentManager.setFragmentResult("resolution_request", result)
            }
            dismiss()
        }
    }
}
