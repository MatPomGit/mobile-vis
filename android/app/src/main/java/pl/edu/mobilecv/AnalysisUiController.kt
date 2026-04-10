package pl.edu.mobilecv

import android.view.LayoutInflater
import android.view.View
import android.widget.ImageButton
import android.widget.SeekBar
import android.widget.TextView
import androidx.core.view.children
import com.google.android.material.chip.Chip
import com.google.android.material.switchmaterial.SwitchMaterial
import com.google.android.material.tabs.TabLayout
import pl.edu.mobilecv.databinding.ActivityMainBinding

/**
 * Kontroler logiki UI dla analizy.
 *
 * Odpowiada za:
 * - przełączanie trybów i filtrów,
 * - dynamiczne ładowanie sekcji kontekstowej dla aktywnego trybu,
 * - utrzymywanie prostego podziału na akcje częste i ustawienia zaawansowane.
 */
class AnalysisUiController(
    private val binding: ActivityMainBinding,
    private val imageProcessor: ImageProcessor,
    private val callbacks: Callbacks,
) {
    interface Callbacks {
        fun ensureModelsForMode(mode: AnalysisMode)
        fun onFilterChanged(filter: OpenCvFilter)
        fun onModeChanged(mode: AnalysisMode)
        fun onEyeTrackingCalibrationRequested()
        fun onUnknownInitialMode(modeName: String)
    }

    private data class DynamicControls(
        val root: View,
        val layoutKernelSize: View? = null,
        val seekBarKernelSize: SeekBar? = null,
        val textViewKernelSize: TextView? = null,
        val layoutVoMaxFeatures: View? = null,
        val seekBarVoMaxFeatures: SeekBar? = null,
        val textViewVoMaxFeatures: TextView? = null,
        val layoutVoMinParallax: View? = null,
        val seekBarVoMinParallax: SeekBar? = null,
        val textViewVoMinParallax: TextView? = null,
        val layoutGeometryMaxPlanes: View? = null,
        val seekBarGeometryMaxPlanes: SeekBar? = null,
        val textViewGeometryMaxPlanes: TextView? = null,
        val layoutVoMesh: View? = null,
        val switchVoMesh: SwitchMaterial? = null,
        val layoutPoseTemporalControls: View? = null,
        val switchPoseSmoothing: SwitchMaterial? = null,
        val switchPoseOneEuro: SwitchMaterial? = null,
        val switchPoseRawVsSmoothed: SwitchMaterial? = null,
        val seekBarPoseEmaAlpha: SeekBar? = null,
        val textViewPoseEmaAlpha: TextView? = null,
    )

    @Volatile
    var currentMode: AnalysisMode = AnalysisMode.entries.first()
        private set

    @Volatile
    var currentFilter: OpenCvFilter = OpenCvFilter.ORIGINAL
        private set

    @Volatile
    var isActiveVisionEnabled: Boolean = false
        private set

    @Volatile
    var isActiveVisionVisualizationEnabled: Boolean = false
        private set

    private var dynamicControls: DynamicControls? = null

    fun setupAll() {
        setupAnalysisTabs()
        setupAdvancedSectionToggle()
        setupGlobalToggles()
        setupEyeTrackingCalibrationFab()
    }

    fun applyInitialMode(modeName: String?) {
        if (modeName.isNullOrBlank()) return
        val index = AnalysisMode.entries.indexOfFirst { it.name == modeName }
        if (index >= 0) {
            binding.tabLayoutModes.getTabAt(index)?.select()
        } else {
            callbacks.onUnknownInitialMode(modeName)
        }
    }

    fun updateDiagnosticsOverlay(
        fps: Double,
        width: Int,
        height: Int,
        processingTimeMs: Long,
        lensFacingFront: Boolean,
        stringProvider: (Int, Array<out Any>) -> String,
    ) {
        val cameraLabel = stringProvider(
            if (lensFacingFront) R.string.diagnostics_camera_front else R.string.diagnostics_camera_back,
            emptyArray(),
        )
        binding.textViewDiagnostics.text = buildString {
            appendLine(stringProvider(R.string.diagnostics_fps, arrayOf(fps)))
            appendLine(stringProvider(R.string.diagnostics_resolution, arrayOf(width, height)))
            appendLine(stringProvider(R.string.diagnostics_processing_time, arrayOf(processingTimeMs)))
            appendLine(stringProvider(R.string.diagnostics_filter, arrayOf(currentFilter.displayName)))
            appendLine(stringProvider(R.string.diagnostics_active_vision, arrayOf(isActiveVisionEnabled)))
            appendLine(
                stringProvider(
                    R.string.diagnostics_active_vision_visualization,
                    arrayOf(isActiveVisionVisualizationEnabled),
                ),
            )
            append(cameraLabel)
        }
    }

    private fun setupAnalysisTabs() {
        AnalysisMode.entries.forEach { mode ->
            binding.tabLayoutModes.addTab(binding.tabLayoutModes.newTab().setText(mode.displayName))
        }
        binding.tabLayoutModes.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                updateFilterChips(AnalysisMode.entries[tab.position])
            }

            override fun onTabUnselected(tab: TabLayout.Tab) = Unit
            override fun onTabReselected(tab: TabLayout.Tab) = Unit
        })
        updateFilterChips(AnalysisMode.entries.first())
    }

    private fun setupAdvancedSectionToggle() {
        val toggleButton: ImageButton = binding.btnToggleAdvanced
        toggleButton.setOnClickListener {
            val isCurrentlyVisible = binding.layoutAdvancedContent.visibility == View.VISIBLE
            binding.layoutAdvancedContent.visibility = if (isCurrentlyVisible) View.GONE else View.VISIBLE
            toggleButton.setImageResource(
                if (isCurrentlyVisible) android.R.drawable.arrow_down_float
                else android.R.drawable.arrow_up_float,
            )
        }
    }

    private fun setupGlobalToggles() {
        binding.switchActiveVision.setOnCheckedChangeListener { _, isChecked ->
            isActiveVisionEnabled = isChecked
            imageProcessor.isActiveVisionEnabled = isChecked
            updateContextualControls()
        }
        binding.switchActiveVisionVisualization.setOnCheckedChangeListener { _, isChecked ->
            isActiveVisionVisualizationEnabled = isChecked
            imageProcessor.isActiveVisionVisualizationEnabled = isChecked
        }
    }

    private fun setupEyeTrackingCalibrationFab() {
        binding.fabEyeTrackingCalibration.setOnClickListener {
            callbacks.onEyeTrackingCalibrationRequested()
        }
    }

    private fun updateFilterChips(mode: AnalysisMode) {
        currentMode = mode
        callbacks.onModeChanged(mode)
        binding.chipGroupFilters.removeAllViews()

        inflateControlsForMode(mode)

        val firstFilter = mode.filters.firstOrNull() ?: run {
            updateContextualControls()
            return
        }
        currentFilter = firstFilter
        callbacks.onFilterChanged(firstFilter)
        binding.textViewCurrentFilter.text = currentFilter.displayName

        mode.filters.forEachIndexed { index, filter ->
            val chip = Chip(binding.root.context).apply {
                text = filter.displayName
                isCheckable = true
                isChecked = index == 0
                setOnClickListener {
                    onFilterChipClicked(filter, mode, this)
                }
            }
            binding.chipGroupFilters.addView(chip)
        }

        callbacks.ensureModelsForMode(mode)
        updateContextHeader(mode)
        updateContextualControls()
    }

    private fun inflateControlsForMode(mode: AnalysisMode) {
        val layoutRes = when (mode) {
            AnalysisMode.MORPHOLOGY -> R.layout.layout_controls_morphology
            AnalysisMode.ODOMETRY,
            AnalysisMode.FULL_ODOMETRY_3D,
            AnalysisMode.GEOMETRY,
            -> R.layout.layout_controls_odometry
            else -> R.layout.layout_controls_detection
        }

        binding.layoutDynamicControlsHost.removeAllViews()
        val view = LayoutInflater.from(binding.root.context)
            .inflate(layoutRes, binding.layoutDynamicControlsHost, false)
        binding.layoutDynamicControlsHost.addView(view)
        dynamicControls = bindDynamicControls(view)
        setupDynamicControlsListeners()
    }

    private fun bindDynamicControls(root: View): DynamicControls {
        fun <T : View> v(id: Int): T? = root.findViewById(id)
        return DynamicControls(
            root = root,
            layoutKernelSize = v(R.id.layoutKernelSize),
            seekBarKernelSize = v(R.id.seekBarKernelSize),
            textViewKernelSize = v(R.id.textViewKernelSize),
            layoutVoMaxFeatures = v(R.id.layoutVoMaxFeatures),
            seekBarVoMaxFeatures = v(R.id.seekBarVoMaxFeatures),
            textViewVoMaxFeatures = v(R.id.textViewVoMaxFeatures),
            layoutVoMinParallax = v(R.id.layoutVoMinParallax),
            seekBarVoMinParallax = v(R.id.seekBarVoMinParallax),
            textViewVoMinParallax = v(R.id.textViewVoMinParallax),
            layoutGeometryMaxPlanes = v(R.id.layoutGeometryMaxPlanes),
            seekBarGeometryMaxPlanes = v(R.id.seekBarGeometryMaxPlanes),
            textViewGeometryMaxPlanes = v(R.id.textViewGeometryMaxPlanes),
            layoutVoMesh = v(R.id.layoutVoMesh),
            switchVoMesh = v(R.id.switchVoMesh),
            layoutPoseTemporalControls = v(R.id.layoutPoseTemporalControls),
            switchPoseSmoothing = v(R.id.switchPoseSmoothing),
            switchPoseOneEuro = v(R.id.switchPoseOneEuro),
            switchPoseRawVsSmoothed = v(R.id.switchPoseRawVsSmoothed),
            seekBarPoseEmaAlpha = v(R.id.seekBarPoseEmaAlpha),
            textViewPoseEmaAlpha = v(R.id.textViewPoseEmaAlpha),
        )
    }

    private fun setupDynamicControlsListeners() {
        val controls = dynamicControls ?: return

        controls.seekBarKernelSize?.apply {
            progress = imageProcessor.morphKernelSize - 1
            updateKernelSizeLabel(imageProcessor.morphKernelSize)
            setOnSeekBarChangeListener(simpleSeekbarListener { progressValue ->
                val half = progressValue + 1
                imageProcessor.morphKernelSize = half
                updateKernelSizeLabel(half)
            })
        }

        controls.seekBarVoMaxFeatures?.apply {
            progress = imageProcessor.voMaxFeatures
            controls.textViewVoMaxFeatures?.text = imageProcessor.voMaxFeatures.toString()
            setOnSeekBarChangeListener(simpleSeekbarListener { progressValue ->
                val value = maxOf(10, progressValue)
                imageProcessor.voMaxFeatures = value
                controls.textViewVoMaxFeatures?.text = value.toString()
            })
        }

        controls.seekBarVoMinParallax?.apply {
            progress = (imageProcessor.voMinParallax * 10).toInt()
            controls.textViewVoMinParallax?.text = "%.1f".format(imageProcessor.voMinParallax)
            setOnSeekBarChangeListener(simpleSeekbarListener { progressValue ->
                val value = progressValue / 10.0
                imageProcessor.voMinParallax = value
                controls.textViewVoMinParallax?.text = "%.1f".format(value)
            })
        }

        controls.seekBarGeometryMaxPlanes?.apply {
            progress = imageProcessor.geometryMaxPlanes - 1
            controls.textViewGeometryMaxPlanes?.text = imageProcessor.geometryMaxPlanes.toString()
            setOnSeekBarChangeListener(simpleSeekbarListener { progressValue ->
                val value = progressValue + 1
                imageProcessor.geometryMaxPlanes = value
                controls.textViewGeometryMaxPlanes?.text = value.toString()
            })
        }

        controls.switchVoMesh?.apply {
            isChecked = imageProcessor.isVoMeshEnabled
            setOnCheckedChangeListener { _, isChecked ->
                imageProcessor.isVoMeshEnabled = isChecked
            }
        }

        controls.switchPoseSmoothing?.apply {
            isChecked = imageProcessor.poseSmoothingEnabled
            setOnCheckedChangeListener { _, isChecked ->
                imageProcessor.poseSmoothingEnabled = isChecked
                val rawVsSmoothed = controls.switchPoseRawVsSmoothed?.isChecked == true
                if (!rawVsSmoothed) {
                    imageProcessor.poseOutputMode = if (isChecked) {
                        PoseOutputMode.SMOOTHED
                    } else {
                        PoseOutputMode.RAW
                    }
                }
            }
        }

        controls.switchPoseOneEuro?.apply {
            isChecked = imageProcessor.poseTemporalFilterType == PoseTemporalFilterType.ONE_EURO
            setOnCheckedChangeListener { _, isChecked ->
                imageProcessor.poseTemporalFilterType = if (isChecked) {
                    PoseTemporalFilterType.ONE_EURO
                } else {
                    PoseTemporalFilterType.EMA
                }
            }
        }

        controls.switchPoseRawVsSmoothed?.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.poseOutputMode = if (isChecked) {
                PoseOutputMode.RAW_VS_SMOOTHED
            } else if (imageProcessor.poseSmoothingEnabled) {
                PoseOutputMode.SMOOTHED
            } else {
                PoseOutputMode.RAW
            }
        }

        controls.seekBarPoseEmaAlpha?.apply {
            progress = (imageProcessor.poseEmaAlpha * 100.0).toInt().coerceIn(5, 95)
            controls.textViewPoseEmaAlpha?.text = "%.2f".format(imageProcessor.poseEmaAlpha)
            setOnSeekBarChangeListener(simpleSeekbarListener { progressValue ->
                val value = (progressValue.coerceIn(5, 95)) / 100.0
                imageProcessor.poseEmaAlpha = value
                controls.textViewPoseEmaAlpha?.text = "%.2f".format(value)
            })
        }
    }

    private fun onFilterChipClicked(filter: OpenCvFilter, mode: AnalysisMode, selectedChip: Chip) {
        if (selectedChip.isChecked) {
            currentFilter = filter
            callbacks.onFilterChanged(filter)
            binding.textViewCurrentFilter.text = filter.displayName
            binding.chipGroupFilters.children
                .filterIsInstance<Chip>()
                .forEach { chip -> if (chip !== selectedChip) chip.isChecked = false }
        } else {
            val defaultFilter = mode.filters.first()
            currentFilter = defaultFilter
            callbacks.onFilterChanged(defaultFilter)
            binding.textViewCurrentFilter.text = defaultFilter.displayName
            (binding.chipGroupFilters.getChildAt(0) as? Chip)?.isChecked = true
        }
        updateContextualControls()
    }

    private fun updateContextHeader(mode: AnalysisMode) {
        binding.textViewContextModuleTitle.text = binding.root.context.getString(
            R.string.context_module_title,
            mode.displayName,
        )
        val hintRes = when (mode) {
            AnalysisMode.MORPHOLOGY -> R.string.context_hint_morphology
            AnalysisMode.ODOMETRY,
            AnalysisMode.FULL_ODOMETRY_3D,
            AnalysisMode.GEOMETRY,
            -> R.string.context_hint_odometry
            else -> R.string.context_hint_detection
        }
        binding.textViewContextHint.text = binding.root.context.getString(hintRes)
    }

    private fun updateContextualControls() {
        val controls = dynamicControls

        controls?.layoutKernelSize?.visibility =
            if (currentMode == AnalysisMode.MORPHOLOGY) View.VISIBLE else View.GONE
        controls?.layoutGeometryMaxPlanes?.visibility =
            if (currentMode == AnalysisMode.GEOMETRY && currentFilter == OpenCvFilter.PLANE_DETECTION) {
                View.VISIBLE
            } else {
                View.GONE
            }

        val isOdometry = currentMode == AnalysisMode.ODOMETRY ||
            currentMode == AnalysisMode.FULL_ODOMETRY_3D
        controls?.layoutVoMaxFeatures?.visibility = if (isOdometry) View.VISIBLE else View.GONE
        controls?.layoutVoMinParallax?.visibility = if (isOdometry) View.VISIBLE else View.GONE
        controls?.layoutVoMesh?.visibility = if (isOdometry) View.VISIBLE else View.GONE

        controls?.layoutPoseTemporalControls?.visibility =
            if (currentMode == AnalysisMode.MARKERS) View.VISIBLE else View.GONE

        binding.fabCalibrationMenu.visibility =
            if (currentMode == AnalysisMode.CALIBRATION) View.VISIBLE else View.GONE
        binding.fabEyeTrackingCalibration.visibility =
            if (currentMode == AnalysisMode.POSE && currentFilter == OpenCvFilter.EYE_TRACKING) {
                View.VISIBLE
            } else {
                View.GONE
            }

        binding.switchActiveVision.visibility =
            if (currentMode == AnalysisMode.FILTERS) View.VISIBLE else View.GONE
        binding.switchActiveVisionVisualization.visibility =
            if (currentMode == AnalysisMode.FILTERS && isActiveVisionEnabled) View.VISIBLE else View.GONE

        binding.fabSavePointCloud.visibility =
            if (currentFilter == OpenCvFilter.POINT_CLOUD) View.VISIBLE else View.GONE
        binding.fabSaveSlamMap.visibility =
            if (currentFilter.isFullOdometry) View.VISIBLE else View.GONE
        binding.fabLoadSlamMap.visibility =
            if (currentFilter.isFullOdometry) View.VISIBLE else View.GONE
    }

    private fun updateKernelSizeLabel(halfKernelSize: Int) {
        val side = 2 * halfKernelSize + 1
        dynamicControls?.textViewKernelSize?.text = binding.root.context.getString(
            R.string.morphology_kernel_size_value,
            side,
            side,
        )
    }

    private fun simpleSeekbarListener(onChanged: (Int) -> Unit): SeekBar.OnSeekBarChangeListener {
        return object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                onChanged(progress)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar) = Unit
        }
    }
}
