package pl.edu.mobilecv

import android.view.View
import android.widget.SeekBar
import androidx.core.view.children
import com.google.android.material.chip.Chip
import com.google.android.material.tabs.TabLayout
import pl.edu.mobilecv.databinding.ActivityMainBinding

/**
 * Kontroler logiki UI dla analizy (zakładki, chipy, slidery i widoczność overlayów).
 */
class AnalysisUiController(
    private val binding: ActivityMainBinding,
    private val imageProcessor: ImageProcessor,
    private val callbacks: Callbacks,
) {
    interface Callbacks {
        /** Wymusza pobranie modeli wymaganych dla aktywnego trybu. */
        fun ensureModelsForMode(mode: AnalysisMode)

        /** Powiadamia o zmianie aktywnego filtra. */
        fun onFilterChanged(filter: OpenCvFilter)

        /** Powiadamia o zmianie aktywnego trybu analizy. */
        fun onModeChanged(mode: AnalysisMode)

        /** Wykonuje akcję kalibracji eye-trackingu. */
        fun onEyeTrackingCalibrationRequested()

        /** Loguje lub obsługuje niepoprawny tryb przekazany z Intent. */
        fun onUnknownInitialMode(modeName: String)
    }

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

    fun setupAll() {
        setupAnalysisTabs()
        setupSliders()
        setupToggles()
        setupEyeTrackingCalibrationFab()
    }

    fun applyInitialMode(modeName: String?) {
        if (modeName.isNullOrBlank()) {
            return
        }
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

    private fun setupSliders() {
        binding.seekBarKernelSize.progress = imageProcessor.morphKernelSize - 1
        updateKernelSizeLabel(imageProcessor.morphKernelSize)
        binding.seekBarKernelSize.setOnSeekBarChangeListener(simpleSeekbarListener { progress ->
            val half = progress + 1
            imageProcessor.morphKernelSize = half
            updateKernelSizeLabel(half)
        })

        binding.seekBarVoMaxFeatures.progress = imageProcessor.voMaxFeatures
        binding.textViewVoMaxFeatures.text = imageProcessor.voMaxFeatures.toString()
        binding.seekBarVoMaxFeatures.setOnSeekBarChangeListener(simpleSeekbarListener { progress ->
            val value = maxOf(10, progress)
            imageProcessor.voMaxFeatures = value
            binding.textViewVoMaxFeatures.text = value.toString()
        })

        binding.seekBarVoMinParallax.progress = (imageProcessor.voMinParallax * 10).toInt()
        binding.textViewVoMinParallax.text = "%.1f".format(imageProcessor.voMinParallax)
        binding.seekBarVoMinParallax.setOnSeekBarChangeListener(simpleSeekbarListener { progress ->
            val value = progress / 10.0
            imageProcessor.voMinParallax = value
            binding.textViewVoMinParallax.text = "%.1f".format(value)
        })

        binding.seekBarGeometryMaxPlanes.progress = imageProcessor.geometryMaxPlanes - 1
        binding.textViewGeometryMaxPlanes.text = imageProcessor.geometryMaxPlanes.toString()
        binding.seekBarGeometryMaxPlanes.setOnSeekBarChangeListener(simpleSeekbarListener { progress ->
            val value = progress + 1
            imageProcessor.geometryMaxPlanes = value
            binding.textViewGeometryMaxPlanes.text = value.toString()
        })

        binding.seekBarPoseEmaAlpha.progress =
            (imageProcessor.poseEmaAlpha * 100.0).toInt().coerceIn(5, 95)
        binding.textViewPoseEmaAlpha.text = "%.2f".format(imageProcessor.poseEmaAlpha)
        binding.seekBarPoseEmaAlpha.setOnSeekBarChangeListener(simpleSeekbarListener { progress ->
            val value = (progress.coerceIn(5, 95)) / 100.0
            imageProcessor.poseEmaAlpha = value
            binding.textViewPoseEmaAlpha.text = "%.2f".format(value)
        })
    }

    private fun setupToggles() {
        binding.switchVoMesh.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.isVoMeshEnabled = isChecked
        }
        binding.switchActiveVision.setOnCheckedChangeListener { _, isChecked ->
            isActiveVisionEnabled = isChecked
            imageProcessor.isActiveVisionEnabled = isChecked
            updateContextualControls()
        }
        binding.switchActiveVisionVisualization.setOnCheckedChangeListener { _, isChecked ->
            isActiveVisionVisualizationEnabled = isChecked
            imageProcessor.isActiveVisionVisualizationEnabled = isChecked
        }

        binding.switchPoseSmoothing.isChecked = imageProcessor.poseSmoothingEnabled
        binding.switchPoseSmoothing.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.poseSmoothingEnabled = isChecked
            if (!binding.switchPoseRawVsSmoothed.isChecked) {
                imageProcessor.poseOutputMode = if (isChecked) {
                    PoseOutputMode.SMOOTHED
                } else {
                    PoseOutputMode.RAW
                }
            }
        }
        binding.switchPoseOneEuro.isChecked =
            imageProcessor.poseTemporalFilterType == PoseTemporalFilterType.ONE_EURO
        binding.switchPoseOneEuro.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.poseTemporalFilterType = if (isChecked) {
                PoseTemporalFilterType.ONE_EURO
            } else {
                PoseTemporalFilterType.EMA
            }
        }
        binding.switchPoseRawVsSmoothed.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.poseOutputMode = if (isChecked) {
                PoseOutputMode.RAW_VS_SMOOTHED
            } else if (imageProcessor.poseSmoothingEnabled) {
                PoseOutputMode.SMOOTHED
            } else {
                PoseOutputMode.RAW
            }
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
        updateContextualControls()
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

    private fun updateContextualControls() {
        binding.layoutKernelSize.visibility =
            if (currentMode == AnalysisMode.MORPHOLOGY) View.VISIBLE else View.GONE
        binding.layoutGeometryMaxPlanes.visibility =
            if (currentMode == AnalysisMode.GEOMETRY && currentFilter == OpenCvFilter.PLANE_DETECTION) {
                View.VISIBLE
            } else {
                View.GONE
            }

        val isOdometry = currentMode == AnalysisMode.ODOMETRY
        binding.layoutVoMaxFeatures.visibility = if (isOdometry) View.VISIBLE else View.GONE
        binding.layoutVoMinParallax.visibility = if (isOdometry) View.VISIBLE else View.GONE
        binding.layoutVoMesh.visibility = if (isOdometry) View.VISIBLE else View.GONE
        binding.layoutPoseTemporalControls.visibility =
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
        binding.textViewKernelSize.text = binding.root.context.getString(
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
