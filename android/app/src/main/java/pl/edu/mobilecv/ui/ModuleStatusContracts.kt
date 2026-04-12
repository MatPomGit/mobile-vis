package pl.edu.mobilecv.ui

import pl.edu.mobilecv.AnalysisMode
import pl.edu.mobilecv.ModuleStatusState
import pl.edu.mobilecv.ModuleStatusStore

/**
 * Jednolity kontrakt prezentacyjny statusu modułu wykorzystywany przez ekrany UI.
 */
data class ModuleStatusPresentation(
    val moduleType: ModuleStatusStore.ModuleType,
    val status: ModuleStatusState,
    val statusLabel: String,
    val lastInitAttemptEpochMs: Long?,
)

/**
 * Mapper kontraktów statusów pomiędzy store, routingiem trybów i warstwą UI.
 */
object ModuleStatusContracts {

    /**
     * Mapuje aktywny tryb analizy na zależny moduł ML.
     */
    fun moduleTypeForMode(mode: AnalysisMode): ModuleStatusStore.ModuleType? = when (mode) {
        AnalysisMode.POSE -> ModuleStatusStore.ModuleType.MEDIAPIPE
        AnalysisMode.YOLO -> ModuleStatusStore.ModuleType.YOLO
        else -> null
    }

    /**
     * Buduje gotowy do renderowania kontrakt statusu modułu.
     */
    fun toPresentation(
        moduleType: ModuleStatusStore.ModuleType,
        snapshot: ModuleStatusStore.ModuleSnapshot,
    ): ModuleStatusPresentation = ModuleStatusPresentation(
        moduleType = moduleType,
        status = snapshot.status,
        statusLabel = statusLabel(snapshot.status),
        lastInitAttemptEpochMs = snapshot.lastInitAttemptEpochMs,
    )

    /**
     * Zwraca stabilną etykietę tekstową statusu do komponentów diagnostycznych.
     */
    fun statusLabel(status: ModuleStatusState): String = when (status) {
        ModuleStatusState.Ready -> "READY"
        ModuleStatusState.Downloading -> "DOWNLOADING"
        ModuleStatusState.Disabled -> "DISABLED"
        is ModuleStatusState.Error -> "ERROR"
    }
}
