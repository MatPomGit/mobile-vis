package pl.edu.mobilecv

import java.util.concurrent.ConcurrentHashMap

/**
 * Centralny store statusu modułów ML/CV.
 *
 * Przechowuje bieżący status modułu, komunikat błędu oraz znacznik czasu ostatniej próby
 * inicjalizacji. Dzięki temu UI korzysta z jednego źródła prawdy zamiast lokalnych flag.
 */
object ModuleStatusStore {

    /** Typy modułów zarządzane przez aplikację. */
    enum class ModuleType {
        MEDIAPIPE,
        YOLO,
        RTMDET,
        TFLITE,
    }

    /** Ustandaryzowane statusy modułów. */
    enum class ModuleStatus {
        READY,
        DOWNLOADING,
        ERROR,
        DISABLED,
    }

    /** Migawka stanu pojedynczego modułu. */
    data class ModuleSnapshot(
        val status: ModuleStatus,
        val errorMessageResId: Int? = null,
        val lastInitAttemptEpochMs: Long? = null,
    )

    private val state = ConcurrentHashMap<ModuleType, ModuleSnapshot>()

    init {
        ModuleType.entries.forEach { module ->
            state[module] = ModuleSnapshot(status = ModuleStatus.DISABLED)
        }
    }

    /** Zwraca aktualny stan wskazanego modułu. */
    fun get(moduleType: ModuleType): ModuleSnapshot =
        state[moduleType] ?: ModuleSnapshot(status = ModuleStatus.DISABLED)

    /** Zwraca pełną migawkę wszystkich modułów. */
    fun snapshot(): Map<ModuleType, ModuleSnapshot> = ModuleType.entries.associateWith { get(it) }

    /** Aktualizuje status modułu i zapisuje timestamp ostatniej próby inicjalizacji. */
    @Synchronized
    fun setStatus(moduleType: ModuleType, status: ModuleStatus, errorMessageResId: Int? = null) {
        val existing = get(moduleType)
        state[moduleType] = existing.copy(
            status = status,
            errorMessageResId = errorMessageResId,
            lastInitAttemptEpochMs = System.currentTimeMillis(),
        )
    }
}
