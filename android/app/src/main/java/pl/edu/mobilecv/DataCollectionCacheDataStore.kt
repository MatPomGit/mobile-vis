package pl.edu.mobilecv

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Persistent storage for telemetry and sync metadata using Jetpack DataStore.
 *
 * This implementation replaces the in-memory ConcurrentHashMap to ensure
 * that error counts and sync states survive app restarts and to avoid
 * main-thread timeouts during intensive data operations.
 */
class DataCollectionCacheDataStore(private val context: Context) {

    companion object {
        private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "data_collection_cache")
        private val LAST_SYNC_TIME = longPreferencesKey("last_sync_time")
        private const val ERROR_PREFIX = "error_count_"
    }

    /**
     * Increments the error count for a specific category.
     */
    suspend fun incrementErrorCount(scope: String, category: String) {
        val key = stringPreferencesKey("$ERROR_PREFIX$scope:$category")
        context.dataStore.edit { preferences ->
            val current = (preferences[key]?.toInt() ?: 0)
            preferences[key] = (current + 1).toString()
        }
    }

    /**
     * Retrieves the map of all recorded errors and their counts.
     */
    val allErrors: Flow<Map<String, Int>> = context.dataStore.data.map { preferences ->
        preferences.asMap()
            .filter { it.key.name.startsWith(ERROR_PREFIX) }
            .mapKeys { it.key.name.removePrefix(ERROR_PREFIX) }
            .mapValues { it.value.toString().toIntOrNull() ?: 0 }
    }

    /**
     * Sets the timestamp of the last successful synchronization.
     */
    suspend fun updateLastSyncTime(timestamp: Long) {
        context.dataStore.edit { preferences ->
            preferences[LAST_SYNC_TIME] = timestamp
        }
    }

    /**
     * Returns the timestamp of the last successful synchronization.
     */
    val lastSyncTime: Flow<Long> = context.dataStore.data.map { preferences ->
        preferences[LAST_SYNC_TIME] ?: 0L
    }

    /**
     * Clears all recorded error counts after a successful sync.
     */
    suspend fun clearErrorCounts() {
        context.dataStore.edit { preferences ->
            val keysToRemove = preferences.asMap().keys.filter { it.name.startsWith(ERROR_PREFIX) }
            keysToRemove.forEach { preferences.remove(it) }
        }
    }
}
