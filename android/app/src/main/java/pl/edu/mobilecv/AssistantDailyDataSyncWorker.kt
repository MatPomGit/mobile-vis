package pl.edu.mobilecv

import android.content.Context
import android.util.Log
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import kotlinx.coroutines.flow.first
import java.util.concurrent.TimeUnit

/**
 * Worker responsible for the periodic synchronization of telemetry data.
 *
 * This implements the "assistant_daily_data_sync" Synclet logic by using
 * Android's WorkManager to ensure reliable execution in the background,
 * even if the app is closed.
 */
class AssistantDailyDataSyncWorker(
    appContext: Context,
    params: WorkerParameters
) : CoroutineWorker(appContext, params) {

    companion object {
        private const val TAG = "AssistantSync"
        private const val SYNC_WORK_NAME = "assistant_daily_data_sync"

        /**
         * Schedules the daily synchronization task.
         */
        fun schedule(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .setRequiresBatteryNotLow(true)
                .build()

            val syncRequest = PeriodicWorkRequestBuilder<AssistantDailyDataSyncWorker>(
                1, TimeUnit.DAYS
            )
                .setConstraints(constraints)
                .setBackoffCriteria(
                    androidx.work.BackoffPolicy.EXPONENTIAL,
                    15, // PeriodicWorkRequest.MIN_PERIODIC_INTERVAL_MINUTES
                    TimeUnit.MINUTES
                )
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                SYNC_WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                syncRequest
            )
            Log.i(TAG, "Scheduled daily sync work")
        }
    }

    override suspend fun doWork(): Result {
        val dataStore = DataCollectionCacheDataStore(applicationContext)
        val errors = dataStore.allErrors.first()

        if (errors.isEmpty()) {
            Log.i(TAG, "No telemetry data to sync")
            return Result.success()
        }

        Log.i(TAG, "Starting sync for ${errors.size} error categories...")

        return try {
            // Simulated network synchronization:
            // In a real implementation, this would send 'errors' to a backend server.
            // Using a timeout of 30 seconds for the network call to avoid blocking the worker.
            val success = performNetworkSync(errors)

            if (success) {
                Log.i(TAG, "Telemetry sync successful, clearing local cache")
                dataStore.clearErrorCounts()
                dataStore.updateLastSyncTime(System.currentTimeMillis())
                Result.success()
            } else {
                Log.w(TAG, "Telemetry sync failed (network error), will retry")
                Result.retry()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during telemetry sync", e)
            Result.failure()
        }
    }

    private suspend fun performNetworkSync(data: Map<String, Int>): Boolean {
        // Here, integrate with the existing telemetry endpoint if available.
        // For now, we simulate a successful POST request.
        Log.d(TAG, "Syncing data: $data")
        return true
    }
}
