package pl.edu.mobilecv.odometry

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Testy polityki auto-recovery dla silników odometrii.
 */
class OdometryRecoveryPolicyTest {

    /**
     * Gdy przez kilka klatek brakuje punktów, silnik VO powinien przejść w re-detekcję/degradację.
     */
    @Test
    fun `visual odometry should degrade after repeated low feature frames`() {
        val engine = VisualOdometryEngine()

        val first = engine.decideRecoveryAction(pointsCount = 0, inlierRatio = 0.0)
        val second = engine.decideRecoveryAction(pointsCount = 0, inlierRatio = 0.0)

        assertEquals(VisualOdometryEngine.RecoveryAction.NONE, first)
        assertEquals(VisualOdometryEngine.RecoveryAction.REDETECT_AND_DEGRADE, second)
    }

    /**
     * Nagły i utrzymany spadek inlier ratio powinien uruchomić lokalny restart trackera.
     */
    @Test
    fun `full odometry should request local tracker restart after inlier drop`() {
        val engine = FullOdometryEngine()

        val stable = engine.decideRecoveryAction(tracksCount = 60, inlierRatio = 0.7)
        val low1 = engine.decideRecoveryAction(tracksCount = 60, inlierRatio = 0.05)
        val low2 = engine.decideRecoveryAction(tracksCount = 60, inlierRatio = 0.06)
        val low3 = engine.decideRecoveryAction(tracksCount = 60, inlierRatio = 0.08)

        assertEquals(FullOdometryEngine.RecoveryAction.NONE, stable)
        assertEquals(FullOdometryEngine.RecoveryAction.NONE, low1)
        assertEquals(FullOdometryEngine.RecoveryAction.NONE, low2)
        assertEquals(FullOdometryEngine.RecoveryAction.LOCAL_TRACKER_RESTART, low3)
    }

    /**
     * Seria złych ramek nie może powodować nieobsłużonych wyjątków w logice recover.
     */
    @Test
    fun `recovery policy should stay deterministic during long failure streak`() {
        val engine = VisualOdometryEngine()
        repeat(20) {
            engine.decideRecoveryAction(pointsCount = 0, inlierRatio = 0.0)
        }

        val recovery = engine.decideRecoveryAction(pointsCount = 120, inlierRatio = 0.5)
        assertEquals(VisualOdometryEngine.RecoveryAction.NONE, recovery)
    }
}
