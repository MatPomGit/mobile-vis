package pl.edu.mobilecv

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Didactic activity that explains the visual-odometry pipeline step by step.
 *
 * Eight stage cards are shown in a scrollable list, each describing one phase of
 * the monocular visual-odometry process:
 *
 * 1. Image acquisition
 * 2. Feature detection (Shi-Tomasi / FAST)
 * 3. Feature tracking (Lucas-Kanade optical flow)
 * 4. Motion estimation (Essential Matrix + RANSAC + SVD)
 * 5. Triangulation and 3-D reconstruction
 * 6. Trajectory and map accumulation (pose composition)
 * 7. From pixel to 3-D world point (camera matrix K, ray casting, parallax)
 * 8. Robot control using full odometry (velocity, navigation)
 *
 * A button at the bottom launches [MainActivity] pre-set to [AnalysisMode.ODOMETRY]
 * so the user can immediately try the real pipeline after reading the theory.
 */
class OdometryTutorialActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "OdometryTutorial"
    }

    /**
     * Represents a single stage in the visual-odometry tutorial.
     *
     * @param stepNumber  1-based index shown in the badge.
     * @param emoji       Unicode emoji icon rendered before the title.
     * @param title       Short stage name.
     * @param description Multi-sentence explanation of what happens in this stage.
     */
    private data class OdometryStage(
        val stepNumber: Int,
        val emoji: String,
        val title: String,
        val description: String,
    )

    /** Lazily built list of all tutorial stages. */
    private val stages: List<OdometryStage> by lazy {
        listOf(
            OdometryStage(
                stepNumber = 1,
                emoji = getString(R.string.tutorial_stage1_emoji),
                title = getString(R.string.tutorial_stage1_title),
                description = getString(R.string.tutorial_stage1_desc),
            ),
            OdometryStage(
                stepNumber = 2,
                emoji = getString(R.string.tutorial_stage2_emoji),
                title = getString(R.string.tutorial_stage2_title),
                description = getString(R.string.tutorial_stage2_desc),
            ),
            OdometryStage(
                stepNumber = 3,
                emoji = getString(R.string.tutorial_stage3_emoji),
                title = getString(R.string.tutorial_stage3_title),
                description = getString(R.string.tutorial_stage3_desc),
            ),
            OdometryStage(
                stepNumber = 4,
                emoji = getString(R.string.tutorial_stage4_emoji),
                title = getString(R.string.tutorial_stage4_title),
                description = getString(R.string.tutorial_stage4_desc),
            ),
            OdometryStage(
                stepNumber = 5,
                emoji = getString(R.string.tutorial_stage5_emoji),
                title = getString(R.string.tutorial_stage5_title),
                description = getString(R.string.tutorial_stage5_desc),
            ),
            OdometryStage(
                stepNumber = 6,
                emoji = getString(R.string.tutorial_stage6_emoji),
                title = getString(R.string.tutorial_stage6_title),
                description = getString(R.string.tutorial_stage6_desc),
            ),
            OdometryStage(
                stepNumber = 7,
                emoji = getString(R.string.tutorial_stage7_emoji),
                title = getString(R.string.tutorial_stage7_title),
                description = getString(R.string.tutorial_stage7_desc),
            ),
            OdometryStage(
                stepNumber = 8,
                emoji = getString(R.string.tutorial_stage8_emoji),
                title = getString(R.string.tutorial_stage8_title),
                description = getString(R.string.tutorial_stage8_desc),
            ),
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_odometry_tutorial)

        findViewById<Button>(R.id.btnTutorialBack).setOnClickListener { finish() }

        val container = findViewById<LinearLayout>(R.id.tutorialStageContainer)
        stages.forEach { stage -> addStageCard(container, stage) }

        findViewById<Button>(R.id.btnLaunchOdometry).setOnClickListener {
            startActivity(
                Intent(this, MainActivity::class.java).apply {
                    putExtra(MenuActivity.EXTRA_MODE, AnalysisMode.ODOMETRY.name)
                }
            )
        }
    }

    /**
     * Inflates [R.layout.item_tutorial_stage_card] for [stage] and appends it to [container].
     *
     * @param container Parent [LinearLayout] inside the tutorial's [android.widget.ScrollView].
     * @param stage     Data for the stage to render.
     */
    private fun addStageCard(container: LinearLayout, stage: OdometryStage) {
        val card = layoutInflater.inflate(R.layout.item_tutorial_stage_card, container, false)
        card.findViewById<TextView>(R.id.textStepBadge).text =
            getString(R.string.tutorial_stage_label, stage.stepNumber)
        card.findViewById<TextView>(R.id.textStageEmoji).text = stage.emoji
        card.findViewById<TextView>(R.id.textStageTitle).text = stage.title
        card.findViewById<TextView>(R.id.textStageDescription).text = stage.description
        container.addView(card)
    }
}
