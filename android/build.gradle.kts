// Top-level build file – configuration shared across sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
}
val app_version_name by extra("1.14")
val app_version_code by extra(14)
