@rem Gradle startup script for Windows
@rem Requires: Java 17+ on PATH and gradle-wrapper.jar (see gradlew comment).

@if "%DEBUG%"=="" @echo off
setlocal

set JAVA_EXE=java.exe
if defined JAVA_HOME set JAVA_EXE=%JAVA_HOME%\bin\java.exe

set CLASSPATH=%~dp0gradle\wrapper\gradle-wrapper.jar
"%JAVA_EXE%" -Xmx64m -Xms64m -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
