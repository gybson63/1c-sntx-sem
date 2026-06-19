plugins {
    id("java")
    id("application")
}

group = "com.sntxsem"
version = "0.1.0"

repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    implementation("com.github.1c-syntax:bsl-context:0.7.0")
    implementation("info.picocli:picocli:4.7.6")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.17.2")
    annotationProcessor("info.picocli:picocli-codegen:4.7.6")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

application {
    mainClass = "com.sntxsem.exporter.ExporterMain"
}

tasks.jar {
    archiveBaseName.set("bsl-context-exporter")
    manifest {
        attributes["Main-Class"] = "com.sntxsem.exporter.ExporterMain"
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })
}
