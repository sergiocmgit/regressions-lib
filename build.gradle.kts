plugins {
	kotlin("jvm") version "1.9.10"
}

group = "com.scmoreno.regression"
version = "0.0.1-SNAPSHOT"

repositories {
	mavenCentral()
}

dependencies {
	implementation("org.jetbrains.kotlinx:multik-core:0.2.2")
	implementation("org.jetbrains.kotlinx:multik-default:0.2.2")

	testImplementation("org.springframework.boot:spring-boot-starter-test:3.1.4")
}

kotlin {
	jvmToolchain(17)
}

tasks.withType<Test> {
	useJUnitPlatform()
}