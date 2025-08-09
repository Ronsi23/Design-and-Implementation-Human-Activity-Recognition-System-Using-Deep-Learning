#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <espnow.h>
#include <ESP8266WiFi.h>
#include "MAX30105.h"
#include "heartRate.h"

Adafruit_MPU6050 mpu;
MAX30105 particleSensor;

typedef struct {
  unsigned long timestamp;
  float accX, accY, accZ;
  float gyroX, gyroY, gyroZ;
  float bpm;
  int beatAvg;
  bool fingerDetected;
} SensorData;

SensorData data;
uint8_t receiverMAC[] = {0x94, 0x3C, 0xC6, 0x33, 0x17, 0x9C}; // GANTI DENGAN MAC ADDRESS Wemos D1 Mini/ESP32 penerima Anda!

// Heart rate variables
const byte RATE_SIZE = 5;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;

const int LED_PIN = 2;
unsigned long lastSend = 0, totalSent = 0, successCount = 0;
const unsigned long INTERVAL = 20; // 50Hz

void OnDataSent(uint8_t *mac_addr, uint8_t status) {
  if (status == 0) successCount++;
  if (totalSent % 50 == 0)
    Serial.printf("Sent: %lu, Success: %.1f%%, Last: %s\n", totalSent,
      (float)successCount / totalSent * 100.0, status == 0 ? "✅" : "❌");
}

void updateHeartRate() {
  long irValue = particleSensor.getIR();

  if (checkForBeat(irValue) == true) {
    long delta = millis() - lastBeat;
    lastBeat = millis();
    beatsPerMinute = 60 / (delta / 1000.0);
    
    if (beatsPerMinute < 255 && beatsPerMinute > 20) {
      rates[rateSpot++] = (byte)beatsPerMinute;
      rateSpot %= RATE_SIZE;
      
      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++)
        beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }
  
  data.fingerDetected = (irValue > 50000);
  data.bpm = beatsPerMinute;
  data.beatAvg = beatAvg;
}

void blinkError() {
  digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  delay(200);
}

void printStats() {
  Serial.printf("\n📈 Statistik Pengiriman:\n");
  Serial.printf("Total Dikirim: %lu\n", totalSent);
  Serial.printf("Berhasil: %lu (%.1f%%)\n", successCount,
                totalSent > 0 ? (float)successCount / totalSent * 100.0 : 0);
  Serial.printf("Gagal: %lu\n", totalSent - successCount);
  Serial.printf("BPM Saat Ini: %.1f, Rata-rata: %d, Jari Terdeteksi: %s\n",
                beatsPerMinute, beatAvg, data.fingerDetected ? "Ya" : "Tidak");
  Serial.printf("Uptime: %lus\n", millis() / 1000);
}

void testSensors() {
  Serial.println("\n🔍 Menguji sensor...");
  for (int i = 0; i < 5; i++) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    long irValue = particleSensor.getIR();
    
    Serial.printf("%d: Acc(%.2f,%.2f,%.2f) Gyro(%.2f,%.2f,%.2f) IR=%ld BPM=%.1f\n",
      i + 1, a.acceleration.x, a.acceleration.y, a.acceleration.z,
      g.gyro.x, g.gyro.y, g.gyro.z, irValue, beatsPerMinute);
    delay(200);
  }
  Serial.println("Pengujian selesai.");
}

void printHelp() {
  Serial.println("\n📋 Perintah yang tersedia:");
  Serial.println("  stats  - Tampilkan statistik pengiriman dan sensor");
  Serial.println("  test   - Uji pembacaan sensor (MPU6050 & MAX30105)");
  Serial.println("  help   - Tampilkan daftar perintah ini");
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (!mpu.begin()) {
    Serial.println("❌ Gagal menginisialisasi MPU6050!");
    while (1) blinkError();
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("✅ MPU6050 OK!");

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("❌ Gagal menginisialisasi MAX30105! Periksa koneksi/power.");
    while (1) blinkError();
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);
  Serial.println("✅ MAX30105 OK!");

  if (esp_now_init() != 0) {
    Serial.println("❌ Gagal menginisialisasi ESP-NOW!");
    while (1) blinkError();
  }
  esp_now_set_self_role(ESP_NOW_ROLE_CONTROLLER);
  esp_now_register_send_cb(OnDataSent);

  if (esp_now_add_peer(receiverMAC, ESP_NOW_ROLE_SLAVE, 1, NULL, 0) != 0) {
    Serial.println("❌ Gagal menambahkan peer!");
    while (1) blinkError();
  }
  Serial.println("✅ ESP-NOW Peer berhasil ditambahkan!");
  Serial.println("\nSiap mengirim data. Letakkan jari di sensor detak jantung...");
  digitalWrite(LED_PIN, LOW);
  delay(1000);
}

void loop() {
  unsigned long now = millis();

  if (now - lastSend >= INTERVAL) {
    // 1. Baca Data MPU6050
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);

    // 2. Perbarui Data Detak Jantung dari MAX30105
    updateHeartRate();

    // 3. Isi Struktur Data untuk Dikirim
    data.timestamp = now;
    data.accX = a.acceleration.x;
    data.accY = a.acceleration.y;
    data.accZ = a.acceleration.z;
    data.gyroX = g.gyro.x;
    data.gyroY = g.gyro.y;
    data.gyroZ = g.gyro.z;

    // 4. Kirim Data via ESP-NOW
    esp_now_send(receiverMAC, (uint8_t *)&data, sizeof(data));
    totalSent++;
    lastSend = now;

    // 5. Output Debug ke Serial Monitor (setiap 50 paket)
    // BARIS INI YANG DIUBAH UNTUK MEMISAHKAN BPM DAN AVG BPM
    if (totalSent % 50 == 1) {
      Serial.printf("Acc(%.2f,%.2f,%.2f) Gyro(%.2f,%.2f,%.2f) BPM=%.1f Avg BPM=%d Jari=%s\n",
                    data.accX, data.accY, data.accZ,
                    data.gyroX, data.gyroY, data.gyroZ,
                    data.bpm, data.beatAvg, // data.bpm adalah BPM instan, data.beatAvg adalah rata-rata BPM
                    data.fingerDetected ? "✅" : "❌");
    }
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "stats") {
      printStats();
    } else if (cmd == "test") {
      testSensors();
    } else if (cmd == "help") {
      printHelp();
    } else {
      Serial.println("Perintah tidak dikenal. Ketik 'help' untuk daftar perintah.");
    }
  }

  yield();
}
