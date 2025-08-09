#include <esp_now.h>
#include <WiFi.h>

typedef struct {
  unsigned long timestamp;
  float accX, accY, accZ;
  float gyroX, gyroY, gyroZ;
  float bpm;
  int beatAvg;
  bool fingerDetected;
} SensorData;

SensorData received;
const int LED_PIN = 2;
unsigned long lastLed = 0, totalRecv = 0, lastRecv = 0;
float dataRate = 0;
unsigned long rateBuf[10] = {20};
int rateIdx = 0;

// Heart rate statistics - Enhanced
float maxBPM = 0, minBPM = 999, avgBPM = 0;
float maxBeatAvg = 0, minBeatAvg = 999, sessionAvgBeatAvg = 0;
unsigned long heartBeatCount = 0;
unsigned long beatAvgCount = 0;

// Data validation counters
unsigned long validDataCount = 0;
unsigned long invalidDataCount = 0;

void onDataReceived(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  if (len != sizeof(SensorData)) {
    invalidDataCount++;
    return; // Validate packet size
  }
  
  memcpy(&received, incomingData, sizeof(received));
  unsigned long now = millis();
  totalRecv++;
  validDataCount++;
  
  // Calculate data rate
  if (totalRecv > 1) {
    rateBuf[rateIdx] = now - lastRecv;
    rateIdx = (rateIdx + 1) % 10;
    unsigned long sum = 0;
    for (int i = 0; i < 10; i++) sum += rateBuf[i];
    dataRate = sum > 0 ? 1000.0 / (sum / 10.0) : 0;
  }
  lastRecv = now;
  
  // Update instantaneous BPM statistics
  if (received.fingerDetected && received.bpm > 20 && received.bpm < 255) {
    heartBeatCount++;
    if (received.bpm > maxBPM) maxBPM = received.bpm;
    if (received.bpm < minBPM) minBPM = received.bpm;
    avgBPM = ((avgBPM * (heartBeatCount - 1)) + received.bpm) / heartBeatCount;
  }
  
  // Update beat average statistics
  if (received.beatAvg > 20 && received.beatAvg < 255) {
    beatAvgCount++;
    if (received.beatAvg > maxBeatAvg) maxBeatAvg = received.beatAvg;
    if (received.beatAvg < minBeatAvg) minBeatAvg = received.beatAvg;
    sessionAvgBeatAvg = ((sessionAvgBeatAvg * (beatAvgCount - 1)) + received.beatAvg) / beatAvgCount;
  }
  
  // Output enhanced CSV format - SESUAI DENGAN WEMOS FORMAT
  Serial.printf("%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%d,%d,%.1f\n",
                received.timestamp, 
                received.accX, received.accY, received.accZ,
                received.gyroX, received.gyroY, received.gyroZ,
                received.bpm, received.beatAvg, 
                received.fingerDetected ? 1 : 0,
                dataRate);
}

void setup() {
  Serial.begin(115200);
  
  // Clear serial buffer
  while(Serial.available()) Serial.read();
  delay(500);
  
  pinMode(LED_PIN, OUTPUT); 
  digitalWrite(LED_PIN, LOW);
  
  WiFi.mode(WIFI_STA); 
  WiFi.disconnect();
  
  if (esp_now_init() != ESP_OK) {
    Serial.println("❌ ESP-NOW failed!");
    while (1) blinkError();
  }
  
  esp_now_register_recv_cb(onDataReceived);
  
  // Print CSV header - SESUAI FORMAT PYTHON
  Serial.println("timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ,bpm,beatAvg,fingerDetected,dataRate");
  Serial.println("✅ ESP32 Receiver Ready - Enhanced Heart Rate Mode");
  Serial.println("📊 Compatible with Wemos D1 Mini smoothed BPM data");
  Serial.println("═══════════════════════════════════════════════════════");
  delay(1000);
}

void loop() {
  // Handle serial commands
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n'); 
    cmd.trim();
    
    if (cmd == "stats") printStats();
    else if (cmd == "reset") resetStats();
    else if (cmd == "help") printHelp();
    else if (cmd == "heart") printHeartStats();
    else if (cmd == "detailed") printDetailedStats();
    else if (cmd == "quality") printDataQuality();
  }
  
  // Blink LED every 500ms
  if (millis() - lastLed >= 500) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    lastLed = millis();
  }
  
  delay(10);
}

void printStats() {
  Serial.printf("\n📈 TRANSMISSION STATISTICS\n");
  Serial.printf("═══════════════════════════════════════\n");
  Serial.printf("📡 Received: %lu packets\n", totalRecv);
  Serial.printf("✅ Valid: %lu packets\n", validDataCount);
  Serial.printf("❌ Invalid: %lu packets\n", invalidDataCount);
  Serial.printf("📊 Success rate: %.1f%%\n", totalRecv > 0 ? (float)validDataCount / totalRecv * 100.0 : 0);
  Serial.printf("⚡ Data rate: %.1f Hz\n", dataRate);
  Serial.printf("⏱️  Last packet: %lums ago\n", millis() - lastRecv);
  Serial.printf("🕐 Uptime: %lus\n", millis() / 1000);
  Serial.printf("\n❤️  CURRENT VALUES:\n");
  Serial.printf("   Instantaneous BPM: %.1f\n", received.bpm);
  Serial.printf("   Beat Average: %d\n", received.beatAvg);
  Serial.printf("   Finger detected: %s\n", received.fingerDetected ? "✅ Yes" : "❌ No");
}

void printHeartStats() {
  Serial.printf("\n❤️  DETAILED HEART RATE STATISTICS\n");
  Serial.printf("═══════════════════════════════════════════════════════\n");
  
  Serial.printf("📊 INSTANTANEOUS BPM:\n");
  Serial.printf("   Current: %.1f BPM\n", received.bpm);
  Serial.printf("   Session Min: %.1f BPM\n", minBPM < 999 ? minBPM : 0);
  Serial.printf("   Session Max: %.1f BPM\n", maxBPM);
  Serial.printf("   Session Avg: %.1f BPM\n", avgBPM);
  Serial.printf("   Valid readings: %lu\n", heartBeatCount);
  
  Serial.printf("\n📈 BEAT AVERAGE (SMOOTHED):\n");
  Serial.printf("   Current: %d BPM\n", received.beatAvg);
  Serial.printf("   Session Min: %.0f BPM\n", minBeatAvg < 999 ? minBeatAvg : 0);
  Serial.printf("   Session Max: %.0f BPM\n", maxBeatAvg);
  Serial.printf("   Session Avg: %.1f BPM\n", sessionAvgBeatAvg);
  Serial.printf("   Valid readings: %lu\n", beatAvgCount);
  
  Serial.printf("\n👆 FINGER DETECTION:\n");
  Serial.printf("   Current status: %s\n", received.fingerDetected ? "✅ Detected" : "❌ Not detected");
  
  float fingerDetectionRate = totalRecv > 0 ? (float)heartBeatCount / totalRecv * 100.0 : 0;
  Serial.printf("   Detection rate: %.1f%% of packets\n", fingerDetectionRate);
}

void printDetailedStats() {
  Serial.printf("\n🔍 DETAILED SYSTEM STATISTICS\n");
  Serial.printf("═══════════════════════════════════════════════════════\n");
  
  Serial.printf("📡 TRANSMISSION:\n");
  Serial.printf("   Total packets: %lu\n", totalRecv);
  Serial.printf("   Valid packets: %lu (%.1f%%)\n", validDataCount, 
                totalRecv > 0 ? (float)validDataCount / totalRecv * 100.0 : 0);
  Serial.printf("   Invalid packets: %lu (%.1f%%)\n", invalidDataCount,
                totalRecv > 0 ? (float)invalidDataCount / totalRecv * 100.0 : 0);
  Serial.printf("   Current rate: %.1f Hz\n", dataRate);
  
  Serial.printf("\n📊 CURRENT SENSOR DATA:\n");
  Serial.printf("   Timestamp: %lu\n", received.timestamp);
  Serial.printf("   Accelerometer: (%.2f, %.2f, %.2f)\n", 
                received.accX, received.accY, received.accZ);
  Serial.printf("   Gyroscope: (%.2f, %.2f, %.2f)\n", 
                received.gyroX, received.gyroY, received.gyroZ);
  
  Serial.printf("\n⏱️  TIMING:\n");
  Serial.printf("   Last packet: %lums ago\n", millis() - lastRecv);
  Serial.printf("   System uptime: %lu seconds\n", millis() / 1000);
  Serial.printf("   Average packet interval: %.1fms\n", 
                totalRecv > 1 ? (float)(millis()) / totalRecv : 0);
}

void printDataQuality() {
  Serial.printf("\n🎯 DATA QUALITY ANALYSIS\n");
  Serial.printf("═══════════════════════════════════════════════════════\n");
  
  float successRate = totalRecv > 0 ? (float)validDataCount / totalRecv * 100.0 : 0;
  float fingerRate = totalRecv > 0 ? (float)heartBeatCount / totalRecv * 100.0 : 0;
  
  Serial.printf("📊 OVERALL QUALITY:\n");
  Serial.printf("   Packet success: %.1f%% ", successRate);
  if (successRate > 95) Serial.println("✅ Excellent");
  else if (successRate > 90) Serial.println("✅ Good");
  else if (successRate > 80) Serial.println("⚠️  Fair");
  else Serial.println("❌ Poor");
  
  Serial.printf("   Finger detection: %.1f%% ", fingerRate);
  if (fingerRate > 80) Serial.println("✅ Excellent");
  else if (fingerRate > 60) Serial.println("✅ Good");
  else if (fingerRate > 40) Serial.println("⚠️  Fair");
  else Serial.println("❌ Poor");
  
  Serial.printf("   Data rate stability: ");
  if (dataRate > 45 && dataRate < 55) Serial.println("✅ Excellent (near 50Hz)");
  else if (dataRate > 40 && dataRate < 60) Serial.println("✅ Good");
  else if (dataRate > 30 && dataRate < 70) Serial.println("⚠️  Fair");
  else Serial.println("❌ Poor");
  
  Serial.printf("\n📈 RECOMMENDATIONS:\n");
  if (successRate < 95) Serial.println("   • Check ESP-NOW connection stability");
  if (fingerRate < 60) Serial.println("   • Improve finger placement on heart rate sensor");
  if (dataRate < 40) Serial.println("   • Check transmitter power and processing load");
  if (successRate > 95 && fingerRate > 80 && dataRate > 45) 
    Serial.println("   ✅ System performing optimally!");
}

void resetStats() {
  totalRecv = 0; 
  validDataCount = 0;
  invalidDataCount = 0;
  dataRate = 0;
  heartBeatCount = 0;
  beatAvgCount = 0;
  maxBPM = 0; 
  minBPM = 999; 
  avgBPM = 0;
  maxBeatAvg = 0;
  minBeatAvg = 999;
  sessionAvgBeatAvg = 0;
  
  for (int i = 0; i < 10; i++) rateBuf[i] = 20;
  Serial.println("🔄 All statistics reset to zero.");
}

void printHelp() {
  Serial.println("\n📋 AVAILABLE COMMANDS:");
  Serial.println("═══════════════════════════════════════════════════════");
  Serial.println("stats     - Show basic transmission and heart rate stats");
  Serial.println("heart     - Show detailed heart rate statistics");
  Serial.println("detailed  - Show comprehensive system information");
  Serial.println("quality   - Show data quality analysis and recommendations");
  Serial.println("reset     - Reset all statistics to zero");
  Serial.println("help      - Show this command list");
  Serial.println("═══════════════════════════════════════════════════════");
  Serial.println("💡 TIP: Use 'quality' command to check system performance");
}

void blinkError() {
  digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  delay(150);
}
