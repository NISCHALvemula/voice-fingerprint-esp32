// Arduino Nano ESP32: runs the speaker-embedding model.
// Laptop sends one 40x121 log-mel window (float32); board returns the 64-d embedding.
#include <ArduTFLite.h>
#include "model_data.h"          // made by make_header.py

constexpr int IN_N  = 40 * 121;  // 4840 input floats
constexpr int OUT_N = 64;        // embedding size
static float inbuf[IN_N];
static byte*  arena = nullptr;
static bool   gotFrame = false;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);     // waits until bridge.py (or Serial Monitor) opens the port
  delay(500);
  Serial.println("BOOT");

  size_t arenaSize = 400 * 1024;
  if (psramFound()) { arena = (byte*)ps_malloc(arenaSize); Serial.println("PSRAM found"); }
  else Serial.println("PSRAM not found");
  if (!arena) { arenaSize = 180 * 1024; arena = (byte*)malloc(arenaSize); }
  if (!arena) { Serial.println("INIT FAIL: no memory for arena"); while (1) delay(1000); }
  Serial.print("arena bytes: "); Serial.println(arenaSize);

  Serial.println("initializing model...");
  if (!modelInit(model_data, arena, arenaSize)) {
    Serial.println("INIT FAIL: modelInit (arena too small or unsupported operator)");
    while (1) delay(1000);
  }
  Serial.println("model ok");
  Serial.setTimeout(5000);
}

void loop() {
  static uint32_t last = 0;
  if (!gotFrame && Serial.available() == 0 && millis() - last > 1000) {
    Serial.println("READY");
    last = millis();
  }
  if (Serial.available() < 2) return;
  if (Serial.peek() != 'S') { Serial.read(); return; }
  gotFrame = true;
  Serial.read();
  if (Serial.read() != 'T') return;

  size_t need = sizeof(inbuf);
  if (Serial.readBytes((char*)inbuf, need) != need) { Serial.write(0xEE); Serial.write(0xEE); return; }

  for (int i = 0; i < IN_N; i++) modelSetInput(inbuf[i], i);

  uint32_t t0 = micros();
  bool ok = modelRunInference();
  uint32_t dt = micros() - t0;
  if (!ok) { Serial.write(0xEE); Serial.write(0xEE); return; }

  float out[OUT_N];
  for (int i = 0; i < OUT_N; i++) out[i] = modelGetOutput(i);

  Serial.write(0xAA); Serial.write(0x55);        // reply header
  Serial.write((uint8_t*)&dt, 4);                // inference time, microseconds
  Serial.write((uint8_t*)out, OUT_N * 4);
}
