// dual_servo_serial_float_decimal.ino
#include <Servo.h>
#include <stdlib.h>   // strtod
#include <ctype.h>    // toupper, tolower

// ─── Servo pins ───────────────────────────────────────────────────────────────
const uint8_t PIN_PITCH = 7;   // pitch 서보
const uint8_t PIN_YAW   = 9;   // yaw 서보

// ─── Laser (transistor / MOSFET gate) pin ────────────────────────────────────
const int SW_PIN = 8;          // 트랜지스터 베이스(또는 MOSFET 게이트)

// 필요 시 각도 보정(도 단위, 소수 허용)
float OFFSET_PITCH = 0.0f;
float OFFSET_YAW   = 0.0f;

// 물리적 한계(도 단위, 소수 허용)
const float MIN_ANG = 0.0f;
const float MAX_ANG = 180.0f;

// 펄스 폭(μs) 매핑 범위 (필요시 MG996R에 맞춰 보정 가능)
const int US_MIN = 544;    // 0°
const int US_MAX = 2400;   // 180°

Servo servoPitch, servoYaw;

// 서보에 실제 쓴 각도(도) — 오프셋/클램프 반영 후 값
float curPitch = 90.0f, curYaw = 90.0f;

// 레이저 상태
bool laserState = false;

// ─── Util: angle/servo helpers ───────────────────────────────────────────────
static inline float clampAngle(float a) {
  if (a < MIN_ANG) a = MIN_ANG;
  if (a > MAX_ANG) a = MAX_ANG;
  return a;
}

static inline int degToUs(float deg) {
  deg = clampAngle(deg);
  float t = deg / 180.0f;
  int us = (int)(US_MIN + (US_MAX - US_MIN) * t + 0.5f);
  return us;
}

void applyAngles(float p_logic, float y_logic) {
  float p_servo_deg = clampAngle(p_logic + OFFSET_PITCH);
  float y_servo_deg = clampAngle(y_logic + OFFSET_YAW);
  servoPitch.writeMicroseconds(degToUs(p_servo_deg));
  servoYaw.writeMicroseconds(degToUs(y_servo_deg));
  curPitch = p_servo_deg;
  curYaw   = y_servo_deg;
}

// ─── Laser helpers ───────────────────────────────────────────────────────────
void setLaser(bool on) {
  digitalWrite(SW_PIN, on ? HIGH : LOW);
  laserState = on;
}

void laserBlinkOnce(unsigned long onMs = 1000, unsigned long offMs = 1000) {
  setLaser(true);
  delay(onMs);
  setLaser(false);
  delay(offMs);
}

// ─── Parser helpers ──────────────────────────────────────────────────────────
// 공백을 건너뛰고 float 하나 파싱 (AVR에서 scanf %f 비권장 → strtod 사용)
static bool parseFloatAfter(const char* &s, float &out) {
  while (*s == ' ') s++;
  char* endp;
  double v = strtod(s, &endp); // AVR에서 double == float
  if (endp == s) return false;
  out = (float)v;
  s = endp;
  return true;
}

// 공백을 건너뛰고 bool 토큰 파싱 (true/false, on/off, high/low, 1/0)
static bool parseBoolAfter(const char* &s, bool &out) {
  while (*s == ' ') s++;
  if (*s == '\0') return false;

  // 숫자 1/0 빠른 처리
  if (*s == '1') { out = true;  s++; return true; }
  if (*s == '0') { out = false; s++; return true; }

  // 단어 토큰 읽기
  char buf[8]; // "false"도 충분
  int i = 0;
  while (*s && *s != ' ' && i < (int)(sizeof(buf)-1)) {
    buf[i++] = (char)tolower(*s++);
  }
  buf[i] = '\0';

  if (!i) return false;

  if (!strcmp(buf,"true") || !strcmp(buf,"t") || !strcmp(buf,"on")  || !strcmp(buf,"high")) { out = true;  return true; }
  if (!strcmp(buf,"false")|| !strcmp(buf,"f") || !strcmp(buf,"off") || !strcmp(buf,"low"))  { out = false; return true; }
  return false;
}

// ─── Arduino lifecycle ───────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // Servo
  servoPitch.attach(PIN_PITCH);
  servoYaw.attach(PIN_YAW);
  delay(300);                // 전원 안정화
  applyAngles(90.0f, 90.0f); // 센터

  // Laser
  pinMode(SW_PIN, OUTPUT);
  setLaser(false);           // 부팅 시 기본 꺼짐

  Serial.println("READY dual-servo v1-float + laser");
  Serial.print("OK C ");
  Serial.print(curPitch, 2); Serial.print(' ');
  Serial.print(curYaw, 2);   Serial.print(' ');
  Serial.println(laserState ? 1 : 0);
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  const char* s = line.c_str();
  char cmd = toupper(*s);
  s++; // 명령 문자 다음으로 이동

  if (cmd == 'S') {                     // 절대: 두 축 (S p y)
    float p, y;
    if (parseFloatAfter(s, p) && parseFloatAfter(s, y)) {
      applyAngles(p, y);
      Serial.print("OK S ");
      Serial.print(curPitch, 2); Serial.print(' ');
      Serial.println(curYaw, 2);
    } else {
      Serial.println("ERR S format");
    }
  }
  else if (cmd == 'P') {                // 절대: pitch만 (P p)
    float p;
    if (parseFloatAfter(s, p)) {
      float y_logic = curYaw - OFFSET_YAW;    // yaw 유지
      applyAngles(p, y_logic);
      Serial.print("OK P ");
      Serial.println(curPitch, 2);
    } else {
      Serial.println("ERR P format");
    }
  }
  else if (cmd == 'Y') {                // 절대: yaw만 (Y y)
    float y;
    if (parseFloatAfter(s, y)) {
      float p_logic = curPitch - OFFSET_PITCH; // pitch 유지
      applyAngles(p_logic, y);
      Serial.print("OK Y ");
      Serial.println(curYaw, 2);
    } else {
      Serial.println("ERR Y format");
    }
  }
  else if (cmd == 'C') {                // 센터
    applyAngles(90.0f, 90.0f);
    Serial.print("OK C ");
    Serial.print(curPitch, 2); Serial.print(' ');
    Serial.print(curYaw, 2);   Serial.print(' ');
    Serial.println(laserState ? 1 : 0);
  }
  else if (cmd == 'Q') {                // 상태 질의: pitch yaw laser
    Serial.print("STATE ");
    Serial.print(curPitch, 2); Serial.print(' ');
    Serial.print(curYaw, 2);   Serial.print(' ');
    Serial.println(laserState ? 1 : 0);
  }
  else if (cmd == 'L') {                // Laser on/off (L true|false / 1|0 / on|off)
    bool on;
    if (parseBoolAfter(s, on)) {
      setLaser(on);
      Serial.print("OK L ");
      Serial.println(laserState ? 1 : 0);
    } else {
      Serial.println("ERR L format");
    }
  }
  else if (cmd == 'B') {                // (옵션) 1회 깜빡임 테스트 (B [on_ms] [off_ms])
    // 인자가 없으면 1000/1000ms 사용
    float onMs = 1000, offMs = 1000;
    // 두 값 모두 존재할 때만 반영
    const char* s2 = s;
    float tmp1, tmp2;
    if (parseFloatAfter(s2, tmp1) && parseFloatAfter(s2, tmp2)) {
      onMs = tmp1; offMs = tmp2;
    }
    laserBlinkOnce((unsigned long)onMs, (unsigned long)offMs);
    Serial.println("OK B");
  }
  else {
    Serial.println("ERR unknown cmd");
  }
}