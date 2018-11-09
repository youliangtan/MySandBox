#define trigPin 13
#define echoPin 12
#include <Servo.h>
Servo servoMain;

void setup() {
  Serial.begin (9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  servoMain.attach(10);
  
}

void loop() {
  long duration, distance;
  digitalWrite(trigPin, LOW);  // Added this line
  delayMicroseconds(2); // Added this line
  digitalWrite(trigPin, HIGH);
//  delayMicroseconds(1000); - Removed this line
  delayMicroseconds(10); // Added this line
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  distance = (duration/2) / 29.1;

  if (distance >= 200 || distance <= 0){
    Serial.println("Out of range");
    Serial.print(distance);
    Serial.println(" cm");
    servoMain.write(180);
    delay(100);
  }
  else {
    Serial.print(distance);
    Serial.println(" cm");
    servoMain.write(0);
    delay(100);
  }

  servoMain.write(95);
  delay(100);

}
