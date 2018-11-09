#define trigPin 13
#define echoPin 12
#define intPin A0
#include <Servo.h>
Servo servoMain;



String inputString = "";
String inputMode[2] = {"I","U"};
bool stringComplete = false;
long distance = 0; 
long intensity = 0;
int count = 0;

void setup() {
  Serial.begin (9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  digitalWrite(trigPin, LOW);
  digitalWrite(echoPin, LOW);
  inputString.reserve(200);
  servoMain.attach(10);
}

void loop() {
  //Serial.print("waiting\n");
   servoMain.write(95);
   delay(300);
  if(stringComplete){
  if(inputString == inputMode[0]){
    intensity = analogRead(intPin);
    Serial.print("Mode1");
  }

  //ultra sonic detection
    else if(inputString[0] == inputMode[1][0]){
    long duration;
    Serial.print(" ultra clicked \n");

    
    //move motor
    servoMain.write(0);
    delay(600);
    servoMain.write(95);
    
    while(true){
      digitalWrite(trigPin, LOW);  // Added this line
      delayMicroseconds(2); // Added this line
      digitalWrite(trigPin, HIGH);
      delayMicroseconds(10); // Added this line
      digitalWrite(trigPin, LOW);
      duration = pulseIn(echoPin, HIGH);
      distance = (duration/2) / 29.1;
      Serial.print(distance);
      Serial.print("\n");
        
        if(distance >= 15){ //threshold
        Serial.print("away\n");
        count ++;
      } 
      
      //detect 5 times far
      if (count > 5){ 
        Serial.print("DONE\n");
        break;
      }
    
    }
  }
  else Serial.print("wrong input");
  
  Serial.print(inputString);
  //sendback third read 2 python
  Serial.print("DONE\n");
  
  //motor turnback and back to default
  servoMain.write(180);
  delay(600);
  Serial.print("DONE\n");
  
  count = 0;
  inputString = "";
  stringComplete = false;
  distance = 0;
  }
}

void serialEvent(){
  if(Serial.available())
  {
  while (Serial.available()) {
    // get the new byte:
    delay(10);
    char inChar = (char)Serial.read();
    if(inChar != '\0')inputString += inChar;
    }
  Serial.print("received!\n");
  stringComplete = true;
  }
}

