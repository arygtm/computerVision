#include <Servo.h>

String inString = "";    // string to hold input

#define flywheelPin 11

#define rollerPin 10
 
#define fanPin 6

#define turretPin 9

Servo fan;
Servo turret;


void setup() {

  pinMode(flywheelPin, OUTPUT);
  pinMode(rollerPin, OUTPUT);
  
  digitalWrite(flywheelPin, LOW);
  digitalWrite(rollerPin, LOW);

  fan.attach(fanPin);
  fan.write(770);
  delay(2000);

  
  
  // Open serial communications and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  Serial.println("init complete");

  
  turret.attach(turretPin);
  turret.write(90);
}

long long prevPrintTimeMs = 0;

int prevTurretPos = 90;

float turretPos = 90;

float turretVel = 0;

const int lashDeg = 20;

bool spinning = false;

bool shouldSpinDown = false;

long long spinTimerStart = 0;

 long long prevServoWriteTimeMs_ = 0;

long long prevMsgReceiveMs_ = 0;

const float angleScaleFactor_ = 100.0;

void writeServo(float servoPos, float servoVel){
  long long curTimeMs = millis();
  if( curTimeMs - prevServoWriteTimeMs_ > 20 && curTimeMs - prevMsgReceiveMs_ < 500){
    float targetAngle = (servoPos + servoVel * (curTimeMs - prevMsgReceiveMs_)/1000.0) * 599.0/60.0 + 600.0;
    
    turret.writeMicroseconds(targetAngle);
    /*Serial.print(turretPos);
    Serial.print(" ");
    Serial.print(turretVel);
    Serial.print(" ");
    Serial.println(targetAngle);*/
    //Serial.println(targetAngle);
    prevServoWriteTimeMs_ = curTimeMs;
  }
}

void loop() {
  // Read serial input:
  long long curTimeMs = millis();
  while (Serial.available() > 0) {
    
    int inChar = Serial.read();
    
    if (inChar == '\n'&& inString != "") {
      int spaceIndex = inString.indexOf(' ');

      if(spaceIndex == -1){
        Serial.println("INVALID SERVO WRITE MESSAGE");
        inString = "";
        return;
      }
      Serial.print(inString);
      
      Serial.print(". StringPos: ");
      turretPos = inString.substring(0,spaceIndex).toFloat()/angleScaleFactor_;
      Serial.print(inString.substring(0,spaceIndex));
      
      Serial.print(". String Vel: ");
      if(inString.indexOf('-') == -1){
        turretVel = inString.substring(spaceIndex + 1).toFloat()/angleScaleFactor_;
      Serial.print(inString.substring(spaceIndex + 1));
      } else {
        turretVel = inString.substring(spaceIndex + 1).toFloat()/angleScaleFactor_;
      }

      Serial.print(". TurretPos:  ");
      Serial.print(turretPos);
      Serial.print(". TurretVel: ");
      Serial.println(turretVel);
      
      inString = "";
      prevMsgReceiveMs_ = curTimeMs;
      
    } else if (isDigit(inChar) || inChar == ' ' || inChar == '-') {
      // convert the incoming byte to a char and add it to the string:
      inString += (char)inChar;
    } else if(inChar == 'w') {
      analogWrite(flywheelPin, 50);
      fan.write(2100);
      //Serial.println("spinningUp");
      spinning = true;
      shouldSpinDown = true;
    } else if(inChar == 's'){
      spinning = false;
      analogWrite(flywheelPin, 0);
      fan.write(770);
      //Serial.println("spinningDown");
    } else if(inChar == 'f'){
      digitalWrite(rollerPin, HIGH);
      delay(90);//TODO: Make this 90
      digitalWrite(rollerPin, LOW);
      //Serial.println("firing");
    } 
      
  }

  long long curPrintTimeMs = millis();
  if(curPrintTimeMs- prevPrintTimeMs > 1000){
    /*Serial.print(turretPos);
    Serial.print(" ");
    Serial.println(turretVel);*/
    
  }
  if(prevMsgReceiveMs_ != 0){
    writeServo(turretPos, turretVel);
  }
  
}
