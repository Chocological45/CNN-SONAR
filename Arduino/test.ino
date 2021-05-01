#include "Arduino.h"
#include "NewPing.h"

#define TR19 6
#define ECHO 7
#define MAX_DIST 150

NewPing sonar(TR19, ECHO, MAX_DIST);

void setup() {
    Serial.begin(9600);

}


void loop() {
    delay(100);
    float cm = sonar.ping_cm();
    Serial.println(cm);
    float mm = sonar.ping();
    mm = mm * 343 / 2000;
    Serial.println(mm);
  
}


/*void setup() {
    Serial.begin(9600); // Starting Serial Terminal
    
    
    pinMode(6, OUTPUT);   
    pinMode(7, INPUT);
 
}


void loop() {
    float d1;
    digitalWrite(6, LOW);
    delayMicroseconds(2);
    digitalWrite(6, HIGH);
    delayMicroseconds(10);
    digitalWrite(6, LOW);
    d1 = pulseIn(7, HIGH);
    //d1=d1*343/2000;
    Serial.println(d1);
     
}
*/
