#include "Arduino.h"
#include "NewPing.h"

#define TRIG_left 3
#define ECHO_left 2

#define TRIG_right 5
#define ECHO_right 4

#define MAX_DIST 400

NewPing left(TRIG_left, ECHO_left, MAX_DIST);     // Left Sensor
NewPing right(TRIG_right, ECHO_right, MAX_DIST);  // Right Sensor

float X, Y;

void setup() {
    Serial.begin(9600);
    
}

void loop() 
{
    float d1,d2, e1, e2, angle;

    // Distance L between the two sensor modules in millimetres
    float L = 78;
    
    // Get the echo time for the ping from the left sensor
    e1 = left.ping();
    // Convert
    d1 = e1 * 343/2000;
    
    delay(15); // Slow down data output

    // Get the echo time for the ping from the right sensor
    e2 = right.ping(); 
    d2 = e2 * 343/2000;


    //Serial.print(d1);
    //Serial.print("\t");
    //Serial.println(d2);
    angle = acos((((d1 * d1) + (L * L) - (d2 * d2))) / (2 * d1 * L));

    //if (d1 <= 1 && d1 >= -1) {
    if (angle < 3 && angle > 0) {
    //{
        //angle = acos((((d1 * d1) + (L * L) - (d2 * d2))) / (2 * d1 * L));
        X = d1 * cos(angle) + L/2;
        Y = d1 * sin(angle);
    
    } 
    else 
    {
        Serial.println("Out of Region!");
    
    }

    delay(15);
    
    Serial.print(X);
    Serial.print("\t");
    Serial.println(Y);
}
