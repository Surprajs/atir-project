import RPi.GPIO as GPIO
import time

class PTUController:
    
    def __init__(self, WIDTH, HEIGHT):
        self.width = WIDTH/2
        self.height = HEIGHT/2
        self.treshold = 100
        self.dutyX = 7
        self.dutyY = 7
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(12,GPIO.OUT)
        GPIO.setup(33,GPIO.OUT)
        self.servoX = GPIO.PWM(12,50)
        self.servoY = GPIO.PWM(33,50)
        self.servoX.start(0)
        self.servoY.start(0)
        
    def __del__(self):
        self.servoX.stop()
        self.servoY.stop()
        GPIO.cleanup()
        
        
    def current_milli_time(self):
        return round(time.time() * 1000)
        
    
    def track(self, posX, posY):
        startTime = self.current_milli_time()
        margin = 30
        
        print(posX, posY)
        print(self.width - posX)

        #180
        if self.width - posX > margin:
            self.dutyX += 0.5
            if self.dutyX < 12:
                self.servoX.ChangeDutyCycle(self.dutyX)
            else:
                self.dutyX = 12
            

            
        if self.width - posX < -margin:
            self.dutyX -= 0.5
            if self.dutyX > 2:
                self.servoX.ChangeDutyCycle(self.dutyX)
            else:
                self.dutyX = 2
            
            
        if self.height - posY < -margin:
            self.dutyY += 1
            if self.dutyY < 10:
                self.servoY.ChangeDutyCycle(self.dutyY)
            else:
                self.dutyY = 10
            
        
            
        if self.height - posY > margin:
            self.dutyY -= 1
            if self.dutyY > 4:
                self.servoY.ChangeDutyCycle(self.dutyY)
            else:
                self.dutyY = 4    
        
        
        
        #360
        #if self.width - posX > margin:
        #    self.dutyX = 6.5
        #    self.servoX.ChangeDutyCycle(self.dutyX)
        #    print("pierwszy")

            
        #if self.width - posX < -margin:
        #    self.dutyX = 7.5
        #    self.servoX.ChangeDutyCycle(self.dutyX)
        #    print("drugi")

            
        #if self.height - posY > margin:
        #    self.dutyY = 8

            
        #if self.height - posY < margin:
        #    self.dutyY = 6

            
        
        #self.servoY.ChangeDutyCycle(self.dutyY)
        
        while self.current_milli_time() - startTime <= self.treshold:
            pass
            
        self.servoX.ChangeDutyCycle(0)
        self.servoY.ChangeDutyCycle(0)
        #self.servoY.stop()
            
        #endTime = self.current_milli_time()
        #while self.current_milli_time() - endTime <= 100:
        #    pass
        
        
        

