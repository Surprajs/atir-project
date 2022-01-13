import RPi.GPIO as GPIO
import time

class PTUController:
    
    def __init__(self, WIDTH, HEIGHT):
        self.width = WIDTH/2
        self.height = HEIGHT/2
        self.treshold = 100
        self.dutyX = 7
        self.dutyY = 7
        self.TickX = 84
        self.TickY = 78
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
        x = int((self.width - posX)/self.TickX)
        y = int((self.height - posY)/self.TickY)
        
        self.dutyX += (0.25*x) 
        if self.dutyX > 12:
            self.dutyX = 12
        elif self.dutyX < 2:
            self.dutyX = 2


        self.dutyY -= (0.25*y) 
        if self.dutyY > 10:
            self.dutyY = 10
        elif self.dutyY < 4:
            self.dutyY = 4

        self.servoY.ChangeDutyCycle(self.dutyY)    
        self.servoX.ChangeDutyCycle(self.dutyX)
               
        while self.current_milli_time() - startTime <= self.treshold:
            pass
            
        self.servoX.ChangeDutyCycle(0)
        self.servoY.ChangeDutyCycle(0)
        
        
        

