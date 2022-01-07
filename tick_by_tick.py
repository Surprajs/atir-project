#tick by tick version
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