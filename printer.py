
from tasks import getTasksJson
import random
class Printer:
    def __init__(self, data):
        self.data = data
    def print(self):
        length = len(self.data)
        json = getTasksJson()
        sample = random.sample(json, length)
        for i in range(length):
            self.data[i]['name'] = sample[i]['name']
        

        
        