from abc import ABCMeta, abstractmethod

class Fruit(metaclass=ABCMeta):
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def bloom(self):
        pass

class Apple(Fruit):
    def __init__(self, name):
        super().__init__(name)
    
    def bloom(self):
        print(f"{self.name} is blooming")

apple = Apple("Apple")
apple.bloom()