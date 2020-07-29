import time

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.clock()
        print(self.label+': entering')
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print(self.label + ': ' + format(self.interval,'.03f'))
