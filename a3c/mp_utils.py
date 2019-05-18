from torch.multiprocessing import Lock, Value

class Counter(object):
    def __init__(self, val=0):
        self.val = Value('i', val)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value
