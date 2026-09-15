from threading import Thread

class ReturnableThread(Thread):
    def __init__(self, target, familyTrainDf, familyTestDf, storeFamily, queue):
        super().__init__()
        self.target = target
        self.args = [familyTrainDf, familyTestDf, storeFamily]
        self.queue = queue

    def run(self):
        self.queue.put(self.target(*self.args))