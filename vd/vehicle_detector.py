import collections


class VehicleDetector:
    def __init__(self, frames):
        self.queued_boxes = collections.deque(maxlen=frames)
        self.current_boxes = collections.deque(maxlen=frames)
