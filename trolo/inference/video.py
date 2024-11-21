import cv2
from typing import Iterator, Dict, Any

class VideoStream:
    def __init__(self, source: str, batch_size: int = 1):
        self.source = source
        self.batch_size = batch_size
        self.cap = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.source)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        frames = []
        frame_ids = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                if frames:
                    yield {'frames': frames, 'frame_ids': frame_ids}
                break
                
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_ids.append(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            
            if len(frames) == self.batch_size:
                yield {'frames': frames, 'frame_ids': frame_ids}
                frames = []
                frame_ids = [] 