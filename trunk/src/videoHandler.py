import cv

class VideoHandler:
    FIRST_FRAME = 1
    LAST_FRAME = -1

    video = None
    frames = []

    def __init__(self, video_path=None):
        if video_path is not None:
            self.video = cv.CaptureFromFile(video_path)
            while(1):
                frame = cv.QueryFrame(self.video)
                if frame:
                    self.frames.append(frame)
                else:
                    break

    def getWidth(self):
        return cv.GetCaptureProperty(self.video, cv.CV_CAP_PROP_FRAME_WIDTH)

    def getHeight(self):
        return cv.GetCaptureProperty(self.video, cv.CV_CAP_PROP_FRAME_HEIGHT)

    def getFrameCount(self):
        # Buggy:
        # return cv.GetCaptureProperty(self.video, cv.CV_CAP_PROP_FRAME_COUNT)
        return len(self.frames)

    def getAllFrames(self):
        return self.frames

    def getFrameRange(self, frame_range):
        return self.frames[frame_range[0]:frame_range[1]]
