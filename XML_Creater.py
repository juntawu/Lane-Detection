import cv2
import numpy as np

class XML(object):
    """
    An object to create the xml file
    """

    def __init__(self, file):
        self.file = cv2.FileStorage(file, cv2.FILE_STORAGE_WRITE, encoding="zh_CN.UTF-8")
        # self.frameIndex = 0
        # self.targetNumber = 2
        # self.targetIndex = 0
        # self.color = 'WHITE'
        # self.type = 'DASHED'
        # self.point = None

    def create_frameNumber(self, frameNumber):
        self.file.write("FrameNumber", frameNumber)

    def create_targetNumber(self, frameIndex, targetNumber):
        self.file.write("Frame%05dTargetNumber" % frameIndex, targetNumber)

    def create_target(self, frameIndex, targetIndex, color, type, position):
        self.file.write("Frame%05dTarget%05d" % (frameIndex, targetIndex), "{" )
        # 输出目标车道线类型
        typeStr = ""
        if (True == color):
            typeStr = typeStr + "白色"
        else:
            typeStr = typeStr + "黄色"
        if (True == type):
            typeStr = typeStr + "虚线"
        else:
            typeStr = typeStr + "实线"
        self.file.write("Type", typeStr)
        # # 输出目标车道线位置
        position = np.array(position).reshape(len(position),1,2)
        self.file.write("Position", position)
        self.file.write("Frame%05dTarget%05d" % (frameIndex, targetIndex), "}")

    def close(self):
        self.file.release()
