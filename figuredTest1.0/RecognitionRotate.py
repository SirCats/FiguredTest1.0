import cv2

import numpy as np
import imutils

from Point import *
import ColorList

class ColorContourRecognition:
    """该类完成以下基本操作：
        2. 亮度对比度调整
        4. 图像读取
        3. 形态学、二值化处理
        5. 获取目标颜色部分的二值图
        6. 寻找轮廓并粗略过滤
        7. 获取第一阶段处理结果（上述操作的总和）
        """

    def __init__(self, id, cap, path):
        self.id = id  # 标志实物图
        # _, self.image = cap.read()  # 从摄像头读取
        self.image = self.readImage(cap, path)
        self.cnts = []
        self.cnt_num = 0
        self.centerP = Point(0, 0)  # 形状的中心点

    def getImage(self):
        return self.image

    def adjustContrastBrightness(self, src, a, g):
        """亮度、对比度调整"""
        h, w, ch = src.shape  # 获取shape的数值，height和width、通道
        # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
        src2 = np.zeros([h, w, ch], src.dtype)
        dst = cv2.addWeighted(src, a, src2, 1 - a, g)
        return dst

    def postprocessMorphology(self, image):
        """形态学、二值化处理"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 进行闭运算，形成封闭图形
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # 腐蚀图像
        closed = cv2.erode(closed, None, iterations=2)
        # 膨胀图像
        thresh = cv2.dilate(closed, None, iterations=2)
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        return thresh

    def readImage(self, cap, path):
        """图像读取"""
        if path == None:
            ret, image = cap.read()
        else:
            image = cv2.imread(path)

        if self.id == 0:
            image = self.adjustContrastBrightness(image, 1.25, -10)
        cv2.imwrite("images/template/01.jpg", image)
        return image

    def getColorPart(self, colorIndict):
        """获取目标颜色部分的二值图"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        colordic = ColorList.getColorList(self.id)
        for d in colordic:
            if d == colorIndict:
                if len(colordic[d]) == 4:
                    mask1 = cv2.inRange(hsv, colordic[d][0], colordic[d][1])
                    mask2 = cv2.inRange(hsv, colordic[d][2], colordic[d][3])
                    thresh = cv2.addWeighted(mask1, 1, mask2, 1, 0)
                else:
                    thresh = cv2.inRange(hsv, colordic[d][0], colordic[d][1])
                break
        thresh = self.postprocessMorphology(thresh)

        # cv2.imshow(d, thresh)
        # cv2.waitKey(0)
        return self.image, thresh

    def getContours(self, color):
        """寻找轮廓并粗略过滤"""
        self.image, thresh = self.getColorPart(color)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # cnt 表示轮廓上的所有点
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if area > 1000:  # 对区域进行初步筛选，过滤掉一些很小的
                self.cnts.append(c)
                cv2.drawContours(self.image, [c], -1, (0, 255, 0), 1)
                # cv2.circle(self.image, (cX, cY), 1, (255, 0, 0), 1)
        self.cnt_num = len(self.cnts)
        # cv2.imshow("getContours01", self.image)
        # cv2.waitKey(0)
        return self.image

    def getAccurateContours(self, color):
        """获取第一阶段处理结果（上述操作的总和）"""
        self.getContours(color)
        return self.image


class ShapeRecognition(ColorContourRecognition):
    """该类继承ColorShapeRecognition,在颜色处理的基础上，进一步进行形状判别
        完成以下操作：
        1. 轮廓逼近
        2. 获取一个轮廓逼近得到的角点
        3. 为图形框出最小矩形
        4. 根据一个轮廓来判断形状
        5. 第二阶段，根据形状过滤掉一部分轮廓
        6. 第三阶段，对颜色、形状相同的轮廓根据尺寸来进行最后一步筛选
        7. 三个阶段完整的识别过程
        """

    def __init__(self, id, cap, path):
        self.shape = ''
        self.vertex = []
        self.hypotenusAngle = 0
        self.centerVector = (0, 0)
        ColorContourRecognition.__init__(self, id, cap, path)

    def getDeviationAngle(self):
        """ 返回斜边角度 """
        p = Point(0, 0)
        slope = p.getSlope(self.vertex[0], self.vertex[1], self.id)
        self.hypotenusAngle = p.angle(slope)

        return self.hypotenusAngle



    def getApprox(self, cnt):  # 轮廓逼近

        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)  # 得到逼近的顶点

        return approx

    def getPreVertex(self, cnt):
        """获取一个轮廓逼近得到的角点"""
        tempVertexs = []
        approx = self.getApprox(cnt)
        corners = len(approx)   # 分析顶点数

        if corners == 3 or corners == 4:
            for cornerIndex in range(0, corners):
                singleTempVertex = Point(approx[cornerIndex][0][0], approx[cornerIndex][0][1])
                tempVertexs.append(singleTempVertex)
                # cv2.circle(self.image,
                #            (tempVertexs[cornerIndex].x, tempVertexs[cornerIndex].y), 2,
                #            (255, 0, 0), 2)
        else:
            pass  # tempVertexs为空
        return tempVertexs, approx

    def getMinRect(self, approx):
        """为图形框出最小矩形"""
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        w = rect[1][0]
        h = rect[1][1]
        wh_ratio = w / float(h)  # 计算长宽比
        # cv2.drawContours(self.image, [box], 0, (0, 0, 255), 2)
        return box, rect, wh_ratio

    def shapeAlysis(self, cnt):
        """根据一个轮廓来判断形状"""
        tempVertexs, approx = self.getPreVertex(cnt)
        if tempVertexs == []:  # 轮廓不是三个顶点或四个顶点时
            self.shape = 'none'
        else:
            corners = len(tempVertexs)
            p = Point(0, 0)
            if corners == 4:
                box, rect, wh_ratio = self.getMinRect(approx)
                if 0.7 <= wh_ratio <= 2.0:
                    self.shape = 'square'
                    tempVertexs = p.numberVertexuadrangle(0, tempVertexs, box)
                else:  # 根据两组对边平行的四边形是平行四边形
                    tempVertexs = p.numberVertexuadrangle(1, tempVertexs, box)
                    if p.parallelogramJudge(tempVertexs):
                        self.shape = 'parallelogram'  # 不要忘记考虑False的情况
                    else:
                        self.shape = 'none'
            else:
                tempVertexs = p.numberVertexTriangle(tempVertexs)  # 对三角形进行编号
                if p.triangleJudge(tempVertexs):
                    self.shape = 'triangle'  # 不要忘记考虑False的情况
                else:
                    self.shape = 'none'
            # 测试编号是否正确
            # for i in range(0, len(tempVertexs)):
            #     cv2.putText(self.image, str(i), (tempVertexs[i].x, tempVertexs[i].y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return tempVertexs, self.shape


    def preliminarynalysis(self, shapeGoal):
        """ 第二阶段，根据形状过滤掉一部分轮廓"""
        tempVertexsfiltered = []
        tempCnt = []
        k = 0
        # for i in range(0,len(self.cnts)):
        # tempCnt = self.cnts
        while k < len(self.cnts):
            c = self.cnts[k]
            tempVertexs, shapeTemp = self.shapeAlysis(c)
            k += 1
            if shapeTemp == 'none':
                continue
            if shapeTemp == shapeGoal:
                tempCnt.append(c)
                tempVertexsfiltered.append(tempVertexs)
                cv2.drawContours(self.image, [c], -1, (255, 0, 0), 1)
                for i in range(0, len(tempVertexs)):
                    cv2.putText(self.image, str(i), (tempVertexs[i].x, tempVertexs[i].y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        self.cnts = []
        self.cnts = tempCnt
        self.cnt_num = len(self.cnts)
        # cv2.imshow("filteredImage", self.image)
        # cv2.waitKey(0)
        return tempVertexsfiltered, self.image

    def ultimateFilter(self, colorGoal, shapeGoal):
        """最后一步筛选，当出现两个轮廓，形状、颜色均相同时"""
        temVertex, _ = self.preliminarynalysis(shapeGoal)
        goalIndex = 0
        if self.cnt_num == 1:
            pass
        elif self.cnt_num == 0:
            print("没有检测到轮廓！")
            exit()
        else:  # 针对粉色和紫色、红色和橙色
            goalArea = cv2.contourArea(self.cnts[0])
            # 针对粉色和紫色，目标一定是多个轮廓中面积最大的一个
            if colorGoal == 'pink' or colorGoal == 'purple' or colorGoal == 'orange':
                for cntIndex in range(0, self.cnt_num):
                    area = cv2.contourArea(self.cnts[cntIndex])
                    if area >= goalArea:
                        goalArea = area
                        goalIndex = cntIndex

            elif colorGoal == 'red':
                for cntIndex in range(0, self.cnt_num):
                    area = cv2.contourArea(self.cnts[cntIndex])
                    if area <= goalArea:
                        goalArea = area
                        goalIndex = cntIndex

            tempCnt = self.cnts[goalIndex]
            self.cnts = []
            self.cnts = tempCnt
            self.cnt_num = len(self.cnts)


        if self.cnt_num == 1:
            self.vertex = temVertex[goalIndex]  # 获取最终排序好的顶点
            y, x = self.image.shape[:2]
            centerImage = (x / 2, y / 2)  # 图像中心点
            M = cv2.moments(self.cnts[0])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.centerP.x = cX
            self.centerP.y = cY
            self.centerVector = (centerImage[0] - cX, centerImage[1] - cY)  # 位移矢量
            self.getDeviationAngle()  # 计算斜边角度
        else:
            print("最终轮廓数：", self.cnt_num)
            exit()

        cv2.imshow("FinallyImage", self.image)
        cv2.waitKey(0)

        return self.centerVector


    def completeRecognition(self, colorGoal, shapeGoal):
        """ 完整的识别过程"""
        self.getAccurateContours(colorGoal)  # 第一次简单过滤掉很小的区域
        self.preliminarynalysis(shapeGoal)   # 第二次根据形状过滤
        self.ultimateFilter(colorGoal, shapeGoal)       # 第三次根据面积过滤


class Rotate:

    def __init__(self):

        self.rotateAngle = 0

    def getRotateAngle(self, capImage, mouldImage, shapeGoal):
        """ 获取旋转角度 """
        deviationAngle = capImage.hypotenusAngle - mouldImage.hypotenusAngle
        print("capImage hypotenusAngle:", capImage.hypotenusAngle)
        print("mould hypotenusAngle:", mouldImage.hypotenusAngle)
        print("deviationAngle:", deviationAngle)
        self.rotateAngle = deviationAngle
        overturnFlag = self.whetherOverturn(capImage, mouldImage, shapeGoal)  # 是否需要翻转，True翻转
        if overturnFlag:

            # self.rotateAngle = deviationAngle + 180
            if deviationAngle > 0:
                """1. a-b>0 同时需要翻转"""
                deviationAngle = 180 - deviationAngle
            else:
                """2. a-b<0 同时需要翻转"""
                deviationAngle = 180 + deviationAngle
                overturnFlag = False

        """
        3. a-b<0 不需要翻转 逆时针
        4. a-b>0 不需要翻转 顺时针
        """
        self.rotateAngle = (deviationAngle, overturnFlag)

        return self.rotateAngle

    def whetherOverturn(self, capImage, mouldImage, shapeGoal):
        """ 判断是否需要翻转"""
        overturnFlag = False

        if shapeGoal == 'triangle':
            self.rotate(3, capImage)

            if 0 <= mouldImage.hypotenusAngle <= 45 or 135 <= mouldImage.hypotenusAngle <= 180:
                if (capImage.vertex[2].y < capImage.centerP.y and mouldImage.vertex[2].y < mouldImage.centerP.y) or (
                        capImage.vertex[2].y > capImage.centerP.y and mouldImage.vertex[2].y > mouldImage.centerP.y):
                    pass
                else:
                    overturnFlag = True
            elif 45 < mouldImage.hypotenusAngle <= 90 or 90 < mouldImage.hypotenusAngle < 135:
                if (capImage.vertex[2].x < capImage.centerP.x and mouldImage.vertex[2].x < mouldImage.centerP.x) or (
                        capImage.vertex[2].x > capImage.centerP.x and mouldImage.vertex[2].x > mouldImage.centerP.x):
                    pass
                else:
                    overturnFlag = True

        return overturnFlag



    def rotate(self, numVertex, image):

        rotateAngle = self.rotateAngle
        if rotateAngle < 0:
            rotateAngle = abs(rotateAngle)
            rotateAngle = math.radians(rotateAngle)
            rotateAngle = -1 * rotateAngle
        elif rotateAngle > 0:
            rotateAngle = math.radians(rotateAngle)

        for vertexIndex in range(0, numVertex):
            image.vertex[vertexIndex] = self.rotatePoint(image.vertex[vertexIndex], image.centerP, rotateAngle)


        return image.vertex


    def rotatePoint(self, p, pCenter, theta):
        """ 某一点逆时针旋转theta（弧度）"""
        pRotated = Point(0, 0)
        pRotated.x = pCenter.x + (p.x - pCenter.x) * math.cos(theta) - (p.y - pCenter.y) * math.sin(theta)
        pRotated.y = pCenter.y + (p.x - pCenter.x) * math.sin(theta) + (p.y - pCenter.y) * math.cos(theta)
        return pRotated



def testCamera():
    """ 摄像头测试 """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    print('启动摄像头...')
    ret, image = cap.read()

def recognitionRotate():  # 完整测试
    mouldPath = 'images/mould/1.jpg'
    # capPath = "images/figured/01.jpg"
    for image_index in range(10, 37):
        capPath = "images/figured/"
        # cappath = "images/simpleImage/blue/"
        # mouldpath = "images/mould/"
        capPath += str(image_index) + ".jpg"
        print(capPath)
        # cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        # print('启动摄像头...')
        for i in range(0, 7):
            if i == 0:
                colorGoal = 'pink'
                shapeGoal = 'triangle'
            elif i == 1:
                colorGoal = 'red'
                shapeGoal = 'triangle'
            elif i == 2:
                colorGoal = 'orange'
                shapeGoal = 'triangle'
            elif i == 3:
                colorGoal = 'yellow'
                shapeGoal = 'parallelogram'
            elif i == 4:
                colorGoal = 'green'
                shapeGoal = 'triangle'
            elif i == 5:
                colorGoal = 'blue'
                shapeGoal = 'square'
            else:
                colorGoal = 'purple'
                shapeGoal = 'triangle'
            mould = ShapeRecognition(1, cap='', path=mouldPath)
            mould.completeRecognition(colorGoal, shapeGoal)
            cap = ShapeRecognition(0, cap='', path=capPath)
            cap.completeRecognition(colorGoal, shapeGoal)
            rotate = Rotate()
            angle = rotate.getRotateAngle(cap, mould, shapeGoal)
            print(colorGoal+" angle", angle)
            cv2.imwrite("images/results/" + str(image_index) + str(colorGoal)+'.jpg', cap.getImage())

def classTest():  # 识别部分，未测试旋转
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    print('启动摄像头...')
    # path = "images/figured/"
    # path = "images/simpleImage/blue/"
    # image_index = input("Please input the index of image:")

    for image_index in range(1, 2):
        path = "images/figured/0"
        # path = "images/simpleImage/blue/"
        # path = "images/mould/"
        path += str(image_index) + ".jpg"
        print(path)
        shapeGoal = 'triangle'
        for i in range(0, 7):

            if i == 0:
                colorGoal = 'pink'
            elif i == 1:
                colorGoal = 'red'
            elif i == 2:
                colorGoal = 'orange'
            elif i == 3:
                colorGoal = 'yellow'
                shapeGoal = 'parallelogram'
            elif i == 4:
                colorGoal = 'green'
            elif i == 5:
                colorGoal = 'blue'
                shapeGoal = 'square'
            else:
                colorGoal = 'purple'
            t1 = ShapeRecognition(1, cap, path)

            t1.completeRecognition('yellow', 'parallelogram')
            #cv2.imwrite("images/results/"+str(image_index)+'yellow.jpg', t1.getImage())

if __name__ == '__main__':

    # classTest()
    recognitionRotate()