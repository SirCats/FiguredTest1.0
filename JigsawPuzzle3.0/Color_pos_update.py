import cv2
import numpy as np
import imutils
import colorsys
from PIL import Image
import collections

import sys

# 根据颜色得到下标，用于颜色数量计算
color_dict={'pink': 0,
            'red': 1,
            'orange': 2,
            'yellow': 3,
            'green': 4,
            'blue': 5,
            'purple': 6,
            'black': 7,
            'gray': 8,
            'white': 9,
            'None': 10
            }


# 颜色字典
"""
def getColorList(flag):#flag=1表示模板颜色匹配
    dict = collections.defaultdict(list)

    #黑色

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    # #灰色

    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray']=color_list

     #白色
    if flag==0:
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 30, 255])
        color_list = []
        color_list.append(lower_white)
        color_list.append(upper_white)
        dict['white'] = color_list


    if flag==1:
        lower_pink = np.array([160, 86, 178])
        upper_pink = np.array([180, 165, 255])
        color_list = []
    else:
        #粉色
        # 粉色
        # H:340-360 S:27%-70% V:50%-100%
        lower_pink = np.array([175, 70, 127])
        upper_pink = np.array([180, 178, 255])
        #
        # lower_pink1=np.array([0,76,178])
        # upper_pink1=np.array([7,180,255])

        # 粉色：H:0-14,S:27%-70%,V:50%-100%
        lower_pink1 = np.array([0, 70, 127])
        upper_pink1 = np.array([5, 178, 255])
        color_list = []
        color_list.append(lower_pink1)
        color_list.append(upper_pink1)
    color_list.append(lower_pink)
    color_list.append(upper_pink)
    dict['pink'] = color_list

    # 红色
    # H：166-180 S:60%-100% V:60%-100%
    # lower_red = np.array([166, 1153, 153])
    # upper_red = np.array([180, 255, 255])
    # color_list = []
    # color_list.append(lower_red)
    # color_list.append(upper_red)
    # dict['red'] = color_list

    # 红色2
    # H:0-10 S: 55%-100% V:30%-100%
    lower_red = np.array([0, 140, 80])
    upper_red = np.array([5, 255, 255])

    # lower_red2 = np.array([166, 153, 153])
    # upper_red2 = np.array([180, 255, 255])
    # lower_red2 = np.array([166, 153, 128])
    # upper_red2 = np.array([180, 255, 255])

    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    # color_list.append(lower_red2)
    # color_list.append(upper_red2)
    dict['red'] = color_list

    # 橙色
    if flag:
        lower_orange = np.array([6, 43, 46])
        upper_orange = np.array([24, 255, 255])
    else:
        # H:14-30,S:40%-100%,V:35%-100%
        lower_orange = np.array([6, 102, 90])
        upper_orange = np.array([15, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    # if flag:
    #     lower_yellow = np.array([25, 43, 46])
    #     upper_yellow = np.array([34, 255, 255])
    # else:
    #     lower_yellow = np.array([25, 43, 180])
    #     upper_yellow = np.array([33, 255, 255])
    lower_yellow = np.array([25, 43, 46])
    upper_yellow = np.array([34, 255, 255])

    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色

    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色

    # lower_cyan = np.array([78, 43, 46])
    # upper_cyan = np.array([99, 255, 255])
    # color_list = []
    # color_list.append(lower_cyan)
    # color_list.append(upper_cyan)
    # dict['cyan'] = color_list

    # 蓝色

    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    if flag:
        lower_purple = np.array([125, 43, 46])
        upper_purple = np.array([159, 255, 255])
    else:
        # H:270-360 S:20%-50%,V:25%-85%
        lower_purple = np.array([135, 50, 64])
        upper_purple = np.array([180, 128, 216])

    # lower_purple1=np.array([170,])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict
    """

def getColorList(flag):#flag=1表示模板颜色匹配
    dict = collections.defaultdict(list)

    #黑色

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    # #灰色

    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray']=color_list

     #白色
    if flag==0:
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 30, 255])
        color_list = []
        color_list.append(lower_white)
        color_list.append(upper_white)
        dict['white'] = color_list


    if flag==1:
        lower_pink = np.array([160, 86, 178])
        upper_pink = np.array([180, 165, 255])
        color_list = []
    else:
        #粉色
        # 粉色
        # H:340-360 S:30%-54% V:70%-100%
        lower_pink = np.array([170, 76, 204])
        upper_pink = np.array([180, 138, 255])
        #
        # lower_pink1=np.array([0,76,178])
        # upper_pink1=np.array([7,180,255])

        # 粉色：H:0-14,S:27%-70%,V:50%-100%
        # lower_pink1 = np.array([0, 70, 127])
        # upper_pink1 = np.array([5, 178, 255])
        color_list = []
        # color_list.append(lower_pink1)
        # color_list.append(upper_pink1)
    color_list.append(lower_pink)
    color_list.append(upper_pink)
    dict['pink'] = color_list

    # 红色
    # H：166-180 S:60%-100% V:60%-100%
    # lower_red = np.array([166, 1153, 153])
    # upper_red = np.array([180, 255, 255])
    # color_list = []
    # color_list.append(lower_red)
    # color_list.append(upper_red)
    # dict['red'] = color_list

    # 红色2
    # H:0-10 S: 55%-100% V:30%-100%
    lower_red = np.array([0, 140, 80])
    upper_red = np.array([3, 255, 204])
    # H:345-360 S:50%-100% V: 45%-100%
    lower_red2 = np.array([170, 140, 115])
    upper_red2 = np.array([180, 255, 255])

    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    color_list.append(lower_red2)
    color_list.append(upper_red2)
    dict['red'] = color_list

    # 橙色
    if flag:
        lower_orange = np.array([6, 43, 46])
        upper_orange = np.array([24, 255, 255])
    else:
        # H:0-6,S:45%-85%,V:60%-100%
        lower_orange = np.array([0, 115, 153])
        upper_orange = np.array([10, 217, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    # if flag:
    #     lower_yellow = np.array([25, 43, 46])
    #     upper_yellow = np.array([34, 255, 255])
    # else:
    #     lower_yellow = np.array([25, 43, 180])
    #     upper_yellow = np.array([33, 255, 255])
    lower_yellow = np.array([25, 43, 46])
    upper_yellow = np.array([34, 255, 255])

    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色

    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色

    # lower_cyan = np.array([78, 43, 46])
    # upper_cyan = np.array([99, 255, 255])
    # color_list = []
    # color_list.append(lower_cyan)
    # color_list.append(upper_cyan)
    # dict['cyan'] = color_list

    # 蓝色

    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    if flag:
        lower_purple = np.array([125, 43, 46])
        upper_purple = np.array([159, 255, 255])
    else:
        # H:270-360 S:20%-50%,V:25%-85%
        # lower_purple = np.array([135, 50, 64])
        # upper_purple = np.array([175, 128, 216])

        lower_purple = np.array([135, 76, 64])
        upper_purple = np.array([180, 138, 204])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict


# 亮度、对比度调整
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    # cv2.imshow("con-bri-demo", dst)
    # cv2.waitKey(0)
    return dst

#==============================================
#颜色判断函数
#==============================================
#某一像素点的rgb转hsv
def rgb_hsv(x,y,image):
    color = image.getpixel((x, y))
    new_color = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    hsv = [int(new_color[0] * 360 / 2), int(new_color[1] * 255), int(new_color[2] * 255)]
    return  hsv

#返回图像某一像素点颜色
def Color(x,y,image):
    new_colors=rgb_hsv(x,y,image)
    # print(new_colors[0])
    dict=getColorList(0)
    for i in dict:
        if dict[i][0][0] <= new_colors[0] <= dict[i][1][0] and dict[i][0][1] <= \
                new_colors[1] <= dict[i][1][1] and dict[i][0][2] <= new_colors[2] <= \
                dict[i][1][2]:
            # print("hkig"+str(i))
            return i
        if len(dict[i])==4:
            if dict[i][2][0] <= new_colors[0] <= dict[i][3][0] and dict[i][2][1] <= \
                    new_colors[1] <= dict[i][3][1] and dict[i][2][2] <= new_colors[2] <= \
                    dict[i][3][2]:
            # if dict[i][0][0] <= new_colors[0] <= dict[i][1][0] and dict[i][0][1] <= \
            #         new_colors[1] <= dict[i][1][1] and dict[i][0][2] <= new_colors[2] <= \
            #         dict[i][1][2]:
                # print("hkig"+str(i))
                return i

#================================================
#形状、颜色分析
#================================================
class Analysis:
    def __init__(self):
        self.shapes = {'triangle': 0, 'square': 0, 'polygons': 0, 'circles': 0,'parallelogram':0}
    def draw_text_info(self, image):
        c1 = self.shapes['triangle']
        c2 = self.shapes['square']
        c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        c5=self.shapes['parallelogram']
        cv2.putText(image, "triangle: "+str(c1), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "square: " + str(c2), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "parallelogram: " + str(c5), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        return  image
    def analy(self, color_tem,shape,image):
        color = []
        for i in range(0, 11):
            color.append([])
        for i in range(0, 11):
            for j in range(0, 11):
                color[i].append(0)
        # print(color[0])
        y, x = image.shape[:2]
        center_p = (x / 2, y / 2)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dict = getColorList(0)
        num = 0
        for d in dict:
            if d != 'gray' and d != 'white' and d != 'black':
                num += 1
                if num == 1:
                    thresh = cv2.inRange(hsv, dict[d][0], dict[d][1])
                if len(dict[d]) == 4:
                    mask1 = cv2.inRange(hsv, dict[d][0], dict[d][1])
                    mask2 = cv2.inRange(hsv, dict[d][2], dict[d][3])
                    mask = cv2.addWeighted(mask1, 1, mask2, 1, 0)
                else:
                    mask = cv2.inRange(hsv, dict[d][0], dict[d][1])
                # cv2.imshow(d,mask)
                # cv2.waitKey(0)
                thresh = cv2.addWeighted(thresh, 1, mask, 1, 0)
                # cv2.imshow("thresh",thresh)
                # cv2.waitKey(0
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 进行闭运算，形成封闭图形
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 腐蚀图像
        closed = cv2.erode(closed, None, iterations=2)
        # 膨胀图像
        thresh = cv2.dilate(closed, None, iterations=2)

        if True:
            cv2.imshow("threshfinal",thresh)
            cv2.waitKey(0)
        # findContours：基于二值图像寻找物体的轮廓
        #       image:      二值图像
        #       mode:       定义轮廓检索模式
        #       method:     定义轮廓近似方法
        #       return:     contours：hierarchy
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # cnt 表示轮廓上的所有点
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        k = 0
        cX = 0
        cY = 0
        for c in cnts:
            # 轮廓逼近
            epsilon = 0.05 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            # 分析几何形状
            corners = len(approx)
            # print("KKK" + str(corners))
            shape_type = ""
            if corners == 3:
                count = self.shapes['triangle']
                count = count + 1
                self.shapes['triangle'] = count
                shape_type = "triangle"
            if corners == 4:
                # 用红色表示有旋转角度的矩形框架
                rect = cv2.minAreaRect(approx)
                w=rect[1][0]
                h=rect[1][1]
                ar=w/float(h)
                # print(ar)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
                if ar>=0.70 and ar<=2.00:
                    count = self.shapes['square']
                    count = count + 1
                    self.shapes['square'] = count
                    shape_type = "square"
                else:
                    count=self.shapes['parallelogram']
                    count=count+1
                    self.shapes['parallelogram']=count
                    shape_type="parallelogram"
            if corners >= 10:
                count = self.shapes['circles']
                count = count + 1
                self.shapes['circles'] = count
                shape_type = "circles"
            if 4 < corners < 10:
                count = self.shapes['polygons']
                count = count + 1
                self.shapes['polygons'] = count
                shape_type = "polygons"

            im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if shape_type==shape:
                # print(k)
                M = cv2.moments(c)
                cX_tem = int(M["m10"] / M["m00"])
                cY_tem = int(M["m01"] / M["m00"])
                # cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
                # cv2.imshow(shape_type,image)
                # cv2.waitKey(0)
                color_end = ''
                #==========================================================
                #根据中心点区域判断颜色
                #==========================================================
                try:
                    for i in range(max(cX_tem - 20,0), min(cX_tem + 20,x)):
                        for j in range(max(cY_tem - 20,0), min(cY_tem+ 20,y)):
                            color_str = Color(i, j, im)
                            if color_str == 'None':
                                poj = 10
                            else:
                                poj = color_dict[str(color_str)]
                            color[k][poj] += 1 #测试会不会出错
                    max_index = 0
                    for i in range(0, 7):
                        if color[k][i] > color[k][max_index]:
                            max_index = i;

                    for key in color_dict.keys():
                        if color_dict[key] == max_index:
                            # print("color:" + key)
                            color_end = key
                except BaseException:
                    print("Error in identify color!")
                    sys.exit()
                else:
                    if color_end == color_tem:
                        cX = cX_tem
                        cY = cY_tem
                        cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
                        cv2.putText(image, str(color_end) + " " + shape_type, (cX - 20, cY - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.circle(image, (cX, cY), 2, (255, 0, 0), 1)

                        if True:
                            cv2.imshow(color_end,image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        break
                k+=1
                # print("k = %d" % k)
        self.draw_text_info(blurred)
        # print("draw...")
        # imwrite_str = "image/test_result/" + str(0) + ".jpg"
        # # cv2.imwrite("image/pattern.jpg",blurred)
        # print("write image...")
        # cv2.imwrite(imwrite_str, image)
        # print(center_p)
        # print("this is cX", (cX, cY))
        return (center_p[0]-cX,center_p[1]-cY)
def main():
    image = cv2.imread("image/pictures/6.jpg")
    # cv2.imshow("initial image", image)
    # cv2.waitKey(0)
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    # print('启动摄像头...')
    # ret, image = cap.read()
    # cv2.destroyAllWindows()
    # image = cv2.resize(image,(960,640))
    # print(i)
    size = image.shape
    # 获取图像的高、宽
    h, w = image.shape[:2]
    # # 调整图像亮度、对比度
    image = contrast_brightness_image(image, 1.3, -10)

    # # # 高斯模糊去噪
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # # cv2.imshow("blured",blured)
    # # # 进行泛洪填充，处理背景
    # mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    # cv2.floodFill(image, mask, (w - 1, h - 1), (0, 0, 0), (50, 35, 50), (185, 190, 190), cv2.FLOODFILL_FIXED_RANGE)
    instance=Analysis()
    org = instance.analy('blue', 'square', image)
    org=instance.analy('yellow','parallelogram',image)
    org=instance.analy('red','triangle',image)
    org = instance.analy('purple', 'triangle', image)
    org = instance.analy('pink', 'triangle', image)
    org = instance.analy('orange', 'triangle', image)
    org = instance.analy('green', 'triangle', image)
    print(org)


if __name__ == '__main__':
    main()