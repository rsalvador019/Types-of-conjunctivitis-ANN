import dlib
import cv2
import numpy as np
import array as ar
from matplotlib import cm
from PIL import Image

def CompressImage_FixSize(FileName):

    Height = 90
    Width = 135

    fd_img = open(FileName, 'r')

    img = Image.open(fd_img)
    img = img.resize((Width, Height), Image.ANTIALIAS)

    img_arr = np.asarray(img)
    img_arr = img_arr.copy()

    img.save('Outputs/'+FileName)

    fd_img.close()

def CompressImage_Template(FileName, Template):

    if Template == 1:
        Height = 40
        Width = 75
        NewFileName = 'CompressImage_Template1(90x60).jpg'

    elif Template == 2:
        Height = 60
        Width = 140
        NewFileName = 'CompressImage_Template2(140x60).jpg'

    else:
        print("Invalid Template Input")

    fd_img = open(FileName, 'r')

    img = Image.open(fd_img)
    img = img.resize((Width, Height), Image.ANTIALIAS)

    img_arr = np.asarray(img)
    img_arr = img_arr.copy()

    img.save('Outputs/'+NewFileName)

    return 'Outputs/'+NewFileName

def VerticalFlip(FileName):
    image = Image.open(FileName)
    image_array = np.asarray(image)
    image_array = image_array.copy()

    temp_image_array = image_array*0

    height, width, RGB = image_array.shape

    x_temp = 0
    y_temp = width - 1

    for x in range(0, height):
        for y in range(0, width):
            temp_image_array[x][y] = image_array[x_temp][y_temp]
            y_temp -= 1
        x_temp += 1
        y_temp = width-1


    NewFileName = 'VerticalFlip.jpg'
    im = Image.fromarray(temp_image_array)
    im.save('Outputs/'+NewFileName)

    return temp_image_array

def MergetoTemplate(FileName, Template, Position):

    if Template == 1:
        face_image = Image.open('Others/Face_Template 01.jpg')
        face = np.asarray(face_image)
        face = face.copy()
        face_height, face_width, face_RGB = face.shape

        face_template_heigth = 235
        face_template_1_width1 = 150
        face_template_1_width2 = 275

        FileName = CompressImage_Template(FileName, 1)

    elif Template == 2:

        face_image = Image.open('Others/Face_Template 02.jpg')
        face = np.asarray(face_image)
        face = face.copy()
        face_height, face_width, face_RGB = face.shape

        face_template_heigth = 553
        face_template_1_width1 = 353
        face_template_1_width2 = 573

        FileName = CompressImage_Template(FileName, 2)

    else:
        print("Invalid Tempate Input")

    Flip_Eye = VerticalFlip(FileName)
    Flip = np.asarray(Flip_Eye)

    eye_image = Image.open(FileName)
    eye = np.asarray(eye_image)
    eye_height, eye_width, eye_RGB = eye.shape

    if Position == 1:
        holder1 = Flip.copy()
        holder2 = eye.copy()

    elif  Position == 2:
        holder1 = eye.copy()
        holder2 = Flip.copy()

    else:
        print("Invalid Position Input")

    x_eye = 0
    y_eye = 0

    for x in range(face_template_heigth, face_template_heigth+eye_height):
        for y in range(face_template_1_width1 , face_template_1_width1+eye_width):
            face[x][y] = holder1[x_eye][y_eye]
            y_eye += 1
        x_eye += 1
        y_eye = 0

    x_eye = 0
    y_eye = 0

    for x in range(face_template_heigth, face_template_heigth+eye_height):
        for y in range(face_template_1_width2 , face_template_1_width2+eye_width):
            face[x][y] = holder2[x_eye][y_eye]
            y_eye += 1
        x_eye += 1
        y_eye = 0

    FileName = 'MergetoTemplate_Template'+str(Template)+'.jpg'
    im = Image.fromarray(face)
    im.save('Outputs/'+FileName)
    im.save(FileName, "JPEG")

    return FileName

def EyelidAndEyeCornerSegmetation(FileName, Position):

    Fix_image = Image.open(FileName)
    Fix = np.asarray(Fix_image)
    Fix = Fix.copy()
    Fix_Height, Fix_Width, Fix_RGB = Fix.shape

    MtoT1 = MergetoTemplate(FileName, 1, Position)
    MtoT2 = MergetoTemplate(FileName, 2, Position)

    predictor_path = 'Others/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    detector = dlib.get_frontal_face_detector()

    img = cv2.imread(MtoT1)

    face = np.asarray(img)
    face = face.copy()

    dets = detector(img)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    imp1 = np.empty([68, 2], dtype = int)

    for b in range(68):
        imp1[b][0] = shape.part(b).x
        imp1[b][1] = shape.part(b).y

    for a in range(36, 42):
        if a == 41:
            face = cv2.line(face, (imp1[a][0], imp1[a][1]), (imp1[36][0], imp1[36][1]), (0, 255, 0), 1)

        else:
            face = cv2.line(face, (imp1[a][0], imp1[a][1]), (imp1[a+1][0], imp1[a+1][1]), (0, 255, 0), 1)

    for a in range(42, 48):
        if a == 47:
            face = cv2.line(face, (imp1[a][0], imp1[a][1]), (imp1[42][0], imp1[42][1]), (0, 255, 0), 1)

        else:
            face = cv2.line(face, (imp1[a][0], imp1[a][1]), (imp1[a+1][0], imp1[a+1][1]), (0, 255, 0), 1)

    NewFileName = 'T_IMP1.jpg'
    im = Image.fromarray(face)
    im.save('Outputs/'+NewFileName)

    img = cv2.imread(MtoT2)

    face = np.asarray(img)
    face = face.copy()

    dets = detector(img)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    imp2 = np.empty([68, 2], dtype = int)

    for b in range(68):
        imp2[b][0] = shape.part(b).x
        imp2[b][1] = shape.part(b).y

    for a in range(36, 42):
        if a == 41:
            face = cv2.line(face, (imp2[a][0], imp2[a][1]), (imp2[36][0], imp2[36][1]), (0, 255, 0), 1)

        else:
            face = cv2.line(face, (imp2[a][0], imp2[a][1]), (imp2[a+1][0], imp2[a+1][1]), (0, 255, 0), 1)

    for a in range(42, 48):
        if a == 47:
            face = cv2.line(face, (imp2[a][0], imp2[a][1]), (imp2[42][0], imp2[42][1]), (0, 255, 0), 1)

        else:
            face = cv2.line(face, (imp2[a][0], imp2[a][1]), (imp2[a+1][0], imp2[a+1][1]), (0, 255, 0), 1)

    NewFileName = 'T_IMP2.jpg'
    im = Image.fromarray(face)
    im.save('Outputs/'+NewFileName)

    temp_imp1 = []
    temp_imp2 = []

    T1_height = 235
    T1_width = 150

    T2_height = 553
    T2_width = 353

    for a in range(36, 48):

        if a > 41:
            T1_width = 275
            T2_width = 573

        x = ((imp1[a][0] - T1_width) * Fix_Width) / 75
        y = ((imp1[a][1] - T1_height) * Fix_Height) / 40

        temp_imp1.append([x, y])


        x = ((imp2[a][0] - T2_width) * Fix_Width) / 140
        y = ((imp2[a][1] - T2_height) * Fix_Height) / 60

        temp_imp2.append([x, y])

        # Sort

    Fix_imp1 = [temp_imp1[0], temp_imp1[3], temp_imp1[1], temp_imp1[2], temp_imp1[5], temp_imp1[4],
                temp_imp1[9], temp_imp1[6], temp_imp1[8], temp_imp1[7], temp_imp1[10], temp_imp1[11]]

    Fix_imp2 = [temp_imp2[0], temp_imp2[3], temp_imp2[1], temp_imp2[2], temp_imp2[5], temp_imp2[4],
                temp_imp2[9], temp_imp2[6], temp_imp2[8], temp_imp2[7], temp_imp2[10], temp_imp2[11]]

    Eval_imp1_x = ar.array('d', [])
    Eval_imp1_y = ar.array('d', [])

    Eval_imp2_x = ar.array('d', [])
    Eval_imp2_y = ar.array('d', [])

    for c in range(6):

        Eval_imp1_x.append(abs(abs(75 - Fix_imp1[c+6][0]) - Fix_imp1[c][0]))
        Eval_imp1_y.append(abs(Fix_imp1[c+6][1] - Fix_imp1[c][1]))

        Eval_imp2_x.append(abs(abs(140 - Fix_imp2[c+6][0]) - Fix_imp2[c][0]))
        Eval_imp2_y.append(abs(Fix_imp2[c+6][1] - Fix_imp2[c][1]))

    Total_imp1 = sum(Eval_imp1_x) + sum(Eval_imp1_y)
    Total_imp2 = sum(Eval_imp2_x) + sum(Eval_imp2_y)

    Final_imp = []

    if Position == 1:

        if Total_imp1 >= Total_imp2:
            for a in range(6, 11, 2):
                Final_imp.append(Fix_imp1[a+1])
                Final_imp.append(Fix_imp1[a])

        elif Total_imp1 < Total_imp2:
            for a in range(6, 11, 2):
                Final_imp.append(Fix_imp2[a+1])
                Final_imp.append(Fix_imp2[a])

    elif Position == 2:

        if Total_imp1 >= Total_imp2:
            for a in range(6):
                Final_imp.append(Fix_imp1[a])

        elif Total_imp1 < Total_imp2:
            for a in range(6):
                Final_imp.append(Fix_imp2[a])

    for a in range(6):
        cv2.circle(Fix, (Final_imp[a][0], Final_imp[a][1]), 1, (0, 0, 255), -1)

    NewFileName = 'EyelidAndEyeCornerSegmetation.jpg'
    im = Image.fromarray(Fix)
    im.save('Outputs/'+NewFileName)

    return Final_imp

def IrisSegmetation(FileName, Position):

    EyelidAndEyeCorner = EyelidAndEyeCornerSegmetation(FileName, Position)

    p1 = (EyelidAndEyeCorner[4][0] + EyelidAndEyeCorner[2][0])/2 , (EyelidAndEyeCorner[4][1] + EyelidAndEyeCorner[2][1])/2
    p2 = (EyelidAndEyeCorner[5][0] + EyelidAndEyeCorner[3][0])/2 , (EyelidAndEyeCorner[5][1] + EyelidAndEyeCorner[3][1])/2
    p3 = (EyelidAndEyeCorner[2][0] + EyelidAndEyeCorner[3][0])/2 , (EyelidAndEyeCorner[2][1] + EyelidAndEyeCorner[3][1])/2
    p4 = (EyelidAndEyeCorner[4][0] + EyelidAndEyeCorner[5][0])/2 , (EyelidAndEyeCorner[4][1] + EyelidAndEyeCorner[5][1])/2

    center = (p2[0] + p1[0])/2 , (p2[1] + p1[1])/2

    r1 = center[1] - p3[1]
    r2 = p4[1] - center[1]

    radius =  ((r1+r2)/2)
    Estimate_radius = (r1+r2)

    img_save = Image.open(FileName)
    img_save = np.asarray(img_save)
    img_save = img_save.copy()

    img = cv2.imread(FileName,0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius = Estimate_radius)   #maxRadius=-1

    if circles is None:
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius= -1)   #maxRadius=-1

    elif circles[0][0][0] == 0.0 or circles[0][0][1] == 0.0 or circles[0][0][2] == 0.0:
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius= -1)   #maxRadius=-1


    x , y, c = circles.shape

    eval_center = []

    flag = False

    for i in circles[0,:]:
        if y > 1:
            eval_center.append(int(img_save[int(i[1])] [int(i[0]) ] [0]) + int(img_save[int(i[1])] [int(i[0]) ] [1]) + int(img_save[int(i[1])] [int(i[0]) ] [2]))     #heightx width
            save_min = min(eval_center)
            Flag = True

        else:
            center = int(i[0]), int(i[1]), int(i[2])
            Flag = False

    if flag:
        for i in circles[0,:]:
            if (int(img_save[int(i[1])] [int(i[0]) ] [0]) + int(img_save[int(i[1])] [int(i[0]) ] [1]) + int(img_save[int(i[1])] [int(i[0]) ] [2]))  == save_min:
                center = int(i[0]), int(i[1]), int(i[2])


    img_save = cv2.circle(img_save,(int(center[0]),int(center[1]) ),radius,(0,0,255),1)

    FileName = 'IrisSegmenation.jpg'
    im = Image.fromarray(img_save)
    im.save('Outputs/'+FileName)

    return int(center[0]), int(center[1]), int(radius)

def DrawOutline(FileName, Position):

    EyelidAndEyeCorner = EyelidAndEyeCornerSegmetation(FileName, Position)
    Iris = IrisSegmetation(FileName, Position)

    eye_image = Image.open(FileName)
    eye = np.asarray(eye_image)
    eye = eye.copy()
    eye_height, eye_width, eye_RGB = eye.shape

    white = eye #* 0
    #white += 255

    white = cv2.line(white, (EyelidAndEyeCorner[0][0], EyelidAndEyeCorner[0][1]), (EyelidAndEyeCorner[2][0], EyelidAndEyeCorner[2][1]), (0, 255, 0), 1)
    white = cv2.line(white, (EyelidAndEyeCorner[2][0], EyelidAndEyeCorner[2][1]), (EyelidAndEyeCorner[3][0], EyelidAndEyeCorner[3][1]), (0, 255, 0), 1)
    white = cv2.line(white, (EyelidAndEyeCorner[3][0], EyelidAndEyeCorner[3][1]), (EyelidAndEyeCorner[1][0], EyelidAndEyeCorner[1][1]), (0, 255, 0), 1)
    white = cv2.line(white, (EyelidAndEyeCorner[1][0], EyelidAndEyeCorner[1][1]), (EyelidAndEyeCorner[5][0], EyelidAndEyeCorner[5][1]), (0, 255, 0), 1)
    white = cv2.line(white, (EyelidAndEyeCorner[5][0], EyelidAndEyeCorner[5][1]), (EyelidAndEyeCorner[4][0], EyelidAndEyeCorner[4][1]), (0, 255, 0), 1)
    white = cv2.line(white, (EyelidAndEyeCorner[4][0], EyelidAndEyeCorner[4][1]), (EyelidAndEyeCorner[0][0], EyelidAndEyeCorner[0][1]), (0, 255, 0), 1)

    center = Iris[0], Iris[1]
    radius = Iris[2]

    white = cv2.circle(white, (center), radius, (0,255,255), 1)

# Redness
    # Left

    if EyelidAndEyeCorner[0][1] > center[1]:
        div = EyelidAndEyeCorner[0][1] - center[1]
        parts = (center[0] - EyelidAndEyeCorner[0][0])/ div
        b = EyelidAndEyeCorner[0][1]
        counter = parts
        flag = False

    elif EyelidAndEyeCorner[0][1] < center[1]:
        div = center[1] - EyelidAndEyeCorner[0][1]
        parts = (center[0] - EyelidAndEyeCorner[0][0])/ div
        counter = 0
        b = EyelidAndEyeCorner[0][1]
        flag = True

    else:
        div = 0
        parts = 0
        b = EyelidAndEyeCorner[0][1]
        counter = parts
        flag = False

    for a in range(EyelidAndEyeCorner[0][0], center[0]):
        if white[b][a][0] == 0 and white[b][a][1] == 255 and white[b][a][2] == 255:
            point_6w = a
            point_7w = a
            point_8w = a
            point_9w = a

            point_th = b
            break

        if flag:
            if counter == parts:
                b += 1
                counter = 0

            counter += 1

        elif not flag:
            if counter == 0:
                b -= 1
                counter = parts

            counter -= 1

    point_1h = (b + EyelidAndEyeCorner[0][1])/2
    point_1w = (a + EyelidAndEyeCorner[0][0])/2

    white = cv2.line(white, (EyelidAndEyeCorner[0][0], EyelidAndEyeCorner[0][1]), (point_1w, point_1h), (0, 255, 0), 1)

    point_2w = point_1w
    point_2h = 0

    for b in range(point_1h, -1, -1):
        if white[b][point_2w][0] == 0 and white[b][point_2w][1] == 255 and white[b][point_2w][2] == 0:
            point_2h = b

    point_3w = point_1w
    point_3h = 0

    for c in range(point_1h, eye_height):
        if white[c][point_3w][0] == 0 and white[c][point_3w][1] == 255 and white[c][point_3w][2] == 0:
            point_3h = c

    white = cv2.line(white,(point_2w, point_2h), (point_3w, point_3h) , (0, 255, 0), 1)

    p4A5Interval = (point_3h - point_2h)/3

    point_4w = point_1w
    point_4h = point_2h + p4A5Interval

    point_5w = point_1w
    point_5h = point_4h + p4A5Interval

    point_6h = 0

    for d in range(point_th, -1, -1):
        if white[d][point_6w][0] == 0 and white[d][point_6w][1] == 255 and white[d][point_6w][2] == 0:
            point_6h = d

    point_9h = 0

    for e in range(point_th, eye_height):
        if white[e][point_9w][0] == 0 and white[e][point_9w][1] == 255 and white[e][point_9w][2] == 0:
            point_9h = e

    white = cv2.line(white,(point_6w, point_6h), (point_9w, point_9h) , (0, 255, 0), 1)

    p7A8Interval = (point_9h - point_6h)/3

    point_7h = point_6h + p7A8Interval

    point_8h = point_7h + p7A8Interval

    white = cv2.line(white,(point_4w, point_4h), (point_7w, point_7h) , (0, 255, 0), 1)
    white = cv2.line(white,(point_5w, point_5h), (point_8w, point_8h) , (0, 255, 0), 1)

    # Right

    if EyelidAndEyeCorner[1][1] > center[1]:
        div = EyelidAndEyeCorner[1][1] - center[1]
        parts = (EyelidAndEyeCorner[1][0] - center[0])/ div
        b = EyelidAndEyeCorner[1][1]
        counter = parts
        flag = False

    elif EyelidAndEyeCorner[1][1] < center[1]:
        div = center[1] - EyelidAndEyeCorner[1][1]
        parts = (EyelidAndEyeCorner[1][0] - center[0])/ div
        counter = 0
        b = EyelidAndEyeCorner[1][1]
        flag = True

    else:
        div = 0
        parts = 0
        b = EyelidAndEyeCorner[1][1]
        counter = parts
        flag = False

    for a in range(EyelidAndEyeCorner[1][0], center[0], -1):
        if white[b][a][0] == 0 and white[b][a][1] == 255 and white[b][a][2] == 255:
            point_15w = a
            point_16w = a
            point_17w = a
            point_18w = a

            point_th = b
            break

        if flag:
            if counter == parts:
                b += 1
                counter = 0

            counter += 1

        elif not flag:
            if counter == 0:
                b -= 1
                counter = parts

            counter -= 1

    point_10h = (b + EyelidAndEyeCorner[1][1])/2
    point_10w = (a + EyelidAndEyeCorner[1][0])/2

    white = cv2.line(white, (EyelidAndEyeCorner[1][0], EyelidAndEyeCorner[1][1]), (point_10w, point_10h), (0, 255, 0), 1)

    point_11w = point_10w
    point_11h = 0

    for b in range(point_10h, -1, -1):
        if white[b][point_11w][0] == 0 and white[b][point_11w][1] == 255 and white[b][point_11w][2] == 0:
            point_11h = b

    point_12w = point_10w
    point_12h = 0

    for c in range(point_10h, eye_height):
        if white[c][point_12w][0] == 0 and white[c][point_12w][1] == 255 and white[c][point_12w][2] == 0:
            point_12h = c

    white = cv2.line(white,(point_11w, point_11h), (point_12w, point_12h) , (0, 255, 0), 1)

    p13A14Interval = (point_12h - point_11h)/3

    point_13w = point_10w
    point_13h = point_11h + p13A14Interval

    point_14w = point_10w
    point_14h = point_13h + p13A14Interval

    point_15h = 0

    for d in range(point_th, -1, -1):
        if white[d][point_15w][0] == 0 and white[d][point_15w][1] == 255 and white[d][point_15w][2] == 0:
            point_15h = d

    point_18h = 0

    for e in range(point_th, eye_height):
        if white[e][point_18w][0] == 0 and white[e][point_18w][1] == 255 and white[e][point_18w][2] == 0:
            point_18h = e

    white = cv2.line(white,(point_15w, point_15h), (point_18w, point_18h) , (0, 255, 0), 1)

    p16A17Interval = (point_18h - point_15h)/3

    point_16h = point_15h + p16A17Interval

    point_17h = point_16h + p16A17Interval

    white = cv2.line(white,(point_13w, point_13h), (point_16w, point_16h) , (0, 255, 0), 1)
    white = cv2.line(white,(point_14w, point_14h), (point_17w, point_17h) , (0, 255, 0), 1)

    #FileName = 'DrawOutline.jpg'
    im = Image.fromarray(white)
    im.save('Outputs/'+FileName)

    return [(point_1h, point_1w), (point_2h, point_2w), (point_3h, point_3w),
    (point_4h, point_4w), (point_5h, point_5w), (point_6h, point_6w),
    (point_7h, point_7w), (point_8h, point_8w), (point_9h, point_9w),
    (point_10h, point_10w), (point_11h, point_11w), (point_12h, point_12w),
    (point_13h, point_13w), (point_14h, point_14w), (point_15h, point_16w),
    (point_16h, point_16w), (point_17h, point_17w), (point_18h, point_18w),
    (EyelidAndEyeCorner[0][1], EyelidAndEyeCorner[0][0]), (EyelidAndEyeCorner[1][1], EyelidAndEyeCorner[1][0])]

def ScleraSegmentation(FileName, Position):

    EyelidAndEyeCorner = EyelidAndEyeCornerSegmetation(FileName, Position)
    Iris = IrisSegmetation(FileName, Position)

    p = DrawOutline(FileName, Position)

    eye_image = Image.open(FileName)
    eye = np.asarray(eye_image)
    eye = eye.copy()
    eye_height, eye_width, eye_RGB = eye.shape

    temp = eye.copy()

    eye = cv2.line(eye,(p[0][1], p[0][0]), (p[1][1], p[1][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[0][1], p[0][0]), (p[18][1], p[18][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[1][1], p[1][0]), (p[18][1], p[18][0]) , (0, 255, 0), 1)

    L1 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[0][1], p[0][0]), (p[18][1], p[18][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[0][1], p[0][0]), (p[2][1], p[2][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[18][1], p[18][0]), (p[2][1], p[2][0]) , (0, 255, 0), 1)

    L2 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[1][1], p[1][0]), (p[5][1], p[5][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[1][1], p[1][0]), (p[3][1], p[3][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[3][1], p[3][0]), (p[6][1], p[6][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[5][1], p[5][0]), (p[6][1], p[6][0]) , (0, 255, 0), 1)

    LM1 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[3][1], p[3][0]), (p[6][1], p[6][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[4][1], p[4][0]), (p[7][1], p[7][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[3][1], p[3][0]), (p[4][1], p[4][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[6][1], p[6][0]), (p[7][1], p[7][0]) , (0, 255, 0), 1)

    LM2 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[4][1], p[4][0]), (p[7][1], p[7][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[2][1], p[2][0]), (p[8][1], p[8][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[4][1], p[4][0]), (p[2][1], p[2][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[7][1], p[7][0]), (p[8][1], p[8][0]) , (0, 255, 0), 1)

    LM3 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[19][1], p[19][0]), (p[9][1], p[9][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[9][1], p[9][0]), (p[10][1], p[10][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[10][1], p[10][0]), (p[19][1], p[19][0]) , (0, 255, 0), 1)

    R1 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[19][1], p[19][0]), (p[9][1], p[9][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[9][1], p[9][0]), (p[11][1], p[11][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[11][1], p[11][0]), (p[19][1], p[19][0]) , (0, 255, 0), 1)

    R2 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[10][1], p[10][0]), (p[14][1], p[14][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[12][1], p[12][0]), (p[15][1], p[15][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[10][1], p[10][0]), (p[12][1], p[12][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[14][1], p[14][0]), (p[15][1], p[15][0]) , (0, 255, 0), 1)

    RM1 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[12][1], p[12][0]), (p[15][1], p[15][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[13][1], p[13][0]), (p[16][1], p[16][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[15][1], p[15][0]), (p[16][1], p[16][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[12][1], p[12][0]), (p[13][1], p[13][0]) , (0, 255, 0), 1)

    RM2 = Mask(eye)

    eye = temp.copy()

    eye = cv2.line(eye,(p[13][1], p[13][0]), (p[16][1], p[16][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[11][1], p[11][0]), (p[17][1], p[17][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[13][1], p[13][0]), (p[11][1], p[11][0]) , (0, 255, 0), 1)
    eye = cv2.line(eye,(p[16][1], p[16][0]), (p[17][1], p[17][0]) , (0, 255, 0), 1)

    RM3 = Mask(eye)

# L1 to L2
    L1_w1 = p[18][1] - 2
    L1_h1 = p[1][0] - 2

    L1_w2 = p[1][1] + 2

    if p[18][0] > p[0][0]:
        L1_h2 = p[18][0] + 2
        L2_h1 = p[0][0] - 2

    else:
        L1_h2 = p[0][0] + 2
        L2_h1 = p[18][0] - 2

    L2_w1 = p[18][1] - 2

    L2_w2 = p[2][1] + 2
    L2_h2 = p[2][0] + 2

# LM1 to LM3
    LM1_w1 = LM2_w1 = LM3_w1 = p[1][1] - 2

    if p[1][0] > p[5][0]:
        LM1_h1 = p[5][0] - 2

    else:
        LM1_h1 = p[1][0] - 2

    if p[5][1] > p[6][1]:
        LM1_w2 = p[5][1] + 2

    else:
        LM1_w2 = p[6][1] + 2

    if p[3][0] > p[6][0]:
        LM1_h2 = p[3][0] + 2
        LM2_h1 = p[6][0] -2

    else:
        LM1_h2 = p[6][0] + 2
        LM2_h1 = p[3][0] -2

    if p[6][1] > p[7][1]:
        LM2_w2 = p[6][1] + 2

    else:
        LM2_w2 = p[7][1] + 2

    if p[4][0] > p[7][0]:
        LM2_h2 = p[4][0] + 2
        LM3_h1 = p[7][0] - 2

    else:
        LM2_h2 = p[7][0] + 2
        LM3_h1 = p[4][0] - 2

    if p[7][1] > p[8][1]:
        LM3_w2 = p[7][1] + 2

    else:
        LM3_w2 = p[8][1] + 2

    if p[2][0] > p[8][0]:
        LM3_h2 = p[2][0] + 2

    else:
        LM3_h2 = p[8][0] + 2

# R1 to R2
    R1_w1 = R2_w1 = p[10][1] - 2
    R1_h1 = p[10][0] - 2

    R1_w2 = R2_w2 = p[19][1] + 2

    if p[9][0] > p[19][0]:
        R1_h2 = p[9][0] + 2
        R2_h1 = p[19][0] - 2

    else:
        R1_h2 = p[19][0] + 2
        R2_h1 = p[9][0] - 2

    R2_h2 = p[11][0] + 2

# RM1 to RM3

    if p[14][1] > p[15][1]:
        RM1_w1 = p[15][1] - 2

    else:
        RM1_w1 = p[14][1] - 2

    if p[14][0] > p[10][0]:
        RM1_h1 = p[10][0] - 2

    else:
        RM1_h1 = p[14][0] - 2

    RM1_w2 = RM2_w2 = RM3_w2 = p[12][1] + 2

    if p[12][0] > p[15][0]:
        RM1_h2 = p[12][0] + 2
        RM2_h1 = p[15][0] - 2

    else:
        RM1_h2 = p[15][0] + 2
        RM2_h1 = p[12][0] - 2

    if p[15][1] > p[16][1]:
        RM2_w1 = p[16][1] - 2

    else:
        RM2_w1 = p[15][1] - 2

    if p[13][0] > p[16][0]:
        RM2_h2 = p[13][0] + 2
        RM3_h1 = p[16][0] - 2

    else:
        RM2_h2 = p[16][0] + 2
        RM3_h1 = p[13][0] - 2

    if p[16][1] > p[17][1]:
        RM3_w1 = p[17][1] - 2

    else:
        RM3_w1 = p[16][1] - 2

    if p[11][0] > p[17][0]:
        RM3_h2 = p[11][0] + 2

    else:
        RM3_h2 = p[17][0] + 2

    FileName = 'L1.jpg'
    im = Image.fromarray(L1)
    im = im.crop((L1_w1, L1_h1, L1_w2, L1_h2))
    im.save('Outputs/'+FileName)

    FileName = 'L2.jpg'
    im = Image.fromarray(L2)
    im = im.crop((L2_w1, L2_h1, L2_w2, L2_h2))
    im.save('Outputs/'+FileName)

    FileName = 'LM1.jpg'
    im = Image.fromarray(LM1)
    im = im.crop((LM1_w1, LM1_h1, LM1_w2, LM1_h2))
    im.save('Outputs/'+FileName)

    FileName = 'LM2.jpg'
    im = Image.fromarray(LM2)
    im = im.crop((LM2_w1, LM2_h1, LM2_w2, LM2_h2))
    im.save('Outputs/'+FileName)

    FileName = 'LM3.jpg'
    im = Image.fromarray(LM3)
    im = im.crop((LM3_w1, LM3_h1, LM3_w2, LM3_h2))
    im.save('Outputs/'+FileName)

    FileName = 'R1.jpg'
    im = Image.fromarray(R1)
    im = im.crop((R1_w1, R1_h1, R1_w2, R1_h2))
    im.save('Outputs/'+FileName)

    FileName = 'R2.jpg'
    im = Image.fromarray(R2)
    im = im.crop((R2_w1, R2_h1, R2_w2, R2_h2))
    im.save('Outputs/'+FileName)

    FileName = 'RM1.jpg'
    im = Image.fromarray(RM1)
    im = im.crop((RM1_w1, RM1_h1, RM1_w2, RM1_h2))
    im.save('Outputs/'+FileName)

    FileName = 'RM2.jpg'
    im = Image.fromarray(RM2)
    im = im.crop((RM2_w1, RM2_h1, RM2_w2, RM2_h2))
    im.save('Outputs/'+FileName)

    FileName = 'RM3.jpg'
    im = Image.fromarray(RM3)
    im = im.crop((RM3_w1, RM3_h1, RM3_w2, RM3_h2))
    im.save('Outputs/'+FileName)

def Mask(mask):

    mask_height, mask_width, mask_RGB = mask.shape

    # Left to Right
    for a in range(mask_height):
        for b in range(mask_width):
            if mask[a][b][0] == 0 and mask[a][b][1] == 255 and mask[a][b][2] == 0:
                break

            else:
                mask[a][b][0] = 0
                mask[a][b][1] = 0
                mask[a][b][2] = 0

    # Right to Left
    for a in range(mask_height-1, -1, -1):
        for b in range(mask_width-1, -1, -1):
            if mask[a][b][0] == 0 and mask[a][b][1] == 255 and mask[a][b][2] == 0:
                break

            else:
                mask[a][b][0] = 0
                mask[a][b][1] = 0
                mask[a][b][2] = 0

    # Top to Bottom
    for b in range(mask_width):
        for a in range(mask_height):
            if mask[a][b][0] == 0 and mask[a][b][1] == 255 and mask[a][b][2] == 0:
                break

            else:
                mask[a][b][0] = 0
                mask[a][b][1] = 0
                mask[a][b][2] = 0

    # Bottom to Top
    for b in range(mask_width-1, -1, -1):
        for a in range(mask_height-1, -1, -1):
            if mask[a][b][0] == 0 and mask[a][b][1] == 255 and mask[a][b][2] == 0:
                break

            else:
                mask[a][b][0] = 0
                mask[a][b][1] = 0
                mask[a][b][2] = 0

    # Erase Green
    for a in range(mask_height-1, -1, -1):
        for b in range(mask_width-1, -1, -1):
            if mask[a][b][0] == 0 and mask[a][b][1] == 255 and mask[a][b][2] == 0:
                mask[a][b][0] = 0
                mask[a][b][1] = 0
                mask[a][b][2] = 0

    return mask

def RegiontoNumber():

    L1_image = Image.open('Outputs/'+'L1.jpg')
    L1 = np.asarray(L1_image)
    L1 = L1.copy()
    L1_height, L1_width, L1_RGB = L1.shape
    L1_CR, L1_S, L1_R, L1_G, L1_B = CheckRednessandIdentifySeverity(L1, L1_height, L1_width)

    L2_image = Image.open('Outputs/'+'L2.jpg')
    L2 = np.asarray(L2_image)
    L2 = L2.copy()
    L2_height, L2_width, L2_RGB = L2.shape
    L2_CR, L2_S, L2_R, L2_G, L2_B = CheckRednessandIdentifySeverity(L2, L2_height, L2_width)

    LM1_image = Image.open('Outputs/'+'LM1.jpg')
    LM1 = np.asarray(LM1_image)
    LM1 = LM1.copy()
    LM1_height, LM1_width, LM1_RGB = LM1.shape
    LM1_CR, LM1_S, LM1_R, LM1_G, LM1_B = CheckRednessandIdentifySeverity(LM1, LM1_height, LM1_width)

    LM2_image = Image.open('Outputs/'+'LM2.jpg')
    LM2 = np.asarray(LM2_image)
    LM2 = LM2.copy()
    LM2_height, LM2_width, LM2_RGB = LM2.shape
    LM2_CR, LM2_S, LM2_R, LM2_G, LM2_B = CheckRednessandIdentifySeverity(LM2, LM2_height, LM2_width)

    LM3_image = Image.open('Outputs/'+'LM3.jpg')
    LM3 = np.asarray(LM3_image)
    LM3 = LM3.copy()
    LM3_height, LM3_width, LM3_RGB = LM3.shape
    LM3_CR, LM3_S, LM3_R, LM3_G, LM3_B = CheckRednessandIdentifySeverity(LM3, LM3_height, LM3_width)

    R1_image = Image.open('Outputs/'+'R1.jpg')
    R1 = np.asarray(R1_image)
    R1 = R1.copy()
    R1_height, R1_width, R1_RGB = R1.shape
    R1_CR, R1_S, R1_R, R1_G, R1_B = CheckRednessandIdentifySeverity(R1, R1_height, R1_width)

    R2_image = Image.open('Outputs/'+'R2.jpg')
    R2 = np.asarray(R2_image)
    R2 = R2.copy()
    R2_height, R2_width, R2_RGB = R2.shape
    R2_CR, R2_S, R2_R, R2_G, R2_B = CheckRednessandIdentifySeverity(R2, R2_height, R2_width)

    RM1_image = Image.open('Outputs/'+'RM1.jpg')
    RM1 = np.asarray(RM1_image)
    RM1 = RM1.copy()
    RM1_height, RM1_width, RM1_RGB = RM1.shape
    RM1_CR, RM1_S, RM1_R, RM1_G, RM1_B = CheckRednessandIdentifySeverity(RM1, RM1_height, RM1_width)

    RM2_image = Image.open('Outputs/'+'RM2.jpg')
    RM2 = np.asarray(RM2_image)
    RM2 = RM2.copy()
    RM2_height, RM2_width, RM2_RGB = RM2.shape
    RM2_CR, RM2_S, RM2_R, RM2_G, RM2_B = CheckRednessandIdentifySeverity(RM2, RM2_height, RM2_width)

    RM3_image = Image.open('Outputs/'+'RM3.jpg')
    RM3 = np.asarray(RM3_image)
    RM3 = RM3.copy()
    RM3_height, RM3_width, RM3_RGB = RM3.shape
    RM3_CR, RM3_S, RM3_R, RM3_G, RM3_B = CheckRednessandIdentifySeverity(RM3, RM3_height, RM3_width)

    #print 'RegiontoNumber'

    Unsorted = [(L1_R, L1_G, L1_B), (L2_R, L2_G, L2_B), (LM1_R, LM1_G, LM1_B), (LM2_R, LM2_G, LM2_B), (LM3_R, LM3_G, LM3_B),
         (R1_R, R1_G, R1_B), (R2_R, R2_G, R2_B), (RM1_R, RM1_G, RM1_B), (RM2_R, RM2_G, RM2_B), (RM3_R, RM3_G, RM3_B)]

    Sorted = sorted(Unsorted)
    rp1 = []

    #print Unsorted
    #print Sorted
    #print sorted(Unsorted)

    index_US_S = sorted(range(len(Unsorted)),key=Unsorted.__getitem__)

    #print index_US_S

    for a in range(10):
        if index_US_S[a] >= 0 and index_US_S[a] < 2:
            rp1.append(0)
        elif index_US_S[a] >= 2 and index_US_S[a] <5:
            rp1.append(1)
        elif index_US_S[a] >= 5 and index_US_S[a] <8:
            rp1.append(2)
        elif index_US_S[a] >= 8 and index_US_S[a] <10:
            rp1.append(3)

    #print len(rp1)
    print 'Initial rp1', rp1


    rp2 = []

    ctr = 0

    for a in range(9):
        if abs(Sorted[a][0] - Sorted[a+1][0]) < 10:
            rp2.append(ctr)

        else:
            ctr += 1
            rp2.append(ctr)

    print 'rp2', rp2

    for a in range(8, 1, -1):
        if rp2[a] == rp2[a-1]:
            rp1[a-1] = rp1[a]

        else:
            continue

    print 'Final rp1', rp1

    #print L1_CR, L2_CR, LM1_CR, LM2_CR, LM3_CR, R1_CR, R2_CR, RM1_CR, RM2_CR, RM3_CR

def CheckRednessandIdentifySeverity(M, height, width):

        black = 0
        white = 0
        gray = 0
        red = 0
        green = 0
        blue = 0
        unknown = 0
        R = 0.0
        G = 0.0
        B = 0.0

        for a in range(height):
            for b in range(width):

                if (M[a][b][0] == 0 and M[a][b][1] == 0 and M[a][b][2] == 0) or (int(M[a][b][0] + M[a][b][1] + M[a][b][2]) < 50):
                    black += 1

                elif ((int(M[a][b][0] + M[a][b][1] + M[a][b][2]) / 3) == M[a][b][1] ):
                    white += 1

                #elif (M[a][b][0] > M[a][b][1]) and (M[a][b][0] > M[a][b][2]) or (M[a][b][1] > M[a][b][2] and M[a][b][1] < M[a][b][2] + 15) or (M[a][b][2] > M[a][b][1] and M[a][b][2] < M[a][b][1] + 15) :
                #    brown += 1

                elif (M[a][b][0] > M[a][b][1]) and (M[a][b][0] > M[a][b][2]) and (M[a][b][1] == M[a][b][2]):
                    red += 1
                    R += M[a][b][0]
                    G += M[a][b][1]
                    B += M[a][b][2]

                elif (M[a][b][0] > M[a][b][1]) and (M[a][b][0] > M[a][b][2]) and (M[a][b][1] < M[a][b][2]):
                    red += 1
                    R += M[a][b][0]
                    G += M[a][b][1]
                    B += M[a][b][2]

                elif (M[a][b][1] > M[a][b][0]) and (M[a][b][1] > M[a][b][2]) and (M[a][b][0] > M[a][b][2]):
                    green += 1

                elif (M[a][b][2] > M[a][b][0]) and (M[a][b][2] > M[a][b][1]) and (M[a][b][1] > M[a][b][0]):
                    blue += 1

                else:
                    unknown += 1


        TotalNoPixels = height * width
        TotalNoRegion = TotalNoPixels - black - unknown

        if red > 0:
            RA = R / red
            GA = G / red
            BA = B / red

        else:
            RA = 0
            GA = 0
            BA = 0


        if (TotalNoRegion/2) < red -5:
            redness = 1

        else:
            redness = 0

        red_percentage = 0.0
        severity = 0

        if red > 0:
            red_percentage = (red/TotalNoRegion) * 100

            if red_percentage < 20.0 and red_percentage > 0.0:
                severity = 0

            elif red_percentage < 50.0 and red_percentage > 20.0:
                severity = 1

            elif red_percentage < 75.0 and red_percentage > 50.0:
                severity = 2

            elif red_percentage < 100.0 and red_percentage > 75.0:
                severity = 3
        else:
            severity = 0


        return redness, severity, RA, GA, BA


        #print 'Black  '+str(black)
        #print 'White  '+str(white)
        #print 'Gray  '+str(gray)
        #print 'Red  '+str(red)
        #print 'Green  '+str(green)
        #print 'Blue  '+str(blue)
        #print TotalNoPixels

        #print ''
