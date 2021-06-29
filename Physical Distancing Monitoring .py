# imports
import cv2
import numpy as np
import time
import argparse

confid = 0.5
thresh = 0.5
mouse_pts = []


# Function to draw Bird Eye View for region of interest(ROI). Red, Yellow, Green points represents risk to human.
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk
def bird_eye_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white
    warped_pts = []
    r = []
    g = []
    y = []
    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image,
                                   (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)),
                                   (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1] * scale_h)), red,
                                   2)

    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image,
                                   (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)),
                                   (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1] * scale_h)),
                                   yellow, 2)

    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])

    for i in bottom_points:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, green, 10)
    for i in y:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
    for i in r:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, red, 10)

    # pad = np.full((100,blank_image.shape[1],3), [110, 110, 100], dtype=np.uint8)
    # cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # blank_image = np.vstack((blank_image,pad))

    return blank_image


# Function to draw bounding boxes according to risk factor for humans in a frame and draw lines between
# boxes according to risk factor between two humans.
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk
def social_distancing_view(frame, distances_mat, boxes, risk_count):
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)

    for i in range(len(boxes)):
        x, y, w, h = boxes[i][:]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)

    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]

        if closeness == 1:
            x, y, w, h = per1[:]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), yellow, 2)

            x1, y1, w1, h1 = per2[:]
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), yellow, 2)

            frame = cv2.line(frame, (int(x + w / 2), int(y + h / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)), yellow, 2)

    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]

        if closeness == 0:
            x, y, w, h = per1[:]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), red, 2)

            x1, y1, w1, h1 = per2[:]
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), red, 2)

            frame = cv2.line(frame, (int(x + w / 2), int(y + h / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)), red, 2)

    pad = np.full((140, frame.shape[1], 3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (100, 100, 0), 2)
    cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 1)
    cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 1)
    cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 1)
    frame = np.vstack((frame, pad))

    return frame


# !/usr/bin/env python

'''
Contains functions to calculate bottom center for all bounding boxes and transform prespective for all points,
calculate distance between humans, calculate width and height scale ratio for bird eye view,
and calculates number of humans at risk, low risk, no risk according to closeness.
'''


# Function to calculate bottom center for all bounding boxes and transform prespective for all points.
def get_transformed_points(boxes, prespective_transform):
    bottom_points = []
    for box in boxes:
        pnts = np.array([[[int(box[0] + (box[2] * 0.5)), int(box[1] + box[3])]]], dtype="float32")
        # pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+(box[3]*0.5))]]] , dtype="float32")
        bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
        bottom_points.append(pnt)

    return bottom_points


# Function calculates distance between two points(humans). distance_w, distance_h represents number
# of pixels in 180cm length horizontally and vertically. We calculate horizontal and vertical
# distance in pixels for two points and get ratio in terms of 180 cm distance using distance_w, distance_h.
# Then we calculate how much cm distance is horizontally and vertically and then using pythagoras
# we calculate distance between points in terms of cm.
def cal_dis(p1, p2, distance_w, distance_h):
    h = abs(p2[1] - p1[1])
    w = abs(p2[0] - p1[0])

    dis_w = float((w / distance_w) * 180)
    dis_h = float((h / distance_h) * 180)

    return int(np.sqrt(((dis_h) ** 2) + ((dis_w) ** 2)))


# Function calculates distance between all pairs and calculates closeness ratio.
def get_distances(boxes1, bottom_points, distance_w, distance_h):
    distance_mat = []
    bxs = []

    for i in range(len(bottom_points)):
        for j in range(len(bottom_points)):
            if i != j:
                dist = cal_dis(bottom_points[i], bottom_points[j], distance_w, distance_h)
                # dist = int((dis*180)/distance)
                if dist <= 150:
                    closeness = 0
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                elif dist > 150 and dist <= 180:
                    closeness = 1
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                else:
                    closeness = 2
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])

    return distance_mat, bxs


# Function gives scale for birds eye view
def get_scale(W, H):
    dis_w = 400
    dis_h = 600

    return float(dis_w / W), float(dis_h / H)


# Function gives count for humans at high risk, low risk and no risk
def get_count(distances_mat):
    r = []
    g = []
    y = []

    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])

    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])

    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])

    return (len(r), len(y), len(g))


# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click
# event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel
# lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in
# horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form
# horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different.

# Function will be called on mouse events

def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]), (70, 70, 70),
                     2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        # print("Point detected")
        # print(mouse_pts)


def calculate_social_distancing():
    count = 0
    vs = cv2.VideoCapture('input_vid_2.mp4')

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))

    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    scale_w, scale_h = get_scale(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_movie = cv2.VideoWriter("output_movie.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))
    bird_movie = cv2.VideoWriter("bird_eye_view.avi", fourcc, fps,(int(width * scale_w), int(height * scale_h)))

    points = []
    global image

    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            print('here')
            break

        (H, W) = frame.shape[:2]

        # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("image")
                    break

            points = mouse_pts

            # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are
        # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
        # This bird eye view then has the property property that points are distributed uniformally horizontally and
        # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are
        # equally distributed, which was not case for normal view.
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

        ####################################################################################

        # YOLO v3
        net =  cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
        ln = net.getLayerNames()
        ln1 = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detecting humans in frame
                if classID == 0:

                    if confidence > confid:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x, y, w, h = boxes[i]

        if len(boxes1) == 0:
            count = count + 1
            continue

        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        person_points = get_transformed_points(boxes1, prespective_transform)

        # Here we will calculate distance between transformed points(humans)
        distances_mat, bxs_mat = get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = get_count(distances_mat)

        frame1 = np.copy(frame)

        # Draw bird eye view and frame with bouding boxes around humans according to risk factor
        bird_image = bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
        img = social_distancing_view(frame1, bxs_mat, boxes1, risk_count)

        # Show/write image and videos
        if count != 0:
            output_movie.write(img)
            #bird_movie.write(bird_image)
            cv2.imshow('output movie',img)
            cv2.imshow('Bird Eye View', bird_image)
            #cv2.imwrite("frame%d.jpg" % count, img)
            #cv2.imwrite("bird_eye_view/frame%d.jpg" % count, bird_image)

        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()


# set mouse callback

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
np.random.seed(42)

calculate_social_distancing()

