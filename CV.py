# # CODE FOR MIRRORING VIDEO
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     # Obtain the dimensions of our frame
#     width = int(cap.get(3))
#     height = int(cap.get(4))
#
#     # Create a new frame
#     image = np.zeros(frame.shape, np.uint8)
#     smaller_img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#
#     image[:height // 2, :width // 2] = cv2.rotate(smaller_img, cv2.ROTATE_180)
#     image[height // 2:, :width // 2] = smaller_img
#     image[:height // 2, width // 2:] = cv2.rotate(smaller_img, cv2.ROTATE_180)
#     image[height // 2:, width // 2:] = smaller_img
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_bound = np.array([50, 50, 90])
#     upper_bound = np.array([255, 255, 130])
#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#     cv2.imshow('Frame', result)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


# # CORNER DETECTION
# import numpy as np
# import cv2
#
# img = cv2.imread('assets/chess.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# corners = cv2.goodFeaturesToTrack(gray, 100, 0.001, 10)
# corners = np.int0(corners)
#
# for corner in corners:
#     x, y = corner.ravel()
#     cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
#
# # for i in range(len(corners)):
# #     for j in range(i + 1, len(corners)):
# #         corner1 = tuple(corners[i][0])
# #         corner2 = tuple(corners[j][0])
# #         color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
# #         cv2.line(img, corner1, corner2, color, 1)
#
# cv2.imshow('Frame', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # TEMPLATE MATCHING(OBJECT DETECTION)
# import cv2
# import numpy as np
#
# img = cv2.resize(cv2.imread('./assets/soccer_practice.jpg', 0), (0, 0), fx=0.7, fy=0.7)
# template = cv2.resize(cv2.imread('./assets/shoe.PNG', 0), (0, 0), fx=0.7, fy=0.7)
#
# h, w = template.shape
#
# # Different methods of doing template matching
# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
#
# for method in methods:
#     img2 = img.copy()
#
#     result = cv2.matchTemplate(img2, template, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     print(method)
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         location = min_loc
#     else:
#         location = max_loc
#     bottom_right = (location[0] + w, location[1] + h)
#     cv2.rectangle(img2, location, bottom_right, 255, 5)
#     cv2.imshow('Match', img2)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
import numpy as np

base_image = cv2.imread('./assets/defcon.png', 0)
template = cv2.imread('./assets/disobey.jpg', 0)

height, width = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]


for method in methods:
    base_image_copy = base_image.copy()

    output_image = cv2.matchTemplate(base_image_copy, template, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(output_image)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0] + height, location[1] + width)
    cv2.rectangle(base_image_copy, location, bottom_right, 255, 2)
    cv2.imshow('Def-HEAD', base_image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
