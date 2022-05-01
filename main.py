import cv2
import numpy as np
import math

desk_img_path = 'images/desk2.png'
prefix_path = 'images/chess/'
black_chess_piece_arr = ['black_pawn.png',
                         'black_horse.png',
                         'black_king.png',
                         'black_rook.png',
                         'black_bishop.png',
                         'black_queen.png']  # threshold = 0.8
white_chess_piece_arr = ['white_pawn.png',
                         'white_horse.png',
                         'white_rook.png',
                         'white_bishop.png',
                         'white_queen.png',
                         'white_king.png']  # threshold = 0.4


def get_img_rect(image_file):
    img = cv2.imread(image_file, 0)
    w_pawn_white, h_pawn_white = img.shape[::-1]
    return [w_pawn_white, h_pawn_white]


def convert_to_black_white(image_file):
    img = cv2.imread(image_file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImg) = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImg


def fix_duplication(loc_arr):
    threshold = 15
    indexes_to_removing = []
    index = 0
    prev_rect = [0, 0]

    for pt in loc_arr[1]:
        if index != 0:
            j = index
            for pt2 in loc_arr[1]:
                if j >= len(loc_arr[1]):
                    break
                if math.fabs(prev_rect[1] - loc_arr[1][j]) + math.fabs(prev_rect[0] - loc_arr[0][j]) < threshold:
                    indexes_to_removing.append(index-1)
                    break
                j = j + 1
        prev_rect = [loc_arr[0][index], loc_arr[1][index]]
        index = index + 1

    arr0 = np.delete(loc_arr[0], indexes_to_removing)
    arr1 = np.delete(loc_arr[1], indexes_to_removing)

    return (arr0, arr1)


def find_chess_piece_position(origin_desk, chess_piece_img, desk_img, threshold):
    desk_black_white = convert_to_black_white(chess_piece_img)
    chess_piece_black_white_template = convert_to_black_white(desk_img)
    img_rect = get_img_rect(chess_piece_img)

    res = cv2.matchTemplate(desk_black_white, chess_piece_black_white_template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)

    fixed_loc = fix_duplication(loc)

    for pt in zip(*fixed_loc[::-1]):
        cv2.rectangle(origin_desk, pt, (pt[0] + img_rect[0], pt[1] + img_rect[1]), (255, 0, 0), 1)
        chess_name = chess_piece_img.replace(prefix_path, '').replace('.png', '')
        cv2.putText(origin_desk, chess_name, (pt[0]-10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))


def find_rect(origin_desk, image_file):
    # Read input image
    img = cv2.imread(image_file)

    # convert from BGR to HSV color space
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    desk_black_white = convert_to_black_white(image_file)

    # apply threshold
    thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)[1]

    # find contours and get one with area about 180*35
    # draw all contours in green and accepted ones in red
    contours = cv2.findContours(desk_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # area_thresh = 0
    min_area = 0.95 * 180 * 35
    max_area = 1.05 * 180 * 35
    result = img.copy()

    counter = []

    for c in contours:
        area = cv2.contourArea(c)
        # cv2.drawContours(result, [c], -1, (0, 255, 0), 4)
        # 1912589.0
        if area > 1000000 and area < 1800000:
            counter.append(area)
            # cv2.drawContours(result, [c], -1, (0, 0, 255), 4)
            min_x = 1000000
            min_y = 1000000
            max_x = 0
            max_y = 0
            for i in c:
                if i[0][0] < min_x:
                    min_x = i[0][0]
                if i[0][1] < min_y:
                    min_y = i[0][1]
                if i[0][0] > max_x:
                    max_x = i[0][0]
                if i[0][1] > max_y:
                    max_y = i[0][1]

    cv2.rectangle(origin_desk, (min_x, min_y), (max_x, max_y), (0, 0, 255), 5)
    desk_name = desk_img_path.replace('.png', '')
    cv2.putText(origin_desk, desk_name, (min_x - 15, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    return [[min_x, min_y], [max_x, max_y]]


def main():
    desk = cv2.imread(desk_img_path)

    desk_coord = find_rect(desk, desk_img_path)

    for black_chess_piece in black_chess_piece_arr:
        find_chess_piece_position(desk, prefix_path + black_chess_piece, desk_img_path, 0.8)

    for white_chess_piece in white_chess_piece_arr:
        find_chess_piece_position(desk, prefix_path + white_chess_piece, desk_img_path, 0.4)

    cv2.imshow('detected', desk)
    cv2.waitKey(0)


# run app
main()
