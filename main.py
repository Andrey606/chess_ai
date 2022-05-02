import cv2
import numpy as np
import math
from enum import Enum
import time
from mss import mss
from PIL import Image

class Chess(Enum):
    PAWN = 1,
    HORSE = 2,
    KING = 3,
    ROOK = 4,
    BISHOP = 5,
    QUEEN = 6


class ChessColor(Enum):
    WHITE = 1,
    BLACK = 2


desk_img_path = 'images/desk2.png'
prefix_path = 'images/chess/'
# threshold = 0.8
black_chess_piece_arr = [{'path': 'black_pawn.png', 'color': ChessColor.BLACK, 'model': Chess.PAWN},
                         {'path': 'black_horse.png', 'color': ChessColor.BLACK, 'model': Chess.HORSE},
                         {'path': 'black_king.png', 'color': ChessColor.BLACK, 'model': Chess.KING},
                         {'path': 'black_rook.png', 'color': ChessColor.BLACK, 'model': Chess.ROOK},
                         {'path': 'black_bishop.png', 'color': ChessColor.BLACK, 'model': Chess.BISHOP},
                         {'path': 'black_queen.png', 'color': ChessColor.BLACK, 'model': Chess.QUEEN}]
# threshold = 0.6
white_chess_piece_arr = [{'path': 'white_pawn.png', 'color': ChessColor.WHITE, 'model': Chess.PAWN},
                         {'path': 'white_horse.png', 'color': ChessColor.WHITE, 'model': Chess.HORSE},
                         {'path': 'white_king.png', 'color': ChessColor.WHITE, 'model': Chess.KING},
                         {'path': 'white_rook.png', 'color': ChessColor.WHITE, 'model': Chess.ROOK},
                         {'path': 'white_bishop.png', 'color': ChessColor.WHITE, 'model': Chess.BISHOP},
                         {'path': 'white_queen.png', 'color': ChessColor.WHITE, 'model': Chess.QUEEN}]


def get_img_rect(image_file):
    img = cv2.imread(image_file, 0)
    w_pawn_white, h_pawn_white = img.shape[::-1]
    return [w_pawn_white, h_pawn_white]


def convert_to_black_white(image_file):
    # img = cv2.imread(image_file)
    gray_img = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImg) = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImg


def fix_duplication(loc_arr):
    threshold = 100
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

    return arr0, arr1


def find_chess_piece_position(origin_desk, chess_piece_img, desk_img, threshold):
    desk_black_white = convert_to_black_white(cv2.imread(prefix_path + chess_piece_img['path']))
    chess_piece_black_white_template = convert_to_black_white(origin_desk)
    img_rect = get_img_rect(prefix_path + chess_piece_img['path'])

    res = cv2.matchTemplate(desk_black_white, chess_piece_black_white_template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)

    fixed_loc = fix_duplication(loc)

    chess_piece_positions = []
    for pt in zip(*fixed_loc[::-1]):
        chess_piece_positions.append({'position': pt,
                                      'color': chess_piece_img['color'],
                                      'model': chess_piece_img['model']})
        cv2.rectangle(origin_desk, pt, (pt[0] + img_rect[0], pt[1] + img_rect[1]), (255, 0, 0), 1)
        chess_name = chess_piece_img['path'].replace(prefix_path, '').replace('.png', '')
        cv2.putText(origin_desk, chess_name, (pt[0], pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    return chess_piece_positions


def find_rect(origin_desk, image_file):
    # Read input image
    # img = cv2.imread(image_file)

    # convert from BGR to HSV color space
    gray = cv2.cvtColor(origin_desk, cv2.COLOR_BGR2GRAY)
    desk_black_white = convert_to_black_white(origin_desk)

    # apply threshold
    thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)[1]

    # find contours and get one with area about 180*35
    # draw all contours in green and accepted ones in red
    contours = cv2.findContours(desk_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # result = img.copy()

    min_x = 1000000
    min_y = 1000000
    max_x = 0
    max_y = 0
    for c in contours:
        area = cv2.contourArea(c)
        # cv2.drawContours(result, [c], -1, (0, 255, 0), 4)
        # 1912589.0
        if area > 1000000 and area < 1800000:
            # cv2.drawContours(result, [c], -1, (0, 0, 255), 4)

            for i in c:
                if i[0][0] < min_x:
                    min_x = i[0][0]
                if i[0][1] < min_y:
                    min_y = i[0][1]
                if i[0][0] > max_x:
                    max_x = i[0][0]
                if i[0][1] > max_y:
                    max_y = i[0][1]

    cv2.rectangle(origin_desk, (min_x, min_y), (max_x, max_y), (0, 0, 255), 4)
    desk_name = desk_img_path.replace('.png', '')
    cv2.putText(origin_desk, desk_name, (min_x, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    return [[min_x, min_y], [max_x, max_y]]


def find_chess_positions(origin_desk, desk_coord, chess_coord):
    # x = horizont (a, b, c, d, e, f, g, h)
    # y = vertical (8, 7, 6, 5, 4, 3, 2, 1)
    # desk_coord - [[x, y], [x, y]]
    rect_size = [(desk_coord[1][0] - desk_coord[0][0])/8, (desk_coord[1][1] - desk_coord[0][1])/8]

    result = []
    for chess in chess_coord:
        pos = str(9 - math.ceil((chess['position'][1] - desk_coord[0][1])/rect_size[0])) + \
              chr(64 + math.ceil((chess['position'][0] - desk_coord[0][0])/rect_size[0]))
        result.append({'color': chess['color'], 'model': chess['model'], 'position': pos})
        cv2.putText(origin_desk,
                    pos,
                    (chess['position'][0], chess['position'][1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0))

    return result


def detect_chess(screen):
    # desk = cv2.imread(desk_img_path)

    start_time = time.time()

    chess_coord = []
    for black_chess_piece in black_chess_piece_arr:
        chess_coord += find_chess_piece_position(screen, black_chess_piece, desk_img_path, 0.8)

    print("1--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    for white_chess_piece in white_chess_piece_arr:
        chess_coord += find_chess_piece_position(screen, white_chess_piece, desk_img_path, 0.6)

    print("2--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    desk_coord = find_rect(screen, desk_img_path)
    chess_positions = find_chess_positions(screen, desk_coord, chess_coord)

    print("3--- %s seconds ---" % (time.time() - start_time))


def main():
    mon = {'top': 0, 'left': 750, 'width': 1300, 'height': 750}
    sct = mss()

    while 1:
        sct.get_pixels(mon)
        img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)

        screen = np.array(img)
        detect_chess(screen)
        cv2.imshow('test', screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()
