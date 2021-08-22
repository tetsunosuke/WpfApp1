# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np


# ベクトルを描画する
def drawAxis(img, start_pt, vec, colour, length):
    # アンチエイリアス
    CV_AA = 16

    # 終了点
    end_pt = (int(start_pt[0] + length * vec[0]), int(start_pt[1] + length * vec[1]))

    # 中心を描画
    cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 5, colour, 1)

    # 軸線を描画
    cv2.line(img, (int(start_pt[0]), int(start_pt[1])), end_pt, colour, 1, CV_AA);

    # 先端の矢印を描画
    angle = math.atan2(vec[1], vec[0])

    qx0 = int(end_pt[0] - 9 * math.cos(angle + math.pi / 4));
    qy0 = int(end_pt[1] - 9 * math.sin(angle + math.pi / 4));
    cv2.line(img, end_pt, (qx0, qy0), colour, 1, CV_AA);

    qx1 = int(end_pt[0] - 9 * math.cos(angle - math.pi / 4));
    qy1 = int(end_pt[1] - 9 * math.sin(angle - math.pi / 4));
    cv2.line(img, end_pt, (qx1, qy1), colour, 1, CV_AA);


if __name__ == '__main__':
    # 画像を読み込む
    src = cv2.imread("test.jpg")

    # グレースケールに変換
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # ２値化
    retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 輪郭を抽出
    #   contours : [領域][Point No][0][x=0, y=1]
    #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
    #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
    #img, contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #img, contours = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
    # 輪郭を描画する
    cv2.drawContours(src, contours, -1, (0, 0, 255))

    # 各輪郭に対する処理
    for i in range(0, len(contours)):

        # 輪郭の領域を計算
        area = cv2.contourArea(contours[i])

        # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
        if area < 10 or 1e5 < area:
            continue
        #print(i, area)


        # 輪郭データを浮動小数点型の配列に格納
        print(contours[i].shape[0], contours[i].shape[2])
        X = np.array(contours[i], dtype=np.float64).reshape((contours[i].shape[0], contours[i].shape[2]))
        print("X", X)

        # PCA（１次元）
        mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float64), maxComponents=1)
        print("mean", mean)
        print("eigenvectors", eigenvectors)

        # 主成分方向のベクトルを描画
        pt = (mean[0][0], mean[0][1])
        #print(mean)
        #print(eigenvectors)
        vec = (eigenvectors[0][0], eigenvectors[0][1])
        drawAxis(src, pt, vec, (255, 255, 0), 150)


    # 表示
    cv2.imshow('output', src)
    cv2.waitKey(0)

    # 終了処理
    cv2.destroyAllWindows()