from imutils import face_utils
import cv2
import sys
#from mtcnn.mtcnn import MTCNN
import time
import dlib
import argparse

class MaskDetect:
    def __init__(self, args) -> None:
        if args.mtcnn:
            self.mtcnn = MTCNN()
            self.detector = self.mtcnn_detector
        else:
            self.dlib_fd = dlib.get_frontal_face_detector()
            #self.dlib_sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.detector = self.dlib_detector

        self.camera = cv2.VideoCapture(0)

        self.record = args.record
        frame_rate = int(self.camera.get(cv2.CAP_PROP_FPS)) # フレームレート
        size = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 動画の画面サイズ
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        self.writer = cv2.VideoWriter('./output.mp4', fmt, frame_rate, size)

    def preprocess(self):
        # 切り抜き処理とか画像に前処理するならここでやる
        return None

    def predict(self):
        # 推論気に渡して結果を得る
        return None

    def postprocess(self):
        # バウンディングボックスの描画
        if self.x is not None:
            cv2.rectangle(self.frame, (self.x, self.y), (self.x+self.w , self.y+self.h), (0,0,255), thickness=3)
        
        return

    def mtcnn_detector(self):
        img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.mtcnn.detect_faces(img)

        # 今は顔ひとつだけを想定
        for face in faces:
            return face['box']
        else:
            return None, None, None, None
        
    def dlib_detector(self):
        # Convert to gray scale
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.dlib_fd(gray)
        # 今は顔ひとつだけを想定
        for face in faces:
            return face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        else:
            return None, None, None, None

    def run(self):
        while True:
            # カメラからフレームの読み込み
            _, self.frame = self.camera.read()

            # 顔検出
            (self.x, self.y, self.w, self.h) = self.detector()
            if self.x is not None:
                # 前処理
                image = self.preprocess()

                # 予測
                ret = self.predict()
        
                # 後処理
                ret = self.postprocess()

            # recordフラグが立ってたら結果を動画で書き出す(MTCNNだと上手くいかないかもぉ、、、)
            if self.record:
                self.writer.write(self.frame)

            # 表示
            cv2.imshow('img', self.frame)
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mtcnn', action='store_true', help='Use MTCNN for face detection')
    parser.add_argument('-r', '--record', action='store_true', help='Record')
    args = parser.parse_args()

    fd = MaskDetect(args)
    fd.run()

