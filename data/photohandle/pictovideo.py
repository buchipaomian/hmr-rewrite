import cv2
import glob

def picvideo(fps,picpath,videopath):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(videopath,fourcc,fps,(224,224))
    imgs = glob.glob(picpath)
    for imgname in imgs:
        frame = cv2.imread(imgname)
        videoWriter.write(frame)
        print(imgname)
    videoWriter.release()
picvideo(24,"resultpic/*.png","resultvideo/result.mp4")