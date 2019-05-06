import cv2
def getFrame(videoPath,svPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame =0
    while True:
        if cap.grab():
            flag,frame = cap.retrieve()
            if not flag:
                continue
            else:
                numFrame+=1
                newPath=svPath+str(numFrame)+".png"
                cv2.imencode('.png',frame)[1].tofile(newPath)
        if cv2.waitKey(10)==27:
            print("finished")
            break
getFrame("sourcevideo/test1.mp4","sourcepic/test1")