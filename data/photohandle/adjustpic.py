from PIL import Image
import os.path
import glob
def picresult(picpath,resultpath):
    for pngfile in glob.glob(picpath):
        img = Image.open(pngfile)
        try:
            new_img = img.resize((224,224),Image.ANTIALIAS)
            new_img.save(os.path.join(resultpath,os.path.basename(pngfile)))
        except Exception as e:
            print(e)  

picresult(r"sourcepic/*.png",r"resultpic/")