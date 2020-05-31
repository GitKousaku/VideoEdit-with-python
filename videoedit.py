import cv2
import argparse
import os.path
from PIL import ImageGrab
import ctypes
import time
import numpy as np
import msvcrt

def pick(fname,frame):
   print('pick',fname,frame)
   cap=cv2.VideoCapture(fname)
   if not cap.isOpened():
      print("Error {} not open".format(fname))
      return
   total_frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
   out=os.path.splitext(fname)
   outputfile=out[0]+"_s.png"
   if total_frame > frame:
      cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
      ret,frame=cap.read()
      if ret == 1:
         cv2.imwrite(outputfile,frame)

def play(fname,pos=0):
   cap=cv2.VideoCapture(fname)
   if not cap.isOpened():
      print("Error {} not open".format(fname))
      return
   w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   total_frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
   cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
   wkey=1
   pos=0
   still=False
   one_shot=False
   while (True):
      
      if pos >=0 and pos < total_frame and ( not still or one_shot):
         ret,frame=cap.read()
         one_shot=False

      msg="Frame:"+str(pos)+"/"+str(total_frame)
      frame2=frame
      cv2.putText(frame2,msg,(100,100),cv2.FONT_HERSHEY_PLAIN,4,(0,0,0)      ,12,cv2.LINE_AA)
      cv2.putText(frame2,msg,(100,100),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),2,cv2.LINE_AA)
      frame2=cv2.resize(frame2,dsize=(int(w/2),int(h/2)))
      cv2.imshow(fname,frame2)
      key=cv2.waitKey(wkey)
      
      if key == -1:
         if not still:
            pos=pos+1
      elif key == ord('9'):
         pos=pos+100
         cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
         one_shot=True
      elif key == ord('7'):
         pos=pos-100
         cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
         one_shot=True
      elif key == ord('6'):
         pos=pos+10
         cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
         one_shot=True
      elif key == ord('4'):
         pos=pos-10
         cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
         one_shot=True
      elif key == ord('3'):
         pos=pos+1
         cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
         one_shot=True
      elif key == ord('1'):
         pos=pos-1
         cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
         one_shot=True
      elif key == ord('q'):
         break
      elif key == 32 or ord('5'):
         still = not still

      elif key == ord('+'):
           wkey=wkey+5
      elif key == ord('-'):
           wkey=wkey-5
           if wkey < 1:
              wkey=1
           
      if pos < 0:
         pos=0
      if pos >= total_frame:
         pos=total_frame 



def cut(fname,pos0,pos1):
   out=os.path.splitext(fname)
   outputfile=out[0]+"_cut.mp4"
   print(outputfile)

   cap=cv2.VideoCapture(fname)
   if not cap.isOpened():
      print("Error {} not open".format(fname))
      return
   w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps=int(cap.get(cv2.CAP_PROP_FPS))
   total_frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
   
   fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
   video=cv2.VideoWriter(outputfile,fourcc,fps,(w,h),True)

   cap.set(cv2.CAP_PROP_POS_FRAMES,pos0)
   wkey=1
   pos=pos0
   while (True):
      
      if pos >=0 and pos < total_frame:
         ret,frame=cap.read()
         if ret == True:
            video.write(frame)
         
         msg="Frame:"+str(pos)+"/"+str(total_frame)
         cv2.putText(frame,msg,(100,100),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),5,cv2.LINE_AA)
         frame=cv2.resize(frame,dsize=(int(w/2),int(h/2)))
         cv2.imshow(fname,frame)
      
      key=cv2.waitKey(wkey)
      
      if key == ord('q'):
         break
      pos=pos+1
      if pos >= pos1:
         break
   cap.release()
   video.release()
   return

def still(fname,sec,fps):
   out=os.path.splitext(fname)
   outputfile=out[0]+"_still.mp4"
   print(outputfile)
   img=cv2.imread(fname,cv2.cv2.IMREAD_COLOR)
   if img is None:
      print("no image")
      return
   h, w = img.shape[:2]
   print(h,w)

   fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
   video=cv2.VideoWriter(outputfile,fourcc,fps,(w,h),True)
   
   pos_end=int(fps*sec)
   print(pos_end)
   wkey=1
   pos=0
   while (True):
      print(pos)
      if pos < pos_end:
         video.write(img)
         img0=img
         msg="Frame:"+str(pos)+"/"+str(pos_end)
         #cv2.putText(img0,msg,(100,100),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),5,cv2.LINE_AA)
         frame=cv2.resize(img0,dsize=(int(w/2),int(h/2)))
         cv2.imshow(fname,frame)
      else:
         break

      key=cv2.waitKey(wkey)
      
      if key == ord('q'):
         break
      pos=pos+1
  
   video.release()
   return

def clip(fname,box):
   out=os.path.splitext(fname)
   outputfile=out[0]+"_clip.mp4"
   print(outputfile)
   
   cap=cv2.VideoCapture(fname)
   if not cap.isOpened():
      print("Error {} not open".format(fname))
      return
   w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps=int(cap.get(cv2.CAP_PROP_FPS))
   size=(box[2]-box[0]+1,box[3]-box[1]+1)
   fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
   video=cv2.VideoWriter(outputfile,fourcc,fps,size,True)

   while True:
     ret,frame=cap.read()
     if not ret:
        break
     img=frame[box[1]:box[3],box[0]:box[2]] # frame[y0:y1,x0:x1])
     video.write(img)
     
     frame=cv2.resize(img,dsize=(int(w/2),int(h/2)))
     cv2.imshow(fname,frame)
     key=cv2.waitKey(1)
     if key == ord('q'):
        break
   video.release()
   cap.release()

def fpschange(fname,fs):
   out=os.path.splitext(fname)
   outputfile=out[0]+"_fs.mp4"
   print(outputfile)

   cap=cv2.VideoCapture(input_file)
   if not cap.isOpened():
      print("Error {} not open".format(fname))
      return
   w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps=cap.get(cv2.CAP_PROP_FPS)
   new_fps=int(fps*fs)
   print(fps,new_fps)
   fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
   video=cv2.VideoWriter(outputfile,fourcc,new_fps,(w,h),True)
   
   while True:
     ret,frame=cap.read()
     if not ret:
        break
     video.write(frame)
     frame=cv2.resize(frame,dsize=(int(w/2),int(h/2)))
     cv2.imshow(fname,frame)
     key=cv2.waitKey(1)
     if key==ord('q'):
        break
   video.release()
   cap.release()

def combine(fnames,w=0,h=0,fps=0):
   outputfile='combine.mp4'
   
   cap=cv2.VideoCapture(fnames[0])
   if not cap.isOpened():
      print("Error {} not open".format(fnames[0]))
      return

   if w==0:
      w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   if h==0:
      h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   if fps==0:
      fps=int(cap.get(cv2.CAP_PROP_FPS))
   cap.release()
   print(cap,fps,w,h)
   fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
   video=cv2.VideoWriter(outputfile,fourcc,fps,(w,h),True)

   count=0
   for fname in fnames:
     print(fname)
     cap=cv2.VideoCapture(fname)
     if not cap.isOpened():
        print("Error {} not open".format(fname))
        return

     c=0
     while True:
        ret,frame = cap.read()
        if not ret:
           cap.release()
           break
        print(fname,c,w,h)
        frame=cv2.resize(frame,(w,h))
        video.write(frame)
        c=c+1
        img=cv2.resize(frame,(int(w/2),int(h/2)))
        cv2.imshow("mon",img)
        key=cv2.waitKey(1)
        print(count)
        count +=1
   video.release()

def pictinpict(fname,pictname,box):
   out=os.path.splitext(fname)
   outputfile=out[0]+"_pip.mp4"
   print(outputfile)
   
   cap=cv2.VideoCapture(fname)
   if not cap.isOpened():
      print("Error {} not open".format(fname))
      return
   w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps=int(cap.get(cv2.CAP_PROP_FPS))

   fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
   video=cv2.VideoWriter(outputfile,fourcc,fps,(w,h),True)
   
   pict=cv2.imread(pictname)
   if pict is None:
      print("Error {} cant read",pictname)
      return
   h0=int(h*box[1])
   h1=int(h*box[3])
   w0=int(w*box[0])
   w1=int(w*box[2])
   pict=cv2.resize(pict,dsize=(int(w*(box[2]-box[0])),int(h*(box[3]-box[1]))))
   
   while True:
     ret,frame=cap.read()
     if not ret:
        break
     frame[h0:h1,w0:w1]=pict
     
     video.write(frame)
     img=cv2.resize(frame,(int(w/2),int(h/2)))
     cv2.imshow(fname,img)
     key=cv2.waitKey(1)

def DesktopCapture(wfname):
   user32=ctypes.windll.user32
   capSize=(user32.GetSystemMetrics(0),user32.GetSystemMetrics(1))
   print("Display Size",capSize)
   fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
   writer=cv2.VideoWriter(wfname,fourcc,10,capSize)
   #Get fps Setting
   count=0
   sTime=time.time()
   FirstFlag=True
   
   while (writer.isOpened()):
      f=ImageGrab.grab()
      f2=np.array(f,dtype=np.uint8)
      f2=cv2.cvtColor(f2,cv2.COLOR_BGR2RGB)
      writer.write(f2)
      #get fps 
      if FirstFlag==True:
         count += 1
         if time.time()-sTime > 1.0:
            writer.set(cv2.CAP_PROP_FPS,count)
            print("FPS {}".format(count))
            FirstFlag=False
      if msvcrt.kbhit():
         ch=msvcrt.getch()
         #print(ch)
         if ch == b'\x1b':  #ESC press
            break
   writer.release()

def PictInsert(fname,pictname,box):
   out=os.path.splitext(fname)
   outputfile=out[0]+"_ins.mp4"
   print(outputfile)
   
   cap=cv2.VideoCapture(fname)
   if not cap.isOpened():
      print("Error {} not open".format(fname))
      return
   w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps=int(cap.get(cv2.CAP_PROP_FPS))
   print(h,w)

   fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
   video=cv2.VideoWriter(outputfile,fourcc,fps,(w,h),True)
   
   pict=cv2.imread(pictname)
   if pict is None:
      print("Error {} cant read",pictname)
      return
   h0=int(h*box[1])
   h1=int(h*box[3])
   w0=int(w*box[0])
   w1=int(w*box[2])
   print(h0,h1,w0,w1)
   pict=cv2.resize(pict,dsize=(int(w*(box[2]-box[0])),int(h*(box[3]-box[1]))))
   pict2gray = cv2.cvtColor(pict,cv2.COLOR_BGR2GRAY)
   ret, mask = cv2.threshold(pict2gray, 50,255, cv2.THRESH_BINARY)
   mask_inv = cv2.bitwise_not(mask)
   
   while True:
     
     ret,frame=cap.read()
     if not ret:
        break
     roi=frame
     img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
     # Take only region of logo from logo image.
     img2_fg = cv2.bitwise_and(pict,pict,mask = mask)

     # Put logo in ROI and modify the main image
     dst = cv2.add(img1_bg,img2_fg)
     
     frame=dst
     video.write(frame)
     img=cv2.resize(frame,(int(w/2),int(h/2)))
     cv2.imshow(fname,img)
     key=cv2.waitKey(1)

   
parser=argparse.ArgumentParser(description="VideoEdit")
parser.add_argument('-i','--input',help='Input Video')
parser.add_argument('-c','--command',help='Command: pick->Pick Picture from Movie / play-> play and anal video / cut -> cut video x to y frame,still-> create vide from still',default='p')
parser.add_argument('-f','--frame',help='Pick Frame',type=int,default=10)
parser.add_argument('-pos','--position',help='Cut Position',type=int,nargs=2)
parser.add_argument('-sec','--second',help='Still Video Time (sec)',type=float)
parser.add_argument('-fps','--fps',help='FPS',type=int,default=30)
parser.add_argument('-a','--area',help='Clip Area x0,y0,x1,y1',type=int,nargs=4,default=(520,100,1650,900))
parser.add_argument('-fs','--fpsscale',help='FPS Change scale ratio',type=float,default=6.0)
parser.add_argument('-files','--files',help='File Names',nargs='*')
parser.add_argument('-pict','--pictname',help='Insert Pict name')
parser.add_argument('-pa','--pictarea',help='Pict in Area x0,y0,x1,y1 ratio',nargs=4,default=(0.4,0.0,0.95,0.6))
parser.add_argument('-wf','--wfname',help='Write File Name',default='sc.mp4')
args=parser.parse_args()

input_file=args.input
frame=args.frame
command=args.command

if command == 'pick':
   pick(input_file,frame)
if command == 'play':
   play(input_file)
if command == 'cut':
   pos0=args.position[0]
   pos1=args.position[1]
   cut(input_file,pos0,pos1)
if command == 'still':
   sec=args.second
   fps=args.fps
   still(input_file,sec,fps)
if command == 'clip':
   box=(args.area[0],args.area[1],args.area[2],args.area[3])
   clip(input_file,box)
if command == 'fpschange':
   fs=args.fpsscale
   fpschange(input_file,fs)
if command == 'combine':
   fnames=args.files
   combine(fnames)
if command == 'pictinpict':
   pictname=args.pictname
   box=(args.pictarea[0],args.pictarea[1],args.pictarea[2],args.pictarea[3])
   pictinpict(input_file,pictname,box)
if command == 'desktopcapture':
   wfname=args.wfname
   DesktopCapture(wfname)
if command == 'pictinsert':
   pictname=args.pictname
   box=(0.0,0.0,1.0,1.0)
   PictInsert(input_file,pictname,box)

