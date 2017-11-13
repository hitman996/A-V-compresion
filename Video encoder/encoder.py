import numpy as np
import cv2
import VCFunctions
import scipy.signal as sp
import pickle


f = open('comp.txt', 'w')
fullf = open('full.txt', 'w')
original = open('original.txt', 'w')

N=2

factor = 4
blocksize = 20
lpF = np.ones((N, N))/N
pyF = sp.convolve2d(lpF, lpF)/N

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

r, c, d = frame.shape

mv  = np.zeros((r/8,c/8, 2))

Yprev = np.zeros((r,c))
print "Press Q to stop capturing video from the webcam"
for i in range(100):
    
    ret, frame = cap.read()

    pickle.dump(frame[:,:,0], original)
    pickle.dump(frame[:,:,1], original)
    pickle.dump(frame[:,:,2], original)

    if ret == True:
        reduced = frame.copy()
        
        frameYUV =VCFunctions.Rgb2Ycbcr(reduced)        

        if i%2 == 0:
            frameFilt = VCFunctions.filterFrame(frameYUV, pyF)
    
            Y = frameFilt[:,:,0]
            Cb = frameFilt[:,:,1]
            Cr = frameFilt[:,:,2]   
        
    #         Cbd = np.zeros((r,c))
            Cbd = Cb[::2,::2]
    #         Crd = np.zeros((r,c))
            Crd = Cr[::2,::2] 
            ##DCT
            ydct, udct, vdct = VCFunctions.dctFrame([Y, Cbd, Crd], factor)
            
            # Remove zeros
            ydctwoz = VCFunctions.removeZeros(ydct, factor)
            udctwoz = VCFunctions.removeZeros(udct, factor)
            vdctwoz = VCFunctions.removeZeros(vdct, factor)
            "pickle dump dct"
            pickle.dump(ydctwoz.astype(np.float16), f)
            pickle.dump(udctwoz.astype(np.float16), f)
            pickle.dump(vdctwoz.astype(np.float16), f)
            
        else:
            Y = frameYUV[:,:,0]
            mv = VCFunctions.motionvector(Y, Yprev, blocksize)
            "pickle dump motion vectors"
            #print( "inside encode else")
            pickle.dump(mv, f)
        
        pickle.dump(frame, fullf)
        Yprev = Y.copy()

        cv2.imshow('Y',Y / 255 )
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
print "Saved the captured video data and compressed video data into .txt files"
print "Uncompressed file size"
print fullf.tell()/(1024*1024) , "MBs"
print "Compressed file size"
print f.tell()/(1024*1024) , "MBs"

cap.release()
cv2.destroyAllWindows()  
f.close()  
fullf.close()