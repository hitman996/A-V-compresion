import cv2
import pickle
import VCFunctions
import numpy as np
import scipy.signal as sp
r,c,d = 480,640,3

g = open('comp.txt', 'r')
ctr = 0

N = 2
lpF = np.ones((N, N))/N
pyF = sp.convolve2d(lpF, lpF)/N

Ybuff = np.zeros((r,c))

try:
    while True:
    
        #print ctr
        ###################Decoder######################################
        if ctr % 2 == 0:
    #         print "Entered decoder even"
            "pickle load"
            yz = pickle.load(g)
            uz = pickle.load(g)
            vz = pickle.load(g)
    #         "Fill zeros and IDCT"
            # Add Zeros
            #print yz.shape
            #print vz.shape
            ydctdec = VCFunctions.fillZeros(yz, 1)
            udctdec = VCFunctions.fillZeros(uz, 0)
            vdctdec = VCFunctions.fillZeros(vz, 0)
    #         print ydctdec.shape
    #         print udctdec.shape
            ydec, udec, vdec = VCFunctions.idctFrame([ydctdec, udctdec, vdctdec])
            
            # upsample
        
            udecu = np.zeros((r,c))
            vdecu = np.zeros((r,c))
            
            udecu[::2, ::2] = udec
            vdecu[::2, ::2] = vdec
            
    #         print vdecu.shape
       
            framedec = np.zeros((r,c,d))
            framedec[:,:,0] = ydec
            framedec[:,:,1] = udecu
            framedec[:,:,2] = vdecu
            
            framedecFilt = VCFunctions.filterFrame(framedec, lpF)
            
            decRGB = VCFunctions.Ycbcr2Rgb(framedecFilt)
            #print "even"
            #decRGB = decRGB[:240, :320]
            cv2.imshow("Decoded", decRGB*255) 
        
        else:
            "pickle load"
            mv = pickle.load(g)
    #         print "Entered decoder odd"
            Yc = Ybuff.copy()
    #             print mv[:,:,0]
            for i in range(1,mv.shape[0]):
                for j in range(1,mv.shape[1]):
                                    
                    if mv[i,j,0] != 0 or mv[i,j,1] != 0:
    #                         print( "mv ",mv[i,j])
                        new_pos = mv[i,j,:].astype(np.int) + np.array([i*8+8, j*8+8])
    #                         print( "new pos", new_pos)
                        old_pos = np.array([i*8+8, j*8+8])
    #                         print( "old_pos", old_pos)
                        
                        if new_pos[0] > 4 and new_pos[1] > 4:
                            Yc[old_pos[0]-4: old_pos[0]+4, 
                               old_pos[1]-4: old_pos[1]+4] = Ybuff[new_pos[0]-4:new_pos[0]+4,
                                                                new_pos[1]-4:new_pos[1]+4]
    #         print Yc.shape
            framedec[:,:,0] = Yc
            framedec[:,:,1] = framedecFilt[:,:,1]
            framedec[:,:,2] = framedecFilt[:,:,2]
            
    #             framedec = VCFunctions.filterFrame(framedec, pyF)
            decRGB = VCFunctions.Ycbcr2Rgb(framedec)
            #print "odd"
#            decRGB = decRGB[:240, :320]
            cv2.imshow('Decoded', decRGB*255)
                        
        Ybuff = framedecFilt[:,:,0].copy()
        ctr += 1
        
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
except(EOFError):
    pass
cv2.destroyAllWindows()
g.close()
        
        
        
