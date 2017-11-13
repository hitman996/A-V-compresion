import numpy as np
import scipy.signal as sp
import scipy.fftpack as sft
#import functions
import cv2

N = 255

RGB = np.matrix([[0.299,     0.587,     0.114],
                 [-0.16864, -0.33107,   0.49970],
                 [0.499813, -0.418531, -0.081282]])
YCbCr = RGB.I

framevectors = np.zeros((480, 640, 3))

def applyDCT(frame, factor):
    """
    """
    '''Extract frame row and column size'''
    r,c = frame.shape
    '''Make an array of 8 '1's '''
    Mr = np.ones(8) 
    '''Let only factor number of 1's and set the rest to zero'''
    Mr[factor:r] = np.zeros(8 - factor)
#     Mr[int(3):r]=np.zeros(5)
    '''Block size is a square matrix 8x8 so Mc is same as Mr'''
    Mc = Mr
#     frame=np.reshape(frame[:,:,1],(-1,8), order='C')
    '''reshape the frame by columns of 8 and corresponding number of rows.
    This will help in Applying DCT to each row which has 8 values now'''
    
    frame=np.reshape(frame,(-1,8), order='C')
    
    X=sft.dct(frame/255.0,axis=1,norm='ortho')
    #apply row filter to each row by matrix multiplication with Mr as a diagonal matrix from the right:
    X=np.dot(X,np.diag(Mr))
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of height 8 by using transposition .T:
    X=np.reshape(X.T,(-1,8), order='C')
    X=sft.dct(X,axis=1,norm='ortho')
    #apply column filter to each row by matrix multiplication with Mc as a diagonal matrix from the right:
    X=np.dot(X,np.diag(Mc))
    #shape it back to original shape:
    X=(np.reshape(X,(-1,r), order='C')).T
    #Set to zero the 7/8 highest spacial frequencies in each direction:
    #X=X*M    
    return X


def DCTFrame(frame, factor):
    '''
    
    :param frame:
    :param factor:
    '''

    dctFrame = np.zeros(frame.shape)
    for i in range(frame.shape[2]):
        dctFrame[:,:,i] = applyDCT(frame[:,:,i], factor)
    return dctFrame
    
    
def dctFrame(layers, factor):
    '''
    
    :param frame:
    :param factor:
    '''
    ydct = applyDCT(layers[0], factor)
    cbdct = applyDCT(layers[1], factor)
    crdct = applyDCT(layers[2], factor)
    
    return ydct, cbdct, crdct

def applyIDCT(frame):
    """
    """
    r,c= frame.shape
    X=np.reshape(frame,(-1,8), order='C')
    X=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of hight 8 (columns: order='F' convention):
    X=np.reshape(X.T,(-1,8), order='C')
    x=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    x=(np.reshape(x,(-1,r), order='C')).T
    return x

def IDCTFrame(frame):
    """
    """
    idctFrame = np.zeros(frame.shape)
    for i in range(frame.shape[2]):
        idctFrame[:,:,i] = applyIDCT(frame[:,:,i])
    return idctFrame

def idctFrame(layers):
    y = applyIDCT(layers[0])
    cbd = applyIDCT(layers[1])
    crd = applyIDCT(layers[2])
    
    return y, cbd, crd


def app420(comp):
    """
    """
    r, c = comp.shape 
    #apply on chroma components with reduced size from Task 2
#     f = np.zeros((r*2,c*2))
#     print comp.shape
    f = np.zeros((r,c))
    #apply on chroma components with reduced size from Task 2
    #f[0::2, 0::2] = comp
    f[0::2, 0::2] = comp[::2,::2]
    f[1::2, ] = f[0::2, ]
    f[:, 1::2] = f[:, 0::2]
    return f

def chomaSubSamp(y, cb, cr):
    """
    """
    r, c = y.shape
    # print frame.shape
    frame = np.zeros((r, c, 3))
    frame[:, :, 0] = y  #astype
    frame[:, :, 1] = app420(cb) #astype
    frame[:, :, 2] = app420(cr) #astype
    return frame

def filterFrame(frame, kernel):
    """
    """
    frame[:, :, 1] = sp.convolve2d(frame[:, :, 1], kernel, mode='same')
    frame[:, :, 2] = sp.convolve2d(frame[:, :, 2], kernel, mode='same')
    return frame

def fillZeros(frame,flag):
    """
    """
    if flag == 1:
        r, c = frame.shape
        factor = r/60 #or c/80
        incRC = 480./r
        rem = 8-factor
    else:
        r, c = frame.shape
        factor = r/30 #or c/80
        incRC = 240./r
        rem = 8-factor
    for i in range(factor, int(r*incRC), 8):
        frame = np.insert(frame, [i], np.zeros(rem).reshape(rem,1), axis=0)
    for i in range(factor, int(c*incRC), 8):
        frame = np.insert(frame,[i], np.zeros(rem), axis=1)
    return frame

def fillZeros01(frame, z):
    """
    """
    r, c = frame.shape
    incRC = 480./r
    rem = 8-z
    for i in range(z, int(r*incRC), 8):
        frame = np.insert(frame, [i], np.zeros(rem).reshape(rem,1), axis=0)
    for i in range(z, int(c*incRC), 8):
        frame = np.insert(frame,[i], np.zeros(rem), axis=1)
    return frame

def DCTWithZeros(frame):
    """
    """
#     dim = 480, 640, 3
    f = np.zeros((480,640,3))
    for i in range(f.shape[2]):
        f[:,:,i] = fillZeros(frame[:,:,i])
    return f



def removeZeros(x, f):
    """
    """
    r,c = x.shape
    rem = 8 - f
    for k in range(f, c+1, f):
        x = np.delete(x, np.arange(k, k+rem),  axis=1)    
    for l in range(f, r+1, f):
        x = np.delete(x, np.arange(l, l+rem).T, axis=0)
    return x

def DCTRemoveZeros(frame, f):
    """
    """
    rframe = np.zeros((60*f, 80*f, frame.shape[2]))
    for i in range(frame.shape[2]):
        rframe[:,:,i] = removeZeros(frame[:,:,i], f)
    return rframe#.astype('int8')

    
def upsample(frame, N):
    """
    """
    r, c = frame.shape
    # print frame.shape
    fu = np.zeros((r*N,c*N))
    fu[::N, ::N] = frame
    return fu

def Rgb2Ycbcr(frame):
    """
    """
    xframe=np.zeros(frame.shape)
    for i in range(frame.shape[0]):
        xframe[i] = np.dot(RGB,frame[i].T).T
    return xframe

def Ycbcr2Rgb(frame):
    """
    """
    xframe=np.zeros(frame.shape)
    for i in range(frame.shape[0]):
        xframe[i] = np.dot(YCbCr/255., frame[i].T).T#/255.
    return xframe

def motionvector(Y, Yprev, blocksize):
    """
    """
    mv = np.zeros((60,80,2))
    block = np.array([8, 8])
    for yblock in range(blocksize):
    # print("yblock=",yblock)
        block[0] = yblock * 8+8;#150
        for xblock in range(blocksize):
            # print("xblock=",xblock)
            block[1] = xblock * 8+8;#200
            # print("block= ", block)
            # current block:
            Yc = Y[block[0]:block[0] + 8, block[1]:block[1] + 8]
            # print("Yc= ",Yc)
            # previous block:
            # Yp=Yprev[block[0]-4 : block[0]+12 ,block[1]-4 : block[1]+12]
            # print("Yp= ",Yp)
            # correlate both to find motion vector
            # print("Yp=",Yp)
            # print(Yc.shape)
            # Some high value for MAE for initialization:
            bestmae = 100.0;
            # For loops for the motion vector, full search at +-8 integer pixels:
            # print "reset counter"
            ctr = 0;
            for ymv in range(-8, 8):
                for xmv in range(-8, 8):
                    diff = Yc - Yprev[block[0] + ymv: block[0] + ymv + 8, block[1] + xmv: block[1] + xmv + 8];
                    mae = sum(sum(np.abs(diff))) / 64;
                    ctr = ctr + 1;
    
                    # print "ctr: ", ctr
    
                    if mae <= bestmae:
                        # print "mae: ", mae
                        bestmae = mae;
                        mv[yblock, xblock, 0] = ymv
                        mv[yblock, xblock, 1] = xmv
    
                    if mae < 1:
                        ctr = 0
                        # print "counter break"
                        break
     
            if bestmae > 10:
                # print "Motion Detected . . . . . . . . . . . . "
                cv2.line(framevectors, (block[1], block[0]), (block[1] + mv[yblock, yblock, 1].astype(int), block[0] + mv[yblock, yblock, 0].astype(int)),
                         (1.0, 1.0, 1.0));
            elif bestmae < 10:
                # print "bestmae", bestmae
                # print "No motion detected  . . . . . . . . . . "
                cv2.line(framevectors, (block[1], block[0]), (block[1], block[0]), (1.0, 1.0, 1.0))
    return mv



def quantizer(x, bit):
    '''
    :param x:
    :param N:
    '''
    q = (np.max(x) - np.min(x))/(2.**bit-1)
    return q, np.round(x/q)

if __name__ == '__main__':
    
    x = np.array([130.25, 222.3, -301.2, 45.6])
    q,xq = quantizer(x, N)
    print xq
    print q*xq


   
# def rgb2ycbcr(frame):
#     R = frame[:,:,2]
#     G = frame[:,:,1]
#     B = frame[:,:,0]
# 
#     #red = frame.copy()
#     Y = (0.299*R + 0.587*G + 0.114*B)
#     Cb = (-0.16864*R - 0.33107*G + 0.49970*B)
#     Cr = (0.499813*R - 0.418531*G - 0.081282*B)
#     return Y, Cb, Cr
# 
# def ycbcr2rgb(framedec):
#     Y = (framedec[:,:,0])/255.
#     Cb = (framedec[:,:,1])/255.
#     Cr = (framedec[:,:,2])/255.
# 
#     '''Compute RGB components'''
#     R = (0.771996*Y -0.404257*Cb + 1.4025*Cr)
#     G = (1.11613*Y - 0.138425*Cb - 0.7144*Cr)
#     B = (1.0*Y + 1.7731*Cb)
#     '''Display RGB Components'''
#     decfile = np.zeros(framedec.shape)
#     decfile[:,:,2] = R
#     decfile[:,:,1] = G
#     decfile[:,:,0] = B
#     return decfile

