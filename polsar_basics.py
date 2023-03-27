## Import all required libraries
import numpy as np
## Used to read images in the mat format
import scipy.io as sio
from scipy import misc
## Used to equalize histograms in images
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
#
import polsar_total_loglikelihood as ptl
import polsar_plot as pplt
#
#
import matplotlib.pyplot as plt#
## Read an image in the mat format
def le_imagem(img_geral, opcao, opcao_image):
    if opcao == 4:
        # Read a file *.txt
        data_aux1 = np.loadtxt(img_geral[0], dtype=float)
        data_aux2 = np.loadtxt(img_geral[1], dtype=float)
        data_aux3 = np.loadtxt(img_geral[2], dtype=float)
        # Read a file *.bmp
        #data_aux1 = cv2.imread(img_geral[0], 0)
        #data_aux2 = cv2.imread(img_geral[1], 0)
        #data_aux3 = cv2.imread(img_geral[2], 0)
        nrows = data_aux1.shape[0]
        ncols = data_aux1.shape[1]
        nc = 9
        img_dat = np.zeros([nrows, ncols, nc])
        # Adjust to intensity channel
        img_dat[:, :, 0] = np.square(data_aux1[:, :])
        img_dat[:, :, 1] = np.square(data_aux2[:, :])
        img_dat[:, :, 2] = np.square(data_aux3[:, :])
        opcao_image = 0
    elif opcao == 5:
        if opcao_image == 1:
            img=sio.loadmat(img_geral)
            mat_aux = img[list(img.keys())[3]]
            nrows = mat_aux.shape[2]
            ncols = mat_aux.shape[3]
            nc = mat_aux.shape[0] * mat_aux.shape[1]
            img_dat = np.zeros([nrows, ncols, nc])
            img_dat[:,:, 0] = np.real(mat_aux[0][0][:][:])
            img_dat[:,:, 1] = np.real(mat_aux[1][1][:][:])
            img_dat[:,:, 2] = np.real(mat_aux[2][2][:][:])
        else:
            img=sio.loadmat(img_geral)
            mat_aux = img[list(img.keys())[4]]
            nrows = mat_aux.shape[2]
            ncols = mat_aux.shape[3]
            nc = mat_aux.shape[0] * mat_aux.shape[1]
            img_dat = np.zeros([nrows, ncols, nc])
            img_dat[:,:, 0] = np.real(mat_aux[0][0][:][:])
            img_dat[:,:, 1] = np.real(mat_aux[1][1][:][:])
            img_dat[:,:, 2] = np.real(mat_aux[2][2][:][:])
    elif opcao == 6:
        if opcao_image == 1:
            img=sio.loadmat(img_geral)
            mat_aux = img[list(img.keys())[5]]
            nrows = mat_aux.shape[2]
            ncols = mat_aux.shape[3]
            nc = mat_aux.shape[0] * mat_aux.shape[1]
            img_dat = np.zeros([nrows, ncols, nc])
            img_dat[:,:, 0] = np.real(mat_aux[0][0][:][:])
            img_dat[:,:, 1] = np.real(mat_aux[1][1][:][:])
            img_dat[:,:, 2] = np.real(mat_aux[2][2][:][:])
        else:
            img=sio.loadmat(img_geral)
            mat_aux = img[list(img.keys())[6]]
            nrows = mat_aux.shape[2]
            ncols = mat_aux.shape[3]
            nc = mat_aux.shape[0] * mat_aux.shape[1]
            img_dat = np.zeros([nrows, ncols, nc])
            img_dat[:,:, 0] = np.real(mat_aux[0][0][:][:])
            img_dat[:,:, 1] = np.real(mat_aux[1][1][:][:])
            img_dat[:,:, 2] = np.real(mat_aux[2][2][:][:])
    else:
        img=sio.loadmat(img_geral)
        img_dat=img['S']
        img_dat=np.squeeze(img_dat)
        img_shp=img_dat.shape
        ncols=img_shp[1]
        nrows=img_shp[0]
        nc=img_shp[len(img_shp)-1]
        opcao_image = 0
    return img_dat, nrows, ncols, nc
#
## Uses the Pauli decomposition
def show_Pauli(data, index, control):
    Ihh = np.real(data[:,:,0])
    Ihv = np.real(data[:,:,1])
    Ivv = np.real(data[:,:,2])
    Ihh=np.sqrt(np.abs(Ihh))
    Ihv=np.sqrt(np.abs(Ihv))/np.sqrt(2)
    Ivv=np.sqrt(np.abs(Ivv))
    R = np.abs(Ihh - Ivv)
    G = (2*Ihv)
    B =  np.abs(Ihh + Ivv)
    R = exposure.equalize_hist(R)
    G = exposure.equalize_hist(G)
    B = exposure.equalize_hist(B)
    Pauli_Image = np.dstack((R,G,B))
    return Pauli_Image
#
## Uses the Pauli decomposition
def show_Pauli_with_smooth_ramp(data, index, control):
    Ihh = np.real(data[:,:,0])
    Ihv = np.real(data[:,:,1])
    Ivv = np.real(data[:,:,2])
    #Ihh=np.sqrt(np.abs(Ihh))
    #Ihv=np.sqrt(np.abs(Ihv))/np.sqrt(2)
    #Ivv=np.sqrt(np.abs(Ivv))
    R = np.abs(Ihh - Ivv)
    G = (2*Ihv)
    B =  np.abs(Ihh + Ivv)
    R = exposure.equalize_hist(R)
    G = exposure.equalize_hist(G)
    B = exposure.equalize_hist(B)
    Pauli_Image = np.dstack((R,G,B))
    return Pauli_Image
#
## The Bresenham algorithm
## Finds out in what octant the radius is located and translate it to the first octant in order to compute the pixels in the
## radius. It translates the Bresenham line back to its original octant
def bresenham(x0, y0, xf, yf):
    x=xf-x0
    y=yf-y0
    m=10000
    ## avoids division by zero
    if abs(x) > 0.01:
        m=y*1.0/x
    ## If m < 0 than the line is in the 2nd or 4th quadrant
    ## print(x,y, m)
    if m<0:
        ## If |m| <= 1 than the line is in the 4th or in the 8th octant
        if abs(m)<= 1:
            ## If x > 0 than the line is in the 8th octant
            if x>0:
                y=y*-1
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                yp=list(np.asarray(yp)*-1)
            ## otherwise the line is in the 4th octant
            else:
                x=x*-1
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp=list(np.asarray(xp)*-1)
        ## otherwise the line is in the 3rd or 7th octant
        else:
            ## If y > 0 than the line is in the 3rd octant
            if y>0:
                x=x*-1
                x,y = y,x
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp,yp = yp,xp
                xp=list(np.asarray(xp)*-1)
            ## otherwise the line is in the 7th octant
            else:
                y=y*-1
                x,y = y,x
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp,yp = yp,xp
                yp=list(np.asarray(yp)*-1)
    ## otherwise the line is in the 1st or 3rd quadrant
    else:
        ## If |m| <= 1 than the line is in the 1st or 5th octant
        if abs(m)<= 1:
            ## if x > 0 than the line is in the 1st octant
            if x>0:
                ##print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
            ## otherwise the line is in the 5th octant
            else:
                x=x*-1
                y=y*-1
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp=list(np.asarray(xp)*-1)
                yp=list(np.asarray(yp)*-1)
        ## otherwise the line is in the 2nd or 6th octant
        else:
            ## If y > 0 than the line is in the 2nd octant
            if y>0:
                x,y = y,x
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp,yp = yp,xp
            ## otherwise the line is in the 6th octant
            else:
                y=y*-1
                x=x*-1
                x,y = y,x
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp,yp = yp,xp
                xp=list(np.asarray(xp)*-1)
                yp=list(np.asarray(yp)*-1)
    xp= list(np.asarray(xp) + x0)
    yp= list(np.asarray(yp) + y0)
    return xp, yp

## Computes the Bresenham line in the first octant. The implementation is based on the article:
## https://www.tutorialandexample.com/bresenhams-line-drawing-algorithm/

def bresenham_FirstOctante(xf, yf):
    x=int(xf)
    y=int(yf)
    xp=[]
    yp=[]
    xp.append(0)
    yp.append(0)
    x_temp=0
    y_temp=0
    pk=2*y-x
    for i in range(x-1):
        ## print(pk)
        if pk<0:
            pk=pk+2*y
            x_temp=x_temp+1
            y_temp=y_temp
        else:
            pk=pk+2*y-2*x
            x_temp=x_temp+1
            y_temp=y_temp+1
        xp.append(int(x_temp))
        yp.append(int(y_temp))
    xp.append(x)
    yp.append(y)
    return xp, yp


## Define the radius
def define_radiais(r, num_r, dx, dy, nrows, ncols, start, end):
    x0 = ncols / 2 - dx
    y0 = nrows / 2 - dy
    t = np.linspace(start, end, num_r, endpoint=False)
    x = x0 + r * np.cos(t)
    y = y0 + r * np.sin(t)
    xr= np.round(x)
    yr= np.round(y)
    return x0, y0, xr, yr

## Check if the extreme points of each radius are inside the image or not.
def test_XY(XC, YC, j, tam_Y, tam_X):
    if XC[j]<0:
        X=0
    elif XC[j]>=tam_X:
        X=tam_X-1
    else:
        X=XC[j]
    if YC[j]<0:
        Y=0
    elif YC[j]>=tam_Y:
        Y=tam_Y-1
    else:
        Y=YC[j]
    return int(X), int(Y)

## Draw the radius in the image and determine the pixels where
## the image will be sampled using the Bresenham algorithm
def desenha_raios(ncols, nrows, nc, RAIO, NUM_RAIOS, img, PI, x0, y0, xr, yr):
    ## Cria vetors e matrizes de apoio
    IT = np.zeros([nrows, ncols])
    const =  5 * np.max(np.max(np.max(PI)))
    MXC = np.zeros([NUM_RAIOS, RAIO])
    MYC = np.zeros([NUM_RAIOS, RAIO])
    MY  = np.zeros([NUM_RAIOS, RAIO, nc])
    for i in range(NUM_RAIOS):
        XC, YC = bresenham(x0, y0, xr[i], yr[i])
        for canal in range(nc):
            Iaux = img[:, :, canal]
            dim = len(XC)
            for j in range(dim-1):
                X,Y = test_XY(XC, YC, j, nrows, ncols)
                MXC[i][j] = X
                MYC[i][j] = Y
                MY[i][j][canal] = Iaux[Y][X]
                IT[Y][X] = const
                PI[Y][X] = const
    return MXC, MYC, MY, IT, PI


## Check the order of the line coordinates in order to call the Bresenham algorithm.
## The Bresenham algorithm assumes that x0 < x1
def verifica_coords(x0, y0, x1, y1):
    flip=0
    if x0>x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        flip=1
    return x0, y0, x1, y1, flip

## Determine the ground truth lines in teh image - it is always a straight line
## The lines are genrated always from the point with the smaller x coordinate to the  point with the larger x coordinate
## Consider the example:
## given the points (10, 15) and (20, 25) generates a ground truth line from (10, 15) to (20, 25)
## given the points (20, 25) and (10, 15) generates a ground truth line from (10, 15) to (20, 25)
## Lines is a list with 4 biary values that indicates what borders of the quadrilateral should be computed
## For instance, if lines[0] = 1 finds the ground truth line that connects the points x1, y1 and x2, y2,
## if lines[1] = 1 finds the ground truth line that connects the points x2, y2 and x3, y3.
## If lines[i]=0 a no ground truth line is computed.

def get_gt_lines(gt_coords, lines):
    '''
    gt_coords:  a list of points coordinates using the xi, yi order from the ROI area
    lines: a vetor indicating the ground truth lines to be computed
    '''
    gt_lines=[]
    for l in range(len(lines)):
        if lines[l]==1:
            if l<3:
                x0, y0, x1, y1, flip=verifica_coords(gt_coords[l][0], gt_coords[l][1], gt_coords[l+1][0], gt_coords[l+1][1])
            else:
                x0, y0, x1, y1, flip=verifica_coords(gt_coords[l][0], gt_coords[l][1], gt_coords[0][0], gt_coords[0][1])
            xp, yp=bresenham(x0, y0, x1, y1)
            if flip==1:
                xp.reverse()
                yp.reverse()
            gt_lines.append([xp,yp])
    return gt_lines
#
## This function computes the indexes from a list where the condition is true
## call: get_indexes(condicao) - example: get_indexes(x>0)
def get_indexes(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'get_indexes' method can only be applied to iterables.{}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)
#
# Ground Truth
# San Francisco ROI 01
#
def GT_sf_roi_01(GT, PI, nrows, ncols, img_rt):
    #
    GT[354,247] = 1
    GT[354,245] = 1
    GT[354,243] = 1
    GT[354,241] = 1
    GT[355,239] = 1
    GT[355,237] = 1
    GT[355,235] = 1
    GT[355,233] = 1
    GT[356,231] = 1
    GT[356,229] = 1
    GT[356,227] = 1
    GT[356,225] = 1
    GT[356,223] = 1
    GT[357,221] = 1
    GT[357,219] = 1
    GT[357,217] = 1
    GT[358,215] = 1
    GT[358,213] = 1
    GT[358,211] = 1
    GT[358,209] = 1
    GT[359,207] = 1
    GT[360,205] = 1
    GT[360,203] = 1
    GT[360,201] = 1
    #
    GT[360,199] = 1
    GT[360,197] = 1
    GT[360,195] = 1
    GT[360,193] = 1
    GT[361,191] = 1
    GT[361,189] = 1
    GT[362,188] = 1
    GT[363,187] = 1
    GT[364,186] = 1
    GT[365,185] = 1
    #
    GT[367,185] = 1
    GT[370,185] = 1
    GT[373,185] = 1
    GT[375,185] = 1
    GT[378,185] = 1
    GT[380,185] = 1
    GT[383,186] = 1
    GT[385,186] = 1
    GT[388,186] = 1
    GT[390,186] = 1
    GT[393,186] = 1
    GT[395,187] = 1
    GT[398,187] = 1
    GT[400,187] = 1
    GT[403,187] = 1
    GT[405,187] = 1
    GT[408,188] = 1
    GT[410,188] = 1
    GT[413,188] = 1
    GT[415,188] = 1
    GT[418,188] = 1
    GT[420,188] = 1
    #
    PIA=PI.copy()
    plt.figure(figsize=(20*img_rt, 20))
    for j in range(nrows):
        for i in range(ncols):
            if (GT[j, i] != 0):
                #ik = np.int(evidence[k, banda])
                #ia = np.int(MXC[k, ik])
                #ja = np.int(MYC[k, ik])
                plt.plot(i, j, marker='o', color="darkorange")
    plt.imshow(PIA)
    plt.show()
    #
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(GT[300:450,170:260])
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(PI[300:450,170:260])
 #imgplot.set_clim(0.0, 0.7)
#ax.set_title('After')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    fig = plt.figure()
    #plt.imshow(PI[300:450,170:200])
    plt.imshow(PI)
def GT_flev_roi_01(GT, PI, nrows, ncols, img_rt):
    #
    GT[287,307] = 1
    GT[289,307] = 1
    GT[291,307] = 1
    GT[293,307] = 1
    GT[295,307] = 1
    GT[297,307] = 1
    GT[299,307] = 1
    GT[301,307] = 1
    GT[303,307] = 1
    GT[305,307] = 1
    GT[307,307] = 1
    GT[309,308] = 1
    GT[311,308] = 1
    GT[313,308] = 1
    GT[315,308] = 1
    GT[317,308] = 1
    GT[309,308] = 1
    GT[311,308] = 1
    GT[313,308] = 1
    GT[315,308] = 1
    GT[317,308] = 1
    GT[319,308] = 1
    GT[321,308] = 1
    GT[323,308] = 1
    GT[325,308] = 1
    GT[327,308] = 1
    #
    GT[283,307] = 1
    GT[283,305] = 1
    GT[283,303] = 1
    GT[283,301] = 1
    GT[283,299] = 1
    GT[283,297] = 1
    GT[283,295] = 1
    GT[283,293] = 1
    GT[283,291] = 1
    GT[284,289] = 1
    GT[284,287] = 1
    GT[284,285] = 1
    GT[284,283] = 1
    GT[284,281] = 1
    GT[284,279] = 1
    GT[284,277] = 1
    GT[284,275] = 1
    GT[284,273] = 1
    GT[284,271] = 1
    GT[284,269] = 1
    GT[284,267] = 1
    GT[284,265] = 1
    GT[284,263] = 1
    GT[284,261] = 1
    GT[284,259] = 1
    GT[284,257] = 1
    GT[284,255] = 1
    GT[284,253] = 1
    GT[284,251] = 1
    GT[284,249] = 1
    GT[285,247] = 1
    GT[285,245] = 1
    GT[285,243] = 1
    GT[285,241] = 1
    GT[285,239] = 1
    GT[285,237] = 1
    GT[285,235] = 1
    GT[285,233] = 1
    GT[285,231] = 1
    GT[285,229] = 1
    GT[285,227] = 1
    GT[285,225] = 1
    GT[285,223] = 1
    GT[285,221] = 1
    GT[285,219] = 1
    GT[285,217] = 1
    GT[285,215] = 1
    GT[285,213] = 1
    GT[285,211] = 1
    GT[285,209] = 1
    GT[285,207] = 1
    GT[285,205] = 1
    GT[285,203] = 1
    GT[285,201] = 1
    GT[285,199] = 1
    GT[285,197] = 1
    GT[285,195] = 1
    GT[285,193] = 1
    GT[285,191] = 1
    GT[285,189] = 1
    GT[285,187] = 1
    GT[285,185] = 1
    GT[285,183] = 1
    GT[285,181] = 1
    GT[285,179] = 1
    GT[285,177] = 1
    GT[287,177] = 1
    GT[289,177] = 1
    GT[291,177] = 1
    GT[293,177] = 1
    GT[295,177] = 1
    GT[297,177] = 1
    GT[299,177] = 1
    GT[301,177] = 1
    GT[303,177] = 1
    GT[305,177] = 1
    GT[307,177] = 1
    GT[309,178] = 1
    GT[311,178] = 1
    GT[313,178] = 1
    GT[315,178] = 1
    GT[317,178] = 1
    GT[309,178] = 1
    GT[311,178] = 1
    GT[313,178] = 1
    GT[315,178] = 1
    GT[317,178] = 1
    GT[319,178] = 1
    GT[321,178] = 1
    GT[323,178] = 1
    GT[325,178] = 1
    GT[327,178] = 1
    #
    GT[327,307] = 1
    GT[327,305] = 1
    GT[327,303] = 1
    GT[327,301] = 1
    GT[327,299] = 1
    GT[327,297] = 1
    GT[327,295] = 1
    GT[327,293] = 1
    GT[327,291] = 1
    GT[328,289] = 1
    GT[328,287] = 1
    GT[328,285] = 1
    GT[328,283] = 1
    GT[328,281] = 1
    GT[328,279] = 1
    GT[328,277] = 1
    GT[328,275] = 1
    GT[328,273] = 1
    GT[328,271] = 1
    GT[328,269] = 1
    GT[328,267] = 1
    GT[328,265] = 1
    GT[328,263] = 1
    GT[328,261] = 1
    GT[328,259] = 1
    GT[328,257] = 1
    GT[328,255] = 1
    GT[328,253] = 1
    GT[328,251] = 1
    GT[328,249] = 1
    GT[329,247] = 1
    GT[329,245] = 1
    GT[329,243] = 1
    GT[329,241] = 1
    GT[329,239] = 1
    GT[329,237] = 1
    GT[329,235] = 1
    GT[329,233] = 1
    GT[329,231] = 1
    GT[329,229] = 1
    GT[329,227] = 1
    GT[329,225] = 1
    GT[329,223] = 1
    GT[329,221] = 1
    GT[329,219] = 1
    GT[329,217] = 1
    GT[329,215] = 1
    GT[329,213] = 1
    GT[329,211] = 1
    GT[329,209] = 1
    GT[329,207] = 1
    GT[329,205] = 1
    GT[329,203] = 1
    GT[329,201] = 1
    GT[329,199] = 1
    GT[329,197] = 1
    GT[329,195] = 1
    GT[329,193] = 1
    GT[329,191] = 1
    GT[329,189] = 1
    GT[329,187] = 1
    GT[329,185] = 1
    GT[329,183] = 1
    GT[329,181] = 1
    GT[329,179] = 1
    GT[329,177] = 1
    #
    PIA=PI.copy()
    plt.figure(figsize=(20*img_rt, 20))
    for j in range(nrows):
        for i in range(ncols):
            if (GT[j, i] != 0):
                plt.plot(i, j, marker='+', color="darkorange")
    plt.imshow(PIA)
    plt.show()
    #
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(GT[260:360,130:330])
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(PI[260:360,130:330])
 #imgplot.set_clim(0.0, 0.7)
#ax.set_title('After')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    #fig = plt.figure()
    #plt.imshow(PI[300:450,170:200])
    #plt.imshow(PI)
def GT_flev_roi_01_bresenham(nrows, ncols):
    #
    GR= np.zeros((nrows, ncols))
    #
    xp, yp = bresenham(283, 307, 328, 308)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    xp, yp = bresenham(285, 172, 283, 307)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    xp, yp = bresenham(285, 172, 329, 173)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    xp, yp = bresenham(328, 308  , 329, 173)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    #plt.figure(figsize=(20*img_rt, 20))
    #for j in range(nrows):
    #    for i in range(ncols):
    #        if (GR[j, i] != 0):
    #            plt.plot(i, j, marker='o', color="darkorange")
    #plt.imshow(PIA)
    #plt.show()
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 2, 1)
    #plt.imshow(GR[260:360,130:330])
    #ax = fig.add_subplot(1, 2, 2)
    #plt.imshow(PIA[260:360,130:330])
    #plt.show()
    return GR
#
def GT_sf_roi_01_bresenham(nrows, ncols):
    #
    GR= np.zeros((nrows, ncols))
    #
    xp, yp = bresenham(367, 185, 420, 188)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    GR[360,199] = 1
    GR[360,197] = 1
    GR[360,195] = 1
    GR[360,193] = 1
    GR[361,191] = 1
    GR[361,189] = 1
    GR[362,188] = 1
    GR[363,187] = 1
    GR[364,186] = 1
    GR[365,185] = 1
    #
    xp, yp = bresenham(354, 247, 360, 201)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    #plt.figure(figsize=(20*img_rt, 20))
    #for j in range(nrows):
    #    for i in range(ncols):
    #        if (GR[j, i] != 0):
    #            plt.plot(i, j, marker='o', color="darkorange")
    #plt.imshow(PIA)
    #plt.show()
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 2, 1)
    #plt.imshow(GR[300:450,170:260])
    #ax = fig.add_subplot(1, 2, 2)
    #plt.imshow(PIA[300:450,170:260])
    #plt.show()
    return GR
    #
def GT_sim_flor(nrows, ncols):
    #
    GR= np.zeros((nrows, ncols))
    # Petal number (frequency)
    beta  = 26
    # Radius
    delta   = 200
    # wave amplitude
    nu    = 5
    x0 = nrows / 2
    y0 = ncols / 2
    t = np.linspace(0, 2 * np.pi, 2 * delta) - np.pi
    r =  (delta - nu * np.cos(beta * t))
    x = x0 + r * np.cos(t)
    y = y0 + r * np.sin(t)
    x = np.round(x)
    y = np.round(y)
    #
    n = len(x)
    for j in range(0,n):
        yp = int(x[j])
        xp = int(y[j])
        GR[xp, yp] =  1
    #
    return GR
    #
def GR_sub_scene_07(nrows, ncols, PIA):
    #
    GR= np.zeros((nrows, ncols))
    #xp, yp = bresenham(120, 280, 105, 270)
    #n = len(xp)
    #for j in range(0,n):
    #    GR[xp[j], yp[j]] =  1
    #
    GR[300, 78]  =  1
    GR[295, 79]  =  1
    GR[290, 80]  =  1
    GR[285, 81]  =  1
    GR[280, 84]  =  1
    GR[275, 87]  =  1
    GR[270, 89]  =  1
    GR[265, 92]  =  1
    GR[260, 94]  =  1
    GR[255, 96]  =  1
    GR[250, 98]  =  1
    GR[247, 100]  =  1
    GR[244, 102]  =  1
    GR[247, 105]  =  1
    GR[244, 109]  =  1
    GR[242, 113]  =  1
    #
    GR[244, 115]  =  1
    GR[246, 117]  =  1
    GR[248, 119]  =  1
    GR[248, 121]  =  1
    GR[248, 123]  =  1
    #
    GR[247, 125]  =  1
    GR[246, 127]  =  1
    GR[245, 128]  =  1
    GR[244, 129]  =  1
    GR[243, 130]  =  1
    GR[242, 131]  =  1
    GR[240, 133]  =  1
    GR[237, 135]  =  1
    GR[235, 137]  =  1
    GR[234, 139]  =  1
    #
    GR[236, 141]  =  1
    GR[237, 143]  =  1
    GR[238, 145]  =  1
    #
    GR[240, 145]  =  1
    GR[245, 146]  =  1
    GR[250, 147]  =  1
    GR[255, 148]  =  1
    GR[260, 149]  =  1
    GR[265, 150]  =  1
    GR[270, 151]  =  1
    #
    GR[275, 154]  =  1
    GR[280, 157]  =  1
    GR[285, 160]  =  1
    GR[290, 163]  =  1
    GR[293, 166]  =  1
    GR[297, 167]  =  1
    #

    img_rt = nrows / ncols
    plt.figure(figsize=(20*img_rt, 20))
    for j in range(nrows):
        for i in range(ncols):
            if (GR[j, i] != 0):
                plt.plot(i, j, marker='o', color="darkorange", markersize=3)
    plt.imshow(PIA)
    plt.show()
    return GR
    #
def GR_sub_scene_10(nrows, ncols, PIA):
    #
    GR= np.zeros((nrows, ncols))
    GR[110, 270]  =  1
    GR[115, 271]  =  1
    GR[120, 272]  =  1
    GR[125, 273]  =  1
    GR[130, 273]  =  1
    GR[135, 274]  =  1
    GR[140, 274]  =  1
    GR[145, 274]  =  1
    GR[150, 274]  =  1
    GR[155, 274]  =  1
    GR[160, 274]  =  1
    GR[165, 274]  =  1
    GR[170, 273]  =  1
    GR[175, 273]  =  1
    GR[180, 273]  =  1
    GR[185, 273]  =  1
    GR[190, 274]  =  1
    GR[195, 275]  =  1
    GR[200, 276]  =  1
    GR[205, 276]  =  1
    GR[210, 277]  =  1
    GR[215, 277]  =  1
    GR[220, 278]  =  1
    GR[225, 279]  =  1
    #
    GR[228, 280]  =  1
    GR[231, 282]  =  1
    #
    GR[232, 285]  =  1
    GR[233, 287]  =  1
    GR[233, 289]  =  1
    GR[233, 290]  =  1
    GR[233, 291]  =  1
    GR[233, 293]  =  1
    GR[232, 295]  =  1
    GR[231, 297]  =  1
    GR[229, 299]  =  1
    GR[228, 301]  =  1
    GR[227, 302]  =  1
    GR[227, 303]  =  1
    GR[227, 305]  =  1
    GR[226, 305]  =  1
    #
    GR[225, 307]  =  1
    GR[220, 308]  =  1
    GR[215, 309]  =  1
    GR[210, 311]  =  1
    GR[207, 315]  =  1
    GR[204, 318]  =  1
    GR[205, 319]  =  1
    GR[202, 317]  =  1
    GR[199, 315]  =  1
    GR[195, 318]  =  1
    GR[190, 320]  =  1
    GR[185, 322]  =  1
    GR[180, 320]  =  1
    GR[175, 324]  =  1
    GR[170, 324]  =  1
    #
    img_rt = nrows / ncols
    plt.figure(figsize=(20*img_rt, 20))
    for j in range(nrows):
        for i in range(ncols):
            if (GR[j, i] != 0):
                plt.plot(i, j, marker='o', color="darkorange", markersize=3)
    plt.imshow(PIA)
    plt.show()
    return GR
#
def GR_la_april_2009_image_1(nrows, ncols, PIA):
    #
    GR= np.zeros((nrows, ncols))
    xp, yp = bresenham(228, 193, 242, 241)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    img_rt = nrows / ncols
    plt.figure(figsize=(20*img_rt, 20))
    for j in range(nrows):
        for i in range(ncols):
            if (GR[j, i] != 0):
                plt.plot(i, j, marker='o', color="darkorange", markersize=3)
    plt.imshow(PIA)
    plt.show()
    #pplt.show_image_pauli_to_file_set(PIA, GR, "GR_la_imag1_april_2009")
    return GR
#
#
def GR_la_may_2015_image_1(nrows, ncols, PIA):
    #
    GR= np.zeros((nrows, ncols))
    xp, yp = bresenham(187, 226, 228, 227)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #xp, yp = bresenham(228, 193, 242, 241)
    #n = len(xp)
    #for j in range(0,n):
    #    GR[xp[j], yp[j]] =  1
    #
    GR[234, 193] = 1
    GR[234, 194] = 1
    GR[234, 195] = 1
    GR[235, 196] = 1
    GR[236, 197] = 1
    GR[237, 198] = 1
    GR[237, 199] = 1
    GR[237, 200] = 1
    GR[237, 201] = 1
    xp, yp = bresenham(237, 201, 246, 210)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    GR[247, 210] = 1
    GR[248, 210] = 1
    GR[248, 211] = 1
    GR[248, 212] = 1
    GR[248, 213] = 1
    GR[249, 213] = 1
    GR[250, 214] = 1
    xp, yp = bresenham(250, 215, 250, 223)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    #
    GR[251, 223] = 1
    GR[251, 224] = 1
    GR[251, 225] = 1
    GR[251, 226] = 1
    GR[251, 227] = 1
    GR[251, 228] = 1
    GR[251, 229] = 1
    GR[252, 229] = 1
    GR[253, 229] = 1
    xp, yp = bresenham(253, 230, 253, 237)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1
    xp, yp = bresenham(254, 238, 257, 242)
    n = len(xp)
    for j in range(0,n):
        GR[xp[j], yp[j]] =  1

    #
    img_rt = nrows / ncols
    plt.figure(figsize=(20*img_rt, 20))
    for j in range(nrows):
        for i in range(ncols):
            if (GR[j, i] != 0):
                plt.plot(i, j, marker='o', color="darkorange", markersize=3)
    plt.imshow(PIA)
    plt.show()
    #pplt.show_image_pauli_to_file_set(PIA, GR, "GR_la_imag1_may_2015")
    return GR
#
#
def GR_la_april_2009_image_2(nrows, ncols, PIA):
    #
    GR= np.zeros((nrows, ncols))
    GR[100, 100] =  1
    #
    img_rt = nrows / ncols
    plt.figure(figsize=(20*img_rt, 20))
    for j in range(nrows):
        for i in range(ncols):
            if (GR[j, i] != 0):
                plt.plot(i, j, marker='o', color="darkorange", markersize=3)
    plt.imshow(PIA)
    plt.show()
    return GR
#
#
def GR_la_may_2015_image_2(nrows, ncols, PIA):
    #
    GR= np.zeros((nrows, ncols))
    GR[100, 100] =  1
    #
    img_rt = nrows / ncols
    plt.figure(figsize=(20*img_rt, 20))
    for j in range(nrows):
        for i in range(ncols):
            if (GR[j, i] != 0):
                plt.plot(i, j, marker='o', color="darkorange", markersize=3)
    plt.imshow(PIA)
    plt.show()
    return GR
#
def crop_image_flev(IM, xl, yl):
    IMC = IM[xl[0]:xl[1], yl[0]:yl[1], :]
    return IMC
def crop_image_sf(IM, xl, yl):
    IMC = IM[xl[0]:xl[1], yl[0]:yl[1], :]
    return IMC
def crop_image_la_img1(IM, xl, yl):
    IMC = IM[xl[0]:xl[1], yl[0]:yl[1], :]
    return IMC
def crop_image_la_img2(IM, xl, yl):
    IMC = IM[xl[0]:xl[1], yl[0]:yl[1], :]
    return IMC
def crop_image_2d_flev(IM, xl, yl):
    IMC = IM[xl[0]:xl[1], yl[0]:yl[1]]
    return IMC
def crop_image_2d_sf(IM, xl, yl):
    IMC = IM[xl[0]:xl[1], yl[0]:yl[1]]
    return IMC
def crop_image_2d_la_imag1(IM, xl, yl):
    IMC = IM[xl[0]:xl[1], yl[0]:yl[1]]
    return IMC
def crop_image_2d_la_imag2(IM, xl, yl):
    IMC = IM[xl[0]:xl[1], yl[0]:yl[1]]
    return IMC
#  Initial guest to the parameters
def initial_guest_tau(z1, z2):
    est = np.mean(z1) / np.mean(z2)
    return est
def initial_guest_rho(z1, z2):
    d = len(z1)
    zaux1 = np.multiply(z1, z2)
    aux1 = np.mean(zaux1)
    aux2 = sum(z1)
    aux3 = sum(z2)
    aux4 = (aux1 * aux2) / d
    div1 = aux1 - aux4
    aux5 = np.var(z1)
    aux6 = np.var(z2)
    div2 = np.sqrt(aux5 * aux6)
    est = np.abs(div1 / div2)**2
    return est
def initial_guest_L(z1, z2):
    L1 = initial_guest_L_each_channel(z1)
    L2 = initial_guest_L_each_channel(z2)
    est = (L1 + L2) / 2
    return est
def initial_guest_L_each_channel(z):
    est = np.mean(z)**2  / np.var(z)
    return est
def same_region_threshold(z, N, matdf1, matdf2):
    l = np.zeros(N)
    for i in range(1, N):
        l[i] = ptl.func_obj_l_L_mu(i,z, N, matdf1, matdf2)
    #plt.plot(l[1:N])
    media = np.mean(l)
    std = np.std(l)
    threshold_sup =  media + std * 0.25
    threshold_inf =  media - std * 0.25
    return threshold_sup, threshold_inf
