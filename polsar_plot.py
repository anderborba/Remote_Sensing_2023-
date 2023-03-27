import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
#
import os.path
#
import polsar_pdfs as pp
import polsar_total_loglikelihood as ptl
#
def show_evidence(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
    PIA=pauli.copy()
    plt.figure(figsize=(20*img_rt, 20))
    for k in range(NUM_RAIOS):
        ik = int(evidence[k, banda])
        ia = int(MXC[k, ik])
        ja = int(MYC[k, ik])
        plt.plot(ia, ja, marker='o', color="darkorange")
    plt.imshow(PIA)
    plt.show()
    return PIA
# Set evidence in a simulated image
def add_evidence_simulated(nrows, ncols, ncanal, evidencias):
    IM  = np.zeros([nrows, ncols, ncanal])
    for canal in range(ncanal):
        for k in range(nrows):
            ik = int(evidencias[k, canal])
            IM[ik, k, canal] = 1
    return IM
## Shows the evidence in simulated image
def show_evidence_simulated(pauli, NUM_RAIOS, img_rt, evidence, banda):
	PIA=pauli.copy()
	plt.figure(figsize=(20*img_rt, 20))
	for k in range(NUM_RAIOS):
    		ik = int(evidence[k, banda])
    		plt.plot(ik, k, marker='o', color="darkorange")
	plt.imshow(PIA)
	plt.show()
#
def plot_pdf_prod_intensities(z, L, rho, s1, s2):
    #
    plt.rc('text', usetex=True)
    #
    m = len(z)
    fz = np.zeros(m)
    for i in range(m):
            fz[i] = pp.pdf_prod_intensities(z[i], L, rho, s1, s2)
    #
    print(fz)
    plt.plot(z[1:m], fz[1:m], 'ro' )
    plt.show()
def plot_log_pdf_prod_intensities(z, L, rho, s1, s2):
    #
    plt.rc('text', usetex=True)
    #
    m = len(z)
    fz = np.zeros(m)
    for i in range(m):
            fz[i] = pp.log_pdf_prod_intensities(z[i], L, rho, s1, s2)
    #
    plt.plot(z[1:m], fz[1:m], 'ro')
    plt.show()
#
#
def plot_pdf_prod_intensities_biv(rho, L, s1, s2):
    #
    plt.rc('text', usetex=True)
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    #
    m = 1000
    z1 = np.zeros(m)
    z2 = np.zeros(m)
    a = 0
    b = 1
    h = (b - a) / m
    for i in range(m):
        z1[i] = a + h * i
        z2[i] = a + h * i
    s = (m, m)
    fz = np.zeros(s)
    for i in range(m):
        for j in range(m):
            fz[i, j] = pp.pdf_ratio_intensities(z1[i], z2[j], rho, L, s1, s2)
    #
    x1, y1 = np.meshgrid(z1, z2)
    surf = ax.plot_surface(x1, y1, fz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #plt.xlabel(r'$\sigma_1$')
    #plt.ylabel(r'$\sigma_2$')
    #plt.title(r'PDF Bivariate Product of Intensities$')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
#
def plot_total_likelihood(z, N, matdf1, matdf2):
    plt.style.use('ggplot')
    pix = np.zeros(N)
    fob = np.zeros(N)
    for j in range(1, N):
        pix[j] = j
        fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)
    plt.plot(pix[1:N], fob[1:N])
    plt.show()
#
def plot_total_likelihood_prod_biv(z1, z2, N, matdf1, matdf2):
    plt.style.use('ggplot')
    pix = np.zeros(N)
    fob = np.zeros(N)
    for j in range(1, N):
        pix[j] = j
        fob[j] = ptl.func_obj_l_intensity_prod_biv(pix[j], z1, z2, N, matdf1, matdf2)
    plt.plot(pix[1:N], fob[1:N])
    plt.show()
#
def plot_total_likelihood_prod_int(z, N, matdf1, matdf2):
    plt.style.use('ggplot')
    pix = np.zeros(N)
    fob = np.zeros(N)
    for j in range(1, N):
        pix[j] = j
        fob[j] = ptl.func_obj_l_intensity_prod(pix[j], z, N, matdf1, matdf2)
    plt.plot(pix[1:N], fob[1:N])
    plt.show()
def plot_total_likelihood_prod_int_sum(z, N, matdf1, matdf2):
    plt.style.use('ggplot')
    pix = np.zeros(N)
    fob = np.zeros(N)
    for j in range(1, N):
        pix[j] = j
        fob[j] = ptl.func_obj_l_intensity_prod_sum(pix[j], z, N, matdf1, matdf2)
    print(fob)
    plt.plot(pix[1:N], fob[1:N])
    plt.show()
#
def show_image(IMG, nrows, ncols, img_rt):
    plt.figure(figsize=(20*img_rt, 20))
    escale = np.mean(IMG) * 2
    plt.imshow(IMG,clim=(0.0, escale), cmap="gray")
    plt.show()
#
def show_image_max(IMG, nrows, ncols, img_rt):
    plt.figure(figsize=(20*img_rt, 20))
    escale = np.max(IMG)
    IMG = IMG / escale
    plt.imshow(IMG,clim=(0.0, escale), cmap="gray")
    #plt.imshow(IMG, cmap="gray")
    plt.show()
#
def show_image_prod_intensities(IMG, nrows, ncols, img_rt, c1, c2):
    plt.figure(figsize=(20*img_rt, 20))
    IMGR = IMG[:, :, c1] * IMG[:, :, c2]
    escale = np.mean(IMGR) * 2
    plt.imshow(IMGR,clim=(0.0, escale), cmap="gray")
    plt.show()
#
def show_image_ratio_intensities(IMG, nrows, ncols, img_rt, c1, c2):
    plt.figure(figsize=(20*img_rt, 20))
    IMGR = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            if IMG[i, j, c1] > 0 and IMG[i, j, c2] > 0:
                IMGR[i, j] = IMG[i, j, c1] / IMG[i, j, c2]
#
    escale = np.mean(IMGR) * 2
    plt.imshow(IMGR,clim=(0.0, escale), cmap="gray")
    plt.show()
def show_image_ratio_intensities_to_files(IMG, nrows, ncols, img_rt, c1, c2, image_name):
    #plt.figure(figsize=(20*img_rt, 20))
    IMGR = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            if IMG[i, j, c1] > 0 and IMG[i, j, c2] > 0:
                IMGR[i, j] = IMG[i, j, c1] / IMG[i, j, c2]
    #
    directory = './figure/'
    image = str(image_name) + '.pdf'
    file_path = os.path.join(directory, image)
    escale = np.mean(IMGR) * 2
    plt.imsave(file_path, IMGR, cmap="gray", vmin = 0, vmax = escale)
#
#def show_image_pauli(IMG, nrows, ncols, img_rt):
#    plt.figure(figsize=(20*img_rt, 20)
#    plt.imshow(IMG)
#    plt.show()
#
def show_image_to_file(IMG, nrows, ncols, image_name):
    directory = './figure/'
    image = str(image_name) + '.pdf'
    file_path = os.path.join(directory, image)
    escale = np.mean(IMG) * 2
    plt.imsave(file_path, IMG, cmap="gray", vmin = 0, vmax = escale)
    #
def show_image_pauli_to_file(pauli, IMAGE, nrows, ncols, img_rt, image_name):
    directory = './figure/'
    image = str(image_name) + '.pdf'
    file_path = os.path.join(directory, image)
    FIG = plt.figure(figsize=(20*img_rt, 20))
    PIA=pauli.copy()
    for i in range(nrows):
        for j in range(ncols):
            if(IMAGE[i,j] != 0):
                #plt.plot(j,i, marker='o', color="darkorange")
                plt.plot(j,i, marker='o', color="darkorange")
    plt.imshow(PIA)
    plt.show()
    FIG.savefig(file_path)
    #
def show_image_pauli_to_file_set(pauli, IMAGE, image_name):
    dim = pauli.shape
    nrows = dim[0]
    ncols = dim[1]
    img_rt = nrows/ncols
    directory = './figure/'
    image = str(image_name) + '.pdf'
    file_path = os.path.join(directory, image)
    FIG = plt.figure(figsize=(20*img_rt, 20))
    PIA=pauli.copy()
    for i in range(nrows):
        for j in range(ncols):
            if(IMAGE[i,j] != 0):
                #plt.plot(j,i, marker='o', color="darkorange", markersize=26)
                # Using to subcene images
                plt.plot(j,i, marker='o', color="darkorange", markersize=10)
                # Using to simulated image
                #plt.plot(j,i, marker='o', color="darkorange", markersize=7)
                #plt.plot(j,i, marker='o', color="yellow")
                #plt.plot(j,i, marker='o', color="blue")
                #plt.plot(j,i, marker='o', color="dimgrey")
                #plt.plot(j,i, marker='o', color="darkblue")
                # contrast 0< alpha <1
                #plt.plot(j,i, marker='o', color="wheat", markersize=26)
                #plt.plot(j,i, marker='o', color="orangered")
                #plt.plot(j,i, marker='o', color="ghostwhite")
                #plt.plot(j,i, marker='o', color="bisque")
                #plt.plot(j,i, marker='o', color="indigo")
                #plt.plot(j,i, marker='o', color="violet")
    plt.imshow(PIA)
    plt.axis('off')
    FIG.savefig(file_path, bbox_inches='tight', pad_inches=0)
#
def show_image_pauli_gray_to_file_set(pauli, IMAGE, image_name):
    dim = pauli.shape
    nrows = dim[0]
    ncols = dim[1]
    PIA_gray = np.zeros([nrows, ncols])
    img_rt = nrows/ncols
    directory = './figure/'
    image = str(image_name) + '.pdf'
    file_path = os.path.join(directory, image)
    FIG = plt.figure(figsize=(20*img_rt, 20))
    PIA=pauli.copy()
    PIA = np.float32(PIA)
    PIA_gray = cv2.cvtColor(PIA, cv2.COLOR_BGR2GRAY)
    for i in range(nrows):
        for j in range(ncols):
            if(IMAGE[i,j] != 0):
                plt.plot(j,i, marker='o', color="darkorange", markersize=20)
    plt.imshow(PIA_gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    FIG.savefig(file_path, bbox_inches='tight', pad_inches=0)
    #tight_layout()
    #plt.show()
    #FIG.savefig(file_path)
    #
def show_image_pauli_to_file_set_unless_img(pauli,image_name):
    dim = pauli.shape
    nrows = dim[0]
    ncols = dim[1]
    img_rt = nrows/ncols
    directory = './figure/'
    image = str(image_name) + '.pdf'
    file_path = os.path.join(directory, image)
    FIG = plt.figure(figsize=(20*img_rt, 20))
    PIA=pauli.copy()
    plt.imshow(PIA)
    plt.axis('off')
    FIG.savefig(file_path, bbox_inches='tight', pad_inches=0)
#
def show_evidence_to_file(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda, image_name):
    directory = './figure/'
    image = str(image_name) + '.pdf'
    file_path = os.path.join(directory, image)
    FIG = plt.figure(figsize=(20*img_rt, 20))
    PIA=pauli.copy()
    for k in range(NUM_RAIOS):
        ik = int(evidence[k, banda])
        ia = int(MXC[k, ik])
        ja = int(MYC[k, ik])
        plt.plot(ia, ja, marker='o', color="darkorange")
    plt.imshow(PIA)
    plt.show()
    FIG.savefig(file_path)
    return PIA
#
def show_image_perfil_h(img, cont, channel):
    plt.style.use('ggplot')
    plt.plot(img[:, cont, channel])
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.show()
#
def fun_smooth_ramp_3d_corner(x, y):
    f1 = 6.0 * x**5 - 15.0 * x**4 + 10 * x**3
    f2 = 6.0 * y**5 - 15.0 * y**4 + 10 * y**3
    f = f1 * f2
    return f
#
def fun_smooth_ramp(x):
    f = 6.0 * x**5 - 15.0 * x**4 + 10 * x**3
    return f
#
def fun_smooth_ramp_3d_set(x, y, nrows, ncols):
    eps = 50;
    sx = 100;
    sy = 200;
    xl = int(nrows/2 - sx);
    xu = int(nrows/2 + sx);
    yl = int(ncols/2 - sy);
    yu = int(ncols/2 + sy);
    z = np.zeros((nrows, ncols))
    # Build a rectangle [xl + eps, xu - eps] X [yl + eps, yu - eps]
    for i in range(xl + eps, xu - eps):
        for j in range(yl + eps, yu - eps):
            z[i, j] = 1.0
    # Upper rectangle [xl - eps, xl + eps] X [yl + eps, yu - eps]
    for i in range(xl - eps, xl + eps):
        for j in range(yl + eps, yu - eps):
            xaux = (i - (xl - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp(xaux)
    # Lower rectangle  [xu - eps, xu + eps] X [yl + eps, yu - eps]
    for i in range(xu - eps, xu + eps):
        for j in range(yl + eps, yu - eps):
            xaux = (i - (xu - eps)) / (2 * eps)
            z[i, j] = 1.0 - fun_smooth_ramp(xaux)
    # left rectangle [xl + eps, xu - eps] X [yl - eps, yl + eps]
    for i in range(xl + eps, xu - eps):
        for j in range(yl - eps, yl + eps):
            yaux = (j - (yl - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp(yaux)
    # right rectangle [xl + eps, xu - eps] X [yu - eps, yu + eps]
    for i in range(xl + eps, xu - eps):
        for j in range(yu - eps, yu + eps):
            yaux = (j - (yu - eps)) / (2 * eps)
            z[i, j] = 1.0 - fun_smooth_ramp(yaux)
    # Smoth ramp to corner
    # [xl - eps, xl + eps] X [yl - eps, yl + eps]
    for i in range(xl - eps, xl + eps):
        for j in range(yl - eps, yl + eps):
            xaux = (i - (xl - eps)) / (2 * eps)
            yaux = (j - (yl - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp_3d_corner(xaux, yaux)
    # [xl - eps, xl + eps] X [yu - eps, yu + eps]
    for i in range(xl - eps, xl + eps):
        for j in range(yu - eps, yu + eps):
            xaux =       (i - (xl - eps)) / (2 * eps)
            yaux = 1.0 - (j - (yu - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp_3d_corner(xaux, yaux)
    # [xu - eps, xu + eps] X [yl - eps, yl + eps]
    for i in range(xu - eps, xu + eps):
        for j in range(yl - eps, yl + eps):
            xaux = 1.0 - (i - (xu - eps)) / (2 * eps)
            yaux =       (j - (yl - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp_3d_corner(xaux, yaux)
    # [xu - eps, xu + eps] X [yu - eps, yu + eps]
    for i in range(xu - eps, xu + eps):
        for j in range(yu - eps, yu + eps):
            xaux =   1.0 -  (i - (xu - eps)) / (2 * eps)
            yaux =   1.0 -  (j - (yu - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp_3d_corner(xaux, yaux)
    return z
def image_contrast_brightness(IM):
    alpha = 0.8
    beta  = 0.0
    IMAUX = alpha * IM + beta
    return IMAUX
#def plot_3d_edge(nrows, ncols, image_name):
    #directory = './figuras/'
    #figure = str(figure_name) + '.pdf'
    #file_path = os.path.join(directory, figure)
    # Domain [a, b] x [c, d]
    #a = 0
    #b = nrows
    #
    #c = 0
    #d = ncols
    #
#    x = np.arange(0, nrows, 1)
#    y = np.arange(0, ncols, 1)
#    x, y = np.meshgrid(x, y)
    #z = fun_smooth_ramp_3d_set(x, y, nrows, ncols)
    #fig = plt.figure(figsize =(14, 9))
#    ax = plt.axes(projection ='3d')
    #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=True)
    #surf = ax.plot_surface(x, y, z,
    #					rstride = 1,
    # 					cstride = 1,
    #					alpha = 0.4,
    #					color= 'lightgreen')
    #  Gráfico reta 1
    #x1 = np.linspace(0, 4, 100)
    #y1 = x1
    #ax.plot(x1, y1, zs=0, zdir='z', color='blue')
    #  Gráfico reta 2
    #x1 = np.linspace(0, 3, 100)
    #y1 = x1 + np.sqrt(2)
    #ax.plot(x1, y1, zs=0, zdir='z', color='blue')
    #  Plot reta 3
    #x1 = np.linspace(0, 4, 100)
    #y1 = -x1 + 4
    #ax.plot(x1, y1, zs=0, zdir='z', color='orange')
    #z2 = 1
    #x2 = np.linspace(a - np.sqrt(z2), a + np.sqrt(z2), 100)
    #y2 =  np.sqrt(z2 - (x2 - a)**2) + b
    #y3 = -np.sqrt(z2 - (x2 - a)**2) + b
    #ax.plot(x2, y2, zs=0, zdir='z', color='blue')
    #ax.plot(x2, y3, zs=0, zdir='z', color='blue')
    #
    #ax.plot(x2, y2, zs=1, zdir='z', color='green')
    #ax.plot(x2, y3, zs=1, zdir='z', color='green')
    # plot curva de nível 2
    #z2 = 2
    #x2 = np.linspace(a - np.sqrt(z2) + epsilon, a + np.sqrt(z2), 100)
    #y2 =  np.sqrt(z2 - (x2 - a)**2) + b
    #y3 = -np.sqrt(z2 - (x2 - a)**2) + b
    #ax.plot(x2, y2, zs=0, zdir='z', color='blue')
    #ax.plot(x2, y3, zs=0, zdir='z', color='blue')
    #
    #ax.plot(x2, y2, zs=2, zdir='z', color='green')
    #ax.plot(x2, y3, zs=2, zdir='z', color='green')
    # plot curva de nível 3
    #z2 = 3
    #x2 = np.linspace(a - np.sqrt(z2) + epsilon, a + np.sqrt(z2), 100)
    #y2 =  np.sqrt(z2 - (x2 - a)**2) + b
    #y3 = -np.sqrt(z2 - (x2 - a)**2) + b
    #ax.plot(x2, y2, zs=0, zdir='z', color='blue')
    #ax.plot(x2, y3, zs=0, zdir='z', color='blue')
    #
    #ax.plot(x2, y2, zs=3, zdir='z', color='green')
    #ax.plot(x2, y3, zs=3, zdir='z', color='green')
    #
    #theta = np.linspace(0.6, 2.0, 100)
    #xc = theta
    #yc = xc + np.sqrt(2)
    #zc = (xc - a)**2 + (yc - b)**2
    #ax.plot(xc, yc, zc, color='red')
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
 #   plt.xlabel('x Pixel')
 #   ax.set_ylabel('y Pixel')
    #ax.set_xlim(0, 4)
    #ax.set_ylim(0, 4)
 #   ax.set_zlabel('Pixel intensity')
 #   ax.set_title('Function smooth ramp')
 #   plt.show()

#
