## Version 03/08/2021
# Article GRSL
# Ref:
# A. A. De Borba,
# M. Marengoni and
# A. C. Frery,
# "Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images,"
# in IEEE Geoscience and Remote Sensing Letters,
#doi: 10.1109/LGRS.2020.3022511.
# bibtex
#@ARTICLE{9203845,
#  author={De Borba, Anderson A. and Marengoni, Maurício and Frery, Alejandro C.},
#  journal={IEEE Geoscience and Remote Sensing Letters},
#  title={Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images},
#  year={2020},
#  volume={},
#  number={},
#  pages={1-5},
#  doi={10.1109/LGRS.2020.3022511}}
#
## Import all required libraries
import numpy as np
import os.path
## Used to read images in the mat format
#import scipy.io as sio
## Used to equalize histograms in images
#from skimage import exposure
## Used to present the images
#import matplotlib as mpl
#import matplotlib.pyplot as plt
## Used to find border evidences
#import math
#from scipy.optimize import dual_annealing
## Used in the DWT and SWT fusion methods
#import pywt
#### Used to find_evidence_bfgs
#from scipy.optimize import minimize
#from scipy.spatial.distance import directed_hausdorff
## Used
### Import mod
# see file  /Misc/mod_code_py.pdf
#
import polsar_basics as pb
#import polsar_loglikelihood as plk
import polsar_fusion as pf
#import polsar_total_loglikelihood as ptl
import polsar_evidence_lib as pel
import polsar_plot as pplt
import polsar_metrics as pm
#
#import matplotlib.pyplot as plt
#
## This function defines the source image and all the dat related to the region where we want
## to find borders
## Defines the ROI center and the ROI boundaries. The ROI is always a quadrilateral defined from the top left corner
## in a clockwise direction.
#
def select_data():
    print("Select the image to be processed:")
    print("1. Flevoland - area 1")
    print("2. San Francisco")
    print("3. Simulated image")
    print("4. Data P band")
    print("5. Los Angeles - Image 1")
    print("6. Los Angeles - Image 2")
    opcao=int(input("Type the option:"))
    if opcao==1:
        print("Computing Flevoland area - region 1")
        imagem="./Data/AirSAR_Flevoland_Enxuto.mat"
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx=278
        dy=64
        ## ROI coordinates
        x1 = 157;
        y1 = 284;
        x2 = 309;
        y2 = 281;
        x3 = 310;
        y3 = 327;
        x4 = 157;
        y4 = 330;
        ## inicial angle to start generating the radius
        alpha_i = 0.0
        ## final angle to start generating the radius
        alpha_f = 2 * np.pi
        ## slack constant
        lim = 14
        ## Radius length
        RAIO=120
        opcao_subscene = 0
    elif opcao==2:
        print("Computing San Francisco Bay area - region 1")
        imagem="./Data/SanFrancisco_Bay.mat"
        print(imagem)
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx=50
        dy=-195
        ## ROI coordinates
        x1 = 180;
        y1 = 362;
        x2 = 244;
        y2 = 354;
        x3 = 250;
        y3 = 420;
        x4 = 188;
        y4 = 427;
        ## inicial angle to start generating the radius
        alpha_i = np.pi
        ## final angle to start generating the radius
        alpha_f = 3 * np.pi / 2
        ## slack constant
        lim = 25
        ## Radius length
        RAIO=120
        opcao_subscene = 0
    elif opcao == 3:
        print("Computing Simulated image")
        imagem="./Data/flor_simulada.mat"
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx = 0
        dy = 0
        ## ROI coordinates
        x1 = 180;
        y1 = 362;
        x2 = 244;
        y2 = 354;
        x3 = 250;
        y3 = 420;
        x4 = 188;
        y4 = 427;
        ## inicial angle to start generating the radius
        alpha_i = 0
        ## final angle to start generating the radius
        alpha_f = 2 * np.pi
        ## slack constant
        lim = 10
        ## Radius length
        RAIO=300
        opcao_subscene = 0
    elif opcao == 4:
        print("Computing Data P Band - DPB")
        print("1. Subscene 7")
        print("2. Subscene 10")
        opcao_subscene=int(input("Type subscene option:"))
        if opcao_subscene == 1:
            imagem = []
            imagem.append("./Data/Img7_HH.txt")
            imagem.append("./Data/Img7_HV.txt")
            imagem.append("./Data/Img7_VV.txt")
            ## values adjusted visually - it needs to be defined more preciselly
            ## delta values from the image center to the ROI center
            dx =  70
            dy = -100
            ## inicial angle to start generating the radius
            alpha_i = np.pi
            ## final angle to start generating the radius
            alpha_f = 2 * np.pi
            ## slack constant
            lim = 15
            ## Radius length
            RAIO=90
        else:
            imagem = []
            imagem.append("./Data/Img10_HH.txt")
            imagem.append("./Data/Img10_HV.txt")
            imagem.append("./Data/Img10_VV.txt")
            ## values adjusted visually - it needs to be defined more preciselly
            ## delta values from the image center to the ROI center
            dx =  -110
            dy =  60
            ## inicial angle to start generating the radius
            alpha_i = np.pi / 3
            ## final angle to start generating the radius
            alpha_f = np.pi + np.pi / 4
            ## slack constant
            lim = 15
            ## Radius length
            RAIO=120
    elif opcao == 5:
        print("Computing Los Angeles area - Image 1")
        print("1. Image - April - 2009")
        print("2. Image - May - 2015")
        opcao_subscene=int(input("Type subscene option:"))
        imagem="./Data/Nizar_Input_LosAngeles2_UAVSAR_LL_6_ver2.mat"
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx=-30
        dy=195
        ## ROI coordinates
        x1 = 157;
        y1 = 284;
        x2 = 309;
        y2 = 281;
        x3 = 310;
        y3 = 327;
        x4 = 157;
        y4 = 330;
        ## inicial angle to start generating the radius
        alpha_i = 0
        ## final angle to start generating the radius
        alpha_f = 5 * np.pi / 12
        ## slack constant
        lim = 14
        ## Radius length
        if opcao_subscene == 1:
            RAIO= 85
        else:
            RAIO= 85
    else:
        print("Computing Los Angeles area - Image 2")
        print("1. Image - April - 2009")
        print("2. Image - May - 2015")
        opcao_subscene=int(input("Type subscene option:"))
        imagem="./Data/Nizar_Input_LosAngeles3_UAVSAR_LL_6_ver2.mat"
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx=278
        dy=64
        ## ROI coordinates
        x1 = 157;
        y1 = 284;
        x2 = 309;
        y2 = 281;
        x3 = 310;
        y3 = 327;
        x4 = 157;
        y4 = 330;
        ## inicial angle to start generating the radius
        alpha_i = 0.0
        ## final angle to start generating the radius
        alpha_f = 2 * np.pi
        ## slack constant
        lim = 14
        ## Radius length
        RAIO=120
        opcao_subscene = 0
    ## Number of radius used to find evidence considering a whole circunference
    NUM_RAIOS=100
    ## adjust the number of radius based on the angle defined above
    if  (alpha_f-alpha_i)!=(2*np.pi):
        NUM_RAIOS=int(NUM_RAIOS*(alpha_f-alpha_i)/(2*np.pi))
    print(NUM_RAIOS)
    #gt_coords=[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    #
    return imagem, dx, dy, RAIO, NUM_RAIOS, alpha_i, alpha_f, lim, opcao, opcao_subscene
# ### A célula abaixo funciona como um main do código de fusão de evidências de borda em imagens POLSAR - ainda deverá ser editado para uma melhor compreensão do código ###
# The code works as main to GRSL2020 codes
#
#cs1 = 'FlevEvChhRoi01Span'
#cs2 = 'FlevEvChhRoi02Span'
#cs3 = 'FlevEvChhRoi03Span'
## Define the image and the data from the ROI in the image
imagem, dx, dy, RAIO, NUM_RAIOS, alpha_i, alpha_f, lim, opcao, opcao_subscene = select_data()
#
## Reads the image and return the image, its shape and the number of channels
img, nrows, ncols, nc = pb.le_imagem(imagem, opcao, opcao_subscene)
#
## Plot parameter
img_rt = nrows/ncols
## show image in each channel
#pb.GR_la_april_2009_image_1(nrows, ncols, PI_AUX)
#channel_1 = 1
#channel_2 = 2
#pplt.show_image_prod_intensities(img, nrows, ncols, img_rt, channel_1, channel_2)
#pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, 'img_teste')
#pplt.show_image_max(img[:, :, 0], nrows, ncols, img_rt)
#pplt.show_image(img[:, :, 0], nrows, ncols, img_rt)
#pplt.show_image_ratio_intensities(img, nrows, ncols, img_rt, channel_1, channel_2)

## Uses the Pauli decomposition to generate a visible image
PI = pb.show_Pauli(img, 1, 0)
PI_AUX = pb.show_Pauli(img, 1, 0)
#
## Define the radius in the ROI
x0, y0, xr, yr = pb.define_radiais(RAIO, NUM_RAIOS, dx, dy, nrows, ncols, alpha_i, alpha_f)
#
MXC, MYC, MY, IT, PI = pb.desenha_raios(ncols, nrows, nc, RAIO, NUM_RAIOS, img, PI, x0, y0, xr, yr)
#
## Define the number of channels to be used to find evidence
## and realize the fusion in the ROI
ncanal = 10
#total_canal = 9
evidencias = np.zeros((NUM_RAIOS, ncanal))
zeros_count_channels = np.zeros((NUM_RAIOS, ncanal))

pel.teste_limiar(RAIO, NUM_RAIOS, 0, MY, lim)
## Find the evidences
# Intensity channals pdf (gamma pdf)
#evidencias[:, 0], zeros_count_channels[:, 0] = pel.find_evidence_bfgs(RAIO, NUM_RAIOS, 0, MY, lim)
#evidencias[:, 0], zeros_count_channels[:, 0] = pel.find_evidence_bfgs_zeros_adaptive(RAIO, NUM_RAIOS, 0, MY, lim)
#evidencias[:, 0], zeros_count_channels[:, 0] = pel.find_evidence_bfgs_zeros_adaptive_loglike_adaptive(RAIO, NUM_RAIOS, 0, MY, lim)
evidencias[:, 0], zeros_count_channels[:, 0] = pel.find_evidence_bfgs_same_region_threshold(RAIO, NUM_RAIOS, 0, MY, lim)
#evidencias[:, 1], zeros_count_channels[:, 1] = pel.find_evidence_bfgs(RAIO, NUM_RAIOS, 1, MY, lim)
evidencias[:, 1], zeros_count_channels[:, 1] = pel.find_evidence_bfgs_same_region_threshold(RAIO, NUM_RAIOS, 1, MY, lim)
#evidencias[:, 2], zeros_count_channels[:, 2] = pel.find_evidence_bfgs(RAIO, NUM_RAIOS, 2, MY, lim)
evidencias[:, 2], zeros_count_channels[:, 2] = pel.find_evidence_bfgs_same_region_threshold(RAIO, NUM_RAIOS, 2, MY, lim)
# Span pdf
#evidencias[:, 3], zeros_count_channels[:, 3] =  pel.find_evidence_bfgs_span(RAIO, NUM_RAIOS, MY, lim)
evidencias[:, 3], zeros_count_channels[:, 3] =  pel.find_evidence_bfgs_span_same_region_threshold(RAIO, NUM_RAIOS, MY, lim)
# Ratio intensities pdf ( inum / idem )
# 0 = HH, 1 = HV, 2 = VV
inum = 0
idem = 1
#evidencias[:, 4], zeros_count_channels[:, 4] = pel.find_evidence_bfgs_intensity_ratio_three_param(RAIO, NUM_RAIOS, MY, lim, inum, idem)
evidencias[:, 4], zeros_count_channels[:, 4] = pel.find_evidence_bfgs_intensity_ratio_three_param_same_region_threshold(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 0
idem = 2
#evidencias[:, 5], zeros_count_channels[:, 5] = pel.find_evidence_bfgs_intensity_ratio_three_param(RAIO, NUM_RAIOS, MY, lim, inum, idem)
evidencias[:, 5], zeros_count_channels[:, 5] = pel.find_evidence_bfgs_intensity_ratio_three_param_same_region_threshold(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 1
idem = 2
#evidencias[:, 6], zeros_count_channels[:, 6] = pel.find_evidence_bfgs_intensity_ratio_three_param(RAIO, NUM_RAIOS, MY, lim, inum, idem)
evidencias[:, 6], zeros_count_channels[:, 6] = pel.find_evidence_bfgs_intensity_ratio_three_param_same_region_threshold(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 1
idem = 0
#evidencias[:, 7], zeros_count_channels[:, 7] = pel.find_evidence_bfgs_intensity_ratio_three_param(RAIO, NUM_RAIOS, MY, lim, inum, idem)
evidencias[:, 7], zeros_count_channels[:, 7] = pel.find_evidence_bfgs_intensity_ratio_three_param_same_region_threshold(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 2
idem = 1
#evidencias[:, 8], zeros_count_channels[:, 8] = pel.find_evidence_bfgs_intensity_ratio_three_param(RAIO, NUM_RAIOS, MY, lim, inum, idem)
evidencias[:, 8], zeros_count_channels[:, 8] = pel.find_evidence_bfgs_intensity_ratio_three_param_same_region_threshold(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 2
idem = 0
#evidencias[:, 9], zeros_count_channels[:, 9] = pel.find_evidence_bfgs_intensity_ratio_three_param(RAIO, NUM_RAIOS, MY, lim, inum, idem)
evidencias[:, 9], zeros_count_channels[:, 9] = pel.find_evidence_bfgs_intensity_ratio_three_param_same_region_threshold(RAIO, NUM_RAIOS, MY, lim, inum, idem)
#
#c1 = 0
#c2 = 2
#evidencias[:, 10] = pel.find_evidence_bfgs_int_prod_biv(RAIO, NUM_RAIOS, MY, lim, c1, c2)
#mul1 = 0
#mul2 = 1
#evidencias[:, 10] = pel.find_evidence_bfgs_prod_int(RAIO, NUM_RAIOS, MY, lim, mul1, mul2)

## Put the evidences in an image
IM = pel.add_evidence(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC)
#IM = pel.add_evidence_without_zeros(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC)
## Computes fusion using mean - metodo = 1
#MEDIA = pf.fusao(IM, 1, NUM_RAIOS)
## Computes fusion using pca - metodo = 2
#pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 0)
#pb.GR_la_may_2015_image_1(nrows, ncols, PI)
#pb.GR_la_april_2009_img_01(nrows, ncols, PI)
print("Image croping:")
comp_princ = np.zeros(nc)
comp_princ_crop = np.zeros(nc)
if opcao == 1:
    ## Crop definition to Flevoland to PCA analisys
    ##  IM[220:400, 130:400, 0]
    xlim = [220, 400]
    ylim = [130, 340]
    IMC = pb.crop_image_flev(IM, xlim, ylim)
    PCA, comp_princ_crop, MCOVAR_crop = pf.fusao(IMC, 2, NUM_RAIOS, comp_princ_crop)
    #pplt.show_image(IMC[:, :, 0], nrows, ncols, img_rt)
elif opcao == 2:
    ## Crop definition to SF to PCA analisys
    ##  IM[220:400, 130:400, 0]
    xlim = [320, 470]
    ylim = [140, 290]
    IMC = pb.crop_image_sf(IM, xlim, ylim)
    PCA, comp_princ_crop, MCOVAR_crop = pf.fusao(IMC, 2, NUM_RAIOS, comp_princ_crop)
elif opcao == 5:
    ## Crop definition to SF to PCA analisys
    ##  IM[220:400, 130:400, 0]
    xlim = [200, 255]
    ylim = [175, 240]
    IMC = pb.crop_image_la_img1(IM, xlim, ylim)
    PCA, comp_princ_crop, MCOVAR_crop = pf.fusao(IMC, 2, NUM_RAIOS, comp_princ_crop)
    #pplt.show_image(IMC[:, :, 0], nrows, ncols, img_rt)
#
#pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 0)
PCA, comp_princ, MCOVAR = pf.fusao(IM, 2, NUM_RAIOS, comp_princ)
## Computes fusion using ROC - metodo = 3
ROC = pf.fusao(IM, 3, NUM_RAIOS, comp_princ)[0]
print("passou")
ROC_THRESHOLD = pf.fusao(IM, 7, NUM_RAIOS, comp_princ)[0]
#
## Testing fusion using SVD - metodo = 4
#SVD = pf.fusao(IM, 4, NUM_RAIOS)
## Testing fusion using SWT - metodo = 5
#SWT = pf.fusao(IM, 5, NUM_RAIOS)
## Testing fusion using DWT - metodo = 6
#DWT = pf.fusao(IM, 6, NUM_RAIOS)
#
print("Calculating metrics and writing to a file:")
if opcao == 1:
    # Def Ground reference
    GR = pb.GT_flev_roi_01_bresenham(nrows, ncols)
    # Hausdorff Distance
    d = np.zeros(ncanal)
    for i in range(ncanal):
        d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
    #
    df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
    # Print to file
    directory = './figure/'
    file = 'hausdorf_metrics_intensities_channels_flev.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(d[i])
            f.write(value)
            f.write('\n')
    file = 'hausdorf_metrics_fusion_flev.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(df))
    file = 'hausdorf_metrics_fusion_threshold_flev.txt'
    df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(df_threshold))
    file = 'principal_comp_flev.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(comp_princ))
    file = 'principal_comp_flev_crop.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(comp_princ_crop))
    file = 'matrix_cov_flev.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(MCOVAR[i, :])
            f.write(value)
        f.write('\n')
    file = 'matrix_cov_crop_flev.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(MCOVAR_crop[i, :])
            f.write(value)
        f.write('\n')
    file = 'zeros_count_intensities_channels_flev.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(zeros_count_channels[:, i])
            f.write(value)
            f.write('\n')
    file = 'zeros_count_arg_min_intensities_channels_flev.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            aux = np.zeros(2)
            aux[0] = str(np.argmin(zeros_count_channels[:, i]))
            aux[1] = str(np.min(zeros_count_channels[:, i]))
            value =str(aux)
            f.write(value)
            f.write('\n')
elif opcao == 2:
    # Def Ground reference
    GR = pb.GT_sf_roi_01_bresenham(nrows, ncols)
    # Hausdorff Distance
    d = np.zeros(ncanal)
    for i in range(ncanal):
        d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
    #
    df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
    # Print to file
    directory = './figure/'
    file = 'hausdorf_metrics_intensities_channels_sf.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(d[i])
            f.write(value)
            f.write('\n')
    file = 'hausdorf_metrics_fusion_sf.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(df))
    file = 'hausdorf_metrics_fusion_threshold_sf.txt'
    df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(df_threshold))
    file = 'principal_comp_sf.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(comp_princ))
    file = 'principal_comp_sf_crop.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(comp_princ_crop))
    file = 'matrix_cov_sf.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(MCOVAR[i, :])
            f.write(value)
        f.write('\n')
    file = 'matrix_cov_crop_sf.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(MCOVAR_crop[i, :])
            f.write(value)
        f.write('\n')
    file = 'zeros_count_intensities_channels_sf.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(zeros_count_channels[:, i])
            f.write(value)
            f.write('\n')
    file = 'zeros_count_arg_min_intensities_channels_sf.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            aux = np.zeros(2)
            aux[0] = str(np.argmin(zeros_count_channels[:, i]))
            aux[1] = str(np.min(zeros_count_channels[:, i]))
            value =str(aux)
            f.write(value)
            f.write('\n')
elif opcao == 3:
    # Def Ground reference
    GR = pb.GT_sim_flor(nrows, ncols)
    pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_sim_image")
    # Hausdorff Distance
    d = np.zeros(ncanal)
    for i in range(ncanal):
        d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
        #
    df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
    # Print to file
    directory = './figure/'
    file = 'hausdorf_metrics_intensities_channels_sim.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(d[i])
            f.write(value)
            f.write('\n')
    file = 'hausdorf_metrics_fusion_sim.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(df))
    file = 'hausdorf_metrics_fusion_threshold_sim.txt'
    df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(df_threshold))
    file = 'principal_comp_sim.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        f.write(str(comp_princ))
    file = 'matrix_cov_sim.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(MCOVAR[i, :])
            f.write(value)
        f.write('\n')
    file = 'zeros_count_intensities_channels_sim.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            value =str(zeros_count_channels[:, i])
            f.write(value)
            f.write('\n')
    file = 'zeros_count_arg_min_intensities_channels_sim.txt'
    file_path = os.path.join(directory, file)
    with open(file_path, 'w') as f:
        for i in range(ncanal):
            aux = np.zeros(2)
            aux[0] = str(np.argmin(zeros_count_channels[:, i]))
            aux[1] = str(np.min(zeros_count_channels[:, i]))
            value =str(aux)
            f.write(value)
            f.write('\n')
elif opcao == 4:
    if opcao_subscene == 1:
        # Def Ground reference
        GR = pb.GR_sub_scene_07(nrows, ncols, PI_AUX)
        pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_sub_scene_07")
        # Hausdorff Distance
        d = np.zeros(ncanal)
        for i in range(ncanal):
            d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
        #
        df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
        # Print to file
        directory = './figure/'
        file = 'hausdorf_metrics_intensities_channels_subscene_07.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(d[i])
                f.write(value)
                f.write('\n')
            file = 'hausdorf_metrics_fusion_subscene_07.txt'
            file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df))
        file = 'hausdorf_metrics_fusion_threshold_subscene_07.txt'
        df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df_threshold))
        file = 'principal_comp_subscene_07.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(comp_princ))
        file = 'matrix_cov_subscene_07.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(MCOVAR[i, :])
                f.write(value)
            f.write('\n')
        file = 'zeros_count_intensities_channels_subscene_07.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(zeros_count_channels[:, i])
                f.write(value)
                f.write('\n')
        file = 'zeros_count_arg_min_intensities_channels_subscene_07.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                aux = np.zeros(2)
                aux[0] = str(np.argmin(zeros_count_channels[:, i]))
                aux[1] = str(np.min(zeros_count_channels[:, i]))
                value =str(aux)
                f.write(value)
                f.write('\n')
    elif opcao_subscene == 2:
        # Def Ground reference
        GR = pb.GR_sub_scene_10(nrows, ncols, PI_AUX)
        pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_sub_scene_10")
        # Hausdorff Distance
        d = np.zeros(ncanal)
        for i in range(ncanal):
            d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
        #
        df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
        # Print to file
        directory = './figure/'
        file = 'hausdorf_metrics_intensities_channels_subscene_10.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(d[i])
                f.write(value)
                f.write('\n')
            file = 'hausdorf_metrics_fusion_subscene_10.txt'
            file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df))
        file = 'hausdorf_metrics_fusion_threshold_subscene_10.txt'
        df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df_threshold))
        file = 'principal_comp_subscene_10.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(comp_princ))
        file = 'matrix_cov_subscene_10.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(MCOVAR[i, :])
                f.write(value)
            f.write('\n')
        file = 'zeros_count_intensities_channels_subscene_10.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(zeros_count_channels[:, i])
                f.write(value)
                f.write('\n')
        file = 'zeros_count_arg_min_intensities_channels_subscene_10.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                aux = np.zeros(2)
                aux[0] = str(np.argmin(zeros_count_channels[:, i]))
                aux[1] = str(np.min(zeros_count_channels[:, i]))
                value =str(aux)
                f.write(value)
                f.write('\n')
    else:
        print("txt print end to subscene image")
elif opcao == 5:
    if opcao_subscene == 1:
        # Def Ground reference
        GR = pb.GR_la_april_2009_image_1(nrows, ncols, PI_AUX)
        #pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_la_april_2009_img_01")
        # Hausdorff Distance
        d = np.zeros(ncanal)
        for i in range(ncanal):
            d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
        #
        df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
        # Print to file
        directory = './figure/'
        file = 'hausdorf_metrics_intensities_channels_la_april_2009_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(d[i])
                f.write(value)
                f.write('\n')
            file = 'hausdorf_metrics_fusion_la_april_2009_img_01.txt'
            file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df))
        file = 'hausdorf_metrics_fusion_threshold_la_april_2009_img_01.txt'
        df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df_threshold))
        file = 'principal_comp_la_april_2009_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(comp_princ))
        file = 'matrix_cov_la_april_2009_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(MCOVAR[i, :])
                f.write(value)
            f.write('\n')
        file = 'zeros_count_intensities_la_april_2009_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(zeros_count_channels[:, i])
                f.write(value)
                f.write('\n')
        file = 'zeros_count_arg_min_intensities_channels_la_april_2009_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                aux = np.zeros(2)
                aux[0] = str(np.argmin(zeros_count_channels[:, i]))
                aux[1] = str(np.min(zeros_count_channels[:, i]))
                value =str(aux)
                f.write(value)
                f.write('\n')
    elif opcao_subscene == 2:
        # Def Ground reference
        GR = pb.GR_la_may_2015_image_1(nrows, ncols, PI_AUX)
        #pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_la_may_2015_img_01")
        # Hausdorff Distance
        d = np.zeros(ncanal)
        for i in range(ncanal):
            d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
        #
        df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
        # Print to file
        directory = './figure/'
        file = 'hausdorf_metrics_intensities_channels_la_may_2015_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(d[i])
                f.write(value)
                f.write('\n')
            file = 'hausdorf_metrics_fusion_la_may_2015_img_01.txt'
            file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df))
        file = 'hausdorf_metrics_fusion_threshold_la_may_2015_img_01.txt'
        df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df_threshold))
        file = 'principal_comp_la_may_2015_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(comp_princ))
        file = 'matrix_cov_la_may_2015_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(MCOVAR[i, :])
                f.write(value)
            f.write('\n')
        file = 'zeros_count_intensities_channels_la_may_2015_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(zeros_count_channels[:, i])
                f.write(value)
                f.write('\n')
        file = 'zeros_count_arg_min_intensities_channels_la_may_2015_img_01.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                aux = np.zeros(2)
                aux[0] = str(np.argmin(zeros_count_channels[:, i]))
                aux[1] = str(np.min(zeros_count_channels[:, i]))
                value =str(aux)
                f.write(value)
                f.write('\n')
    else:
        print("txt print end to LA image")
else:
    if opcao_subscene == 1:
        # Def Ground reference
        GR = pb.GR_la_april_2009_image_2(nrows, ncols, PI_AUX)
        pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_la_april_2009_img_02")
        # Hausdorff Distance
        d = np.zeros(ncanal)
        for i in range(ncanal):
            d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
        #
        df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
        # Print to file
        directory = './figure/'
        file = 'hausdorf_metrics_intensities_channels_la_april_2009_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(d[i])
                f.write(value)
                f.write('\n')
            file = 'hausdorf_metrics_fusion_la_april_2009_img_02.txt'
            file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df))
        file = 'hausdorf_metrics_fusion_threshold_la_april_2009_img_02.txt'
        df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df_threshold))
        file = 'principal_comp_la_april_2009_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(comp_princ))
        file = 'matrix_cov_la_april_2009_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(MCOVAR[i, :])
                f.write(value)
            f.write('\n')
        file = 'zeros_count_intensities_la_april_2009_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(zeros_count_channels[:, i])
                f.write(value)
                f.write('\n')
        file = 'zeros_count_arg_min_intensities_channels_la_april_2009_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                aux = np.zeros(2)
                aux[0] = str(np.argmin(zeros_count_channels[:, i]))
                aux[1] = str(np.min(zeros_count_channels[:, i]))
                value =str(aux)
                f.write(value)
                f.write('\n')
    elif opcao_subscene == 2:
        # Def Ground reference
        GR = pb.GR_la_may_2015_image_2(nrows, ncols, PI_AUX)
        pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_la_may_2015_img_02.txt")
        # Hausdorff Distance
        d = np.zeros(ncanal)
        for i in range(ncanal):
            d[i] = pm.hausdorff_distance_aab(GR, IM[:, :, i], nrows, ncols)
        #
        df = pm.hausdorff_distance_aab(GR, ROC, nrows, ncols)
        # Print to file
        directory = './figure/'
        file = 'hausdorf_metrics_intensities_channels_la_may_2015_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(d[i])
                f.write(value)
                f.write('\n')
            file = 'hausdorf_metrics_fusion_la_may_2015_img_02.txt'
            file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df))
        file = 'hausdorf_metrics_fusion_threshold_la_may_2015_img_02.txt'
        df_threshold = pm.hausdorff_distance_aab(GR, ROC_THRESHOLD, nrows, ncols)
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(df_threshold))
        file = 'principal_comp_la_may_2015_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(str(comp_princ))
        file = 'matrix_cov_la_may_2015_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(MCOVAR[i, :])
                f.write(value)
            f.write('\n')
        file = 'zeros_count_intensities_channels_la_may_2015_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                value =str(zeros_count_channels[:, i])
                f.write(value)
                f.write('\n')
        file = 'zeros_count_arg_min_intensities_channels_la_may_2015_img_02.txt'
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            for i in range(ncanal):
                aux = np.zeros(2)
                aux[0] = str(np.argmin(zeros_count_channels[:, i]))
                aux[1] = str(np.min(zeros_count_channels[:, i]))
                value =str(aux)
                f.write(value)
                f.write('\n')
    else:
        print("txt print end to LA image")

#
print("Select like the image can be showed:")
print("1. Print on the screen")
print("2. Print on the pdf files")
opcao1=int(input("Type the option:"))

if opcao1 == 1:
    #PI = pplt.image_contrast_brightness(PI)
    #GR = pb.GR_sub_scene_07(nrows, ncols, PI_AUX)
    #pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_sub_scene_07")
    #GR = pb.GR_sub_scene_10(nrows, ncols, PI_AUX)
    #pplt.show_image_pauli_to_file_set(PI_AUX, GR, "GR_sub_scene_10")
    #pb.GT_sub_scene_07(nrows, ncols, PI)
    # Print the evidences fusion
    for i in range(ncanal):
        PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, i)
    #
    #pf.show_fusion_evidence(PI, nrows, ncols, MEDIA, img_rt)
    #pf.show_fusion_evidence(PI, nrows, ncols, PCA, img_rt)
    pf.show_fusion_evidence(PI, nrows, ncols, ROC, img_rt)
    #pf.show_fusion_evidence(PI, nrows, ncols, DWT, img_rt)
    #pf.show_fusion_evidence(PI, nrows, ncols, SWT, img_rt)
    #pf.show_fusion_evidence(PI, nrows, ncols, SVD, img_rt)
else:
    if opcao == 1:
        # Plot a pdf gray  to each channel
        image_name_flev_gray = np.zeros(ncanal, dtype='U256')
        image_name_flev_gray[0] = "FlevChh_gray"
        image_name_flev_gray[1] = "FlevChv_gray"
        image_name_flev_gray[2] = "FlevCvv_gray"
        image_name_flev_gray[3] = "FlevSpan_gray"
        image_name_flev_gray[4] = "FlevRatioHHHVRoi01_gray"
        image_name_flev_gray[5] = "FlevRatioHHVVRoi01_gray"
        image_name_flev_gray[6] = "FlevRatioHVVVRoi01_gray"
        image_name_flev_gray[7] = "FlevRatioHVHHRoi01_gray"
        image_name_flev_gray[8] = "FlevRatioVVHVRoi01_gray"
        image_name_flev_gray[9] = "FlevRatioVVHHRoi01_gray"
        #
        pplt.show_image_to_file(img[:, :, 0], nrows, ncols, image_name_flev_gray[0])
        pplt.show_image_to_file(img[:, :, 1], nrows, ncols, image_name_flev_gray[1])
        pplt.show_image_to_file(img[:, :, 2], nrows, ncols, image_name_flev_gray[2])
        pplt.show_image_to_file(img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2] , nrows, ncols, image_name_flev_gray[3])
        channel_1 = 0
        channel_2 = 1
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_flev_gray[4])
        channel_1 = 0
        channel_2 = 2
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_flev_gray[5])
        channel_1 = 1
        channel_2 = 2
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_flev_gray[6])
        channel_1 = 1
        channel_2 = 0
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_flev_gray[7])
        channel_1 = 2
        channel_2 = 1
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_flev_gray[8])
        channel_1 = 2
        channel_2 = 0
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_flev_gray[9])
        # # Plot a pdf to each channel
        print("Select Full Image or Crop Image:")
        print("10. Full image")
        print("20. Crop image")
        opcao_crop=int(input("Type the option:"))
        if opcao_crop == 10:
            #PI_AUX= pplt.image_contrast_brightness(PI_AUX)
            # Define name to each channel
            image_name_flev = np.zeros(ncanal, dtype='U256')
            image_name_flev[0] = "FlevEvChhRoi01"
            image_name_flev[1] = "FlevEvChvRoi01"
            image_name_flev[2] = "FlevEvCvvRoi01"
            image_name_flev[3] = "FlevEvSpanRoi01"
            image_name_flev[4] = "FlevEvRatioHHHVRoi01"
            image_name_flev[5] = "FlevEvRatioHHVVRoi01"
            image_name_flev[6] = "FlevEvRatioHVVVRoi01"
            image_name_flev[7] = "FlevEvRatioHVHHRoi01"
            image_name_flev[8] = "FlevEvRatioVVHVRoi01"
            image_name_flev[9] = "FlevEvRatioVVHHRoi01"
            # Define name to evidences fusion
            image_name_ROC_flev = "FlevFusionRodRoi01"
            for i in range(ncanal):
                pplt.show_image_pauli_to_file(PI_AUX, IM[:, :, i], nrows, ncols, img_rt, image_name_flev[i])
            # Print the evidences fusion
            pplt.show_image_pauli_to_file(PI_AUX, ROC, nrows, ncols, img_rt, image_name_ROC_flev)
        else:
            IM_crop  = pb.crop_image_flev(IM, xlim, ylim)
            ROC_crop = pb.crop_image_2d_flev(ROC, xlim, ylim)
            ROC_crop_threshold = pb.crop_image_2d_flev(ROC_THRESHOLD, xlim, ylim)
            PI_crop  = pb.crop_image_flev(PI_AUX, xlim, ylim)
            #PI_crop  = pplt.image_contrast_brightness(PI_crop)
            # Define name to each channel
            image_name_flev_crop = np.zeros(ncanal, dtype='U256')
            image_name_flev_crop[0] = "FlevEvChhRoi01_crop"
            image_name_flev_crop[1] = "FlevEvChvRoi01_crop"
            image_name_flev_crop[2] = "FlevEvCvvRoi01_crop"
            image_name_flev_crop[3] = "FlevEvSpanRoi01_crop"
            image_name_flev_crop[4] = "FlevEvRatioHHHVRoi01_crop"
            image_name_flev_crop[5] = "FlevEvRatioHHVVRoi01_crop"
            image_name_flev_crop[6] = "FlevEvRatioHVVVRoi01_crop"
            image_name_flev_crop[7] = "FlevEvRatioHVHHRoi01_crop"
            image_name_flev_crop[8] = "FlevEvRatioVVHVRoi01_crop"
            image_name_flev_crop[9] = "FlevEvRatioVVHHRoi01_crop"
            # Define name to evidences fusion
            image_name_ROC_crop = "FlevFusionRocRoi01_crop"
            image_name_ROC_crop_threshold = "FlevFusionRocRoi01_crop_threshold"
            #
            for i in range(ncanal):
                pplt.show_image_pauli_to_file_set(PI_crop, IM_crop[:, :, i] , image_name_flev_crop[i])
            # Print the evidences fusion
            pplt.show_image_pauli_to_file_set(PI_crop, ROC_crop, image_name_ROC_crop)
            pplt.show_image_pauli_to_file_set(PI_crop, ROC_crop_threshold, image_name_ROC_crop_threshold)
        #
    elif opcao == 2:
        # Plot a pdf gray to each channel
        image_name_sf_gray = np.zeros(ncanal, dtype='U256')
        image_name_sf_gray[0] = "SfChh_gray"
        image_name_sf_gray[1] = "SfChv_gray"
        image_name_sf_gray[2] = "SfCvv_gray"
        image_name_sf_gray[3] = "SfSpan_gray"
        image_name_sf_gray[4] = "SfRatioHHHVRoi01_gray"
        image_name_sf_gray[5] = "SfRatioHHVVRoi01_gray"
        image_name_sf_gray[6] = "SfRatioHVVVRoi01_gray"
        image_name_sf_gray[7] = "SfRatioHVHHRoi01_gray"
        image_name_sf_gray[8] = "SfRatioVVHVRoi01_gray"
        image_name_sf_gray[9] = "SfRatioVVHHRoi01_gray"
        #
        pplt.show_image_to_file(img[:, :, 0], nrows, ncols, image_name_sf_gray[0])
        pplt.show_image_to_file(img[:, :, 1], nrows, ncols, image_name_sf_gray[1])
        pplt.show_image_to_file(img[:, :, 2], nrows, ncols, image_name_sf_gray[2])
        pplt.show_image_to_file(img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2] , nrows, ncols, image_name_sf_gray[3])
        channel_1 = 0
        channel_2 = 1
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sf_gray[4])
        channel_1 = 0
        channel_2 = 2
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sf_gray[5])
        channel_1 = 1
        channel_2 = 2
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sf_gray[6])
        channel_1 = 1
        channel_2 = 0
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sf_gray[7])
        channel_1 = 2
        channel_2 = 1
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sf_gray[8])
        channel_1 = 2
        channel_2 = 0
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sf_gray[9])
        # Plot a pdf to each channel
        # Crop image to print
        # # Plot a pdf to each channel
        print("Select Full Image or Crop Image:")
        print("10. Full image")
        print("20. Crop image")
        opcao_crop=int(input("Type the option:"))
        if opcao_crop == 10:
            # Define name to each channel
            #PI_AUX= pplt.image_contrast_brightness(PI_AUX)
            image_name_sf = np.zeros(ncanal, dtype='U256')
            image_name_sf[0] = "SfEvChhRoi01"
            image_name_sf[1] = "SfEvChvRoi01"
            image_name_sf[2] = "SfEvCvvRoi01"
            image_name_sf[3] = "SfEvSpanRoi01"
            image_name_sf[4] = "SfEvRatioHHHVRoi01"
            image_name_sf[5] = "SfEvRatioHHVVRoi01"
            image_name_sf[6] = "SfEvRatioHVVVRoi01"
            image_name_sf[7] = "SfEvRatioHVHHRoi01"
            image_name_sf[8] = "SfEvRatioVVHVRoi01"
            image_name_sf[9] = "SfEvRatioVVHHRoi01"
            # Define name to evidences fusion
            image_name_ROC_sf = "SfFusionRocRoi01"
            for i in range(ncanal):
                pplt.show_image_pauli_to_file(PI_AUX, IM[:, :, i], nrows, ncols, img_rt, image_name_sf[i])
            # Print the evidences fusion
            pplt.show_image_pauli_to_file(PI_AUX, ROC, nrows, ncols, img_rt, image_name_ROC_sf)
        else:
            IM_crop  = pb.crop_image_sf(IM, xlim, ylim)
            ROC_crop = pb.crop_image_2d_sf(ROC, xlim, ylim)
            ROC_crop_threshold = pb.crop_image_2d_sf(ROC_THRESHOLD, xlim, ylim)
            PI_crop  = pb.crop_image_sf(PI_AUX, xlim, ylim)
            #PI_crop  = pplt.image_contrast_brightness(PI_crop)
            # Define name to each channel
            image_name_sf_crop = np.zeros(ncanal, dtype='U256')
            image_name_sf_crop[0] = "SfEvChhRoi01_crop"
            image_name_sf_crop[1] = "SfEvChvRoi01_crop"
            image_name_sf_crop[2] = "SfEvCvvRoi01_crop"
            image_name_sf_crop[3] = "SfEvSpanRoi01_crop"
            image_name_sf_crop[4] = "SfEvRatioHHHVRoi01_crop"
            image_name_sf_crop[5] = "SfEvRatioHHVVRoi01_crop"
            image_name_sf_crop[6] = "SfEvRatioHVVVRoi01_crop"
            image_name_sf_crop[7] = "SfEvRatioHVHHRoi01_crop"
            image_name_sf_crop[8] = "SfEvRatioVVHVRoi01_crop"
            image_name_sf_crop[9] = "SfEvRatioVVHHRoi01_crop"
            # Define name to evidences fusion
            image_name_ROC_crop = "SfFusionRocRoi01_crop"
            image_name_ROC_crop_gray_threshold = "SfFusionRocRoi01_crop_threshold"
            #
            for i in range(ncanal):
                #pplt.show_image_pauli_to_file_set(PI_crop1, IM_crop[:, :, i] , image_name_sf_crop[i])
                pplt.show_image_pauli_to_file_set(PI_crop, IM_crop[:, :, i], image_name_sf_crop[i])
            # Print the evidences fusion
            pplt.show_image_pauli_to_file_set(PI_crop, ROC_crop, image_name_ROC_crop)
            pplt.show_image_pauli_to_file_set(PI_crop, ROC_crop_threshold, image_name_ROC_crop_gray_threshold)
    elif opcao == 3:
        # Plot a pdf gray to each channel
        image_name_sim_gray = np.zeros(ncanal, dtype='U256')
        image_name_sim_gray[0] = "SimChh_gray"
        image_name_sim_gray[1] = "SimChv_gray"
        image_name_sim_gray[2] = "SimCvv_gray"
        image_name_sim_gray[3] = "SimSpan_gray"
        image_name_sim_gray[4] = "SimRatioHHHVRoi01_gray"
        image_name_sim_gray[5] = "SimRatioHHVVRoi01_gray"
        image_name_sim_gray[6] = "SimRatioHVVVRoi01_gray"
        image_name_sim_gray[7] = "SimRatioHVHHRoi01_gray"
        image_name_sim_gray[8] = "SimRatioVVHVRoi01_gray"
        image_name_sim_gray[9] = "SimRatioVVHHRoi01_gray"
        #
        pplt.show_image_to_file(img[:, :, 0], nrows, ncols, image_name_sim_gray[0])
        pplt.show_image_to_file(img[:, :, 1], nrows, ncols, image_name_sim_gray[1])
        pplt.show_image_to_file(img[:, :, 2], nrows, ncols, image_name_sim_gray[2])
        pplt.show_image_to_file(img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2] , nrows, ncols, image_name_sim_gray[3])
        channel_1 = 0
        channel_2 = 1
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sim_gray[4])
        channel_1 = 0
        channel_2 = 2
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sim_gray[5])
        channel_1 = 1
        channel_2 = 2
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sim_gray[6])
        channel_1 = 1
        channel_2 = 0
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sim_gray[7])
        channel_1 = 2
        channel_2 = 1
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sim_gray[8])
        channel_1 = 2
        channel_2 = 0
        pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_sim_gray[9])
        # Plot a pdf to each channel
        # Crop image to print
        # # Plot a pdf to each channel
        print("Select Full Image or Crop Image:")
        print("10. Full image")
        print("20. Crop image - To simulated image is not necessary to choose - Insert 10")
        opcao_crop=int(input("Type the option:"))
        if opcao_crop == 10:
            # Define name to each channel
            #PI_AUX= pplt.image_contrast_brightness(PI_AUX)
            image_name_sim = np.zeros(ncanal, dtype='U256')
            image_name_sim[0] = "SimEvChhRoi01"
            image_name_sim[1] = "SimEvChvRoi01"
            image_name_sim[2] = "SimEvCvvRoi01"
            image_name_sim[3] = "SimEvSpanRoi01"
            image_name_sim[4] = "SimEvRatioHHHVRoi01"
            image_name_sim[5] = "SimEvRatioHHVVRoi01"
            image_name_sim[6] = "SimEvRatioHVVVRoi01"
            image_name_sim[7] = "SimEvRatioHVHHRoi01"
            image_name_sim[8] = "SimEvRatioVVHVRoi01"
            image_name_sim[9] = "SimEvRatioVVHHRoi01"
            # Define name to evidences fusion
            image_name_ROC_sim = "SimFusionRocRoi01"
            image_name_ROC_sim_threshold = "SimFusionRocRoi01_threshold"
            for i in range(ncanal):
                pplt.show_image_pauli_to_file_set(PI_AUX, IM[:, :, i], image_name_sim[i])
            # Print the evidences fusion
            pplt.show_image_pauli_to_file_set(PI_AUX, ROC, image_name_ROC_sim)
            pplt.show_image_pauli_to_file_set(PI_AUX, ROC_THRESHOLD, image_name_ROC_sim_threshold)
    elif opcao == 4:
        if opcao_subscene == 1:
        # Plot a pdf gray to each channel
            image_name_dpb_gray = np.zeros(ncanal, dtype='U256')
            image_name_dpb_gray[0] = "DpbChh_sc07_gray"
            image_name_dpb_gray[1] = "DpbChv_sc07_gray"
            image_name_dpb_gray[2] = "DpbCvv_sc07_gray"
            image_name_dpb_gray[3] = "DpbSpan_sc07_gray"
            image_name_dpb_gray[4] = "DpbRatioHHHVRoi01_sc07_gray"
            image_name_dpb_gray[5] = "DpbRatioHHVVRoi01_sc07_gray"
            image_name_dpb_gray[6] = "DpbRatioHVVVRoi01_sc07_gray"
            image_name_dpb_gray[7] = "DpbRatioHVHHRoi01_sc07_gray"
            image_name_dpb_gray[8] = "DpbRatioVVHVRoi01_sc07_gray"
            image_name_dpb_gray[9] = "DpbRatioVVHHRoi01_sc07_gray"
            #
            pplt.show_image_to_file(img[:, :, 0], nrows, ncols, image_name_dpb_gray[0])
            pplt.show_image_to_file(img[:, :, 1], nrows, ncols, image_name_dpb_gray[1])
            pplt.show_image_to_file(img[:, :, 2], nrows, ncols, image_name_dpb_gray[2])
            pplt.show_image_to_file(img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2] , nrows, ncols, image_name_dpb_gray[3])
            channel_1 = 0
            channel_2 = 1
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[4])
            channel_1 = 0
            channel_2 = 2
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[5])
            channel_1 = 1
            channel_2 = 2
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[6])
            channel_1 = 1
            channel_2 = 0
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[7])
            channel_1 = 2
            channel_2 = 1
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[8])
            channel_1 = 2
            channel_2 = 0
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[9])
            # Plot a pdf to each channel
            # Crop image to print
            # # Plot a pdf to each channel
            print("Select Full Image or Crop Image:")
            print("10. Full image")
            print("20. Crop image - To data P band image is not necessary to choose - Insert 10")
            opcao_crop=int(input("Type the option:"))
            if opcao_crop == 10:
                # Define name to each channel
                #PI_AUX= pplt.image_contrast_brightness(PI_AUX)
                image_name_dpb = np.zeros(ncanal, dtype='U256')
                image_name_dpb[0] = "DpbEvChhRoi01_sc07"
                image_name_dpb[1] = "DpbEvChvRoi01_sc07"
                image_name_dpb[2] = "DpbEvCvvRoi01_sc07"
                image_name_dpb[3] = "DpbEvSpanRoi01_sc07"
                image_name_dpb[4] = "DpbEvRatioHHHVRoi01_sc07"
                image_name_dpb[5] = "DpbEvRatioHHVVRoi01_sc07"
                image_name_dpb[6] = "DpbEvRatioHVVVRoi01_sc07"
                image_name_dpb[7] = "DpbEvRatioHVHHRoi01_sc07"
                image_name_dpb[8] = "DpbEvRatioVVHVRoi01_sc07"
                image_name_dpb[9] = "DpbEvRatioVVHHRoi01_sc07"
                # Define name to evidences fusion
                image_name_ROC_dpb = "DpbFusionRocRoi01_sc07"
                image_name_ROC_dpb_threshold = "DpbFusionRocRoi01_sc07_threshold"
                for i in range(ncanal):
                    pplt.show_image_pauli_to_file_set(PI_AUX, IM[:, :, i], image_name_dpb[i])
            # Print the evidences fusion
                pplt.show_image_pauli_to_file_set(PI_AUX, ROC, image_name_ROC_dpb)
                pplt.show_image_pauli_to_file_set(PI_AUX, ROC_THRESHOLD, image_name_ROC_dpb_threshold)
        else:
            # Plot a pdf gray to each channel
                image_name_dpb_gray = np.zeros(ncanal, dtype='U256')
                image_name_dpb_gray[0] = "DpbChh_sc10_gray"
                image_name_dpb_gray[1] = "DpbChv_sc10_gray"
                image_name_dpb_gray[2] = "DpbCvv_sc10_gray"
                image_name_dpb_gray[3] = "DpbSpan_sc10_gray"
                image_name_dpb_gray[4] = "DpbRatioHHHVRoi01_sc10_gray"
                image_name_dpb_gray[5] = "DpbRatioHHVVRoi01_sc10_gray"
                image_name_dpb_gray[6] = "DpbRatioHVVVRoi01_sc10_gray"
                image_name_dpb_gray[7] = "DpbRatioHVHHRoi01_sc10_gray"
                image_name_dpb_gray[8] = "DpbRatioVVHVRoi01_sc10_gray"
                image_name_dpb_gray[9] = "DpbRatioVVHHRoi01_sc10_gray"
                #
                pplt.show_image_to_file(img[:, :, 0], nrows, ncols, image_name_dpb_gray[0])
                pplt.show_image_to_file(img[:, :, 1], nrows, ncols, image_name_dpb_gray[1])
                pplt.show_image_to_file(img[:, :, 2], nrows, ncols, image_name_dpb_gray[2])
                pplt.show_image_to_file(img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2] , nrows, ncols, image_name_dpb_gray[3])
                channel_1 = 0
                channel_2 = 1
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[4])
                channel_1 = 0
                channel_2 = 2
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[5])
                channel_1 = 1
                channel_2 = 2
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[6])
                channel_1 = 1
                channel_2 = 0
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[7])
                channel_1 = 2
                channel_2 = 1
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[8])
                channel_1 = 2
                channel_2 = 0
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[9])
                # Plot a pdf to each channel
                # Crop image to print
                # # Plot a pdf to each channel
                print("Select Full Image or Crop Image:")
                print("10. Full image")
                print("20. Crop image - To data P band image is not necessary to choose - Insert 10")
                opcao_crop=int(input("Type the option:"))
                if opcao_crop == 10:
                    # Define name to each channel
                    #PI_AUX= pplt.image_contrast_brightness(PI_AUX)
                    image_name_dpb = np.zeros(ncanal, dtype='U256')
                    image_name_dpb[0] = "DpbEvChhRoi01_sc10"
                    image_name_dpb[1] = "DpbEvChvRoi01_sc10"
                    image_name_dpb[2] = "DpbEvCvvRoi01_sc10"
                    image_name_dpb[3] = "DpbEvSpanRoi01_sc10"
                    image_name_dpb[4] = "DpbEvRatioHHHVRoi01_sc10"
                    image_name_dpb[5] = "DpbEvRatioHHVVRoi01_sc10"
                    image_name_dpb[6] = "DpbEvRatioHVVVRoi01_sc10"
                    image_name_dpb[7] = "DpbEvRatioHVHHRoi01_sc10"
                    image_name_dpb[8] = "DpbEvRatioVVHVRoi01_sc10"
                    image_name_dpb[9] = "DpbEvRatioVVHHRoi01_sc10"
                    # Define name to evidences fusion
                    image_name_ROC_dpb = "DpbFusionRocRoi01_sc10"
                    image_name_ROC_dpb_threshold = "DpbFusionRocRoi01_sc10_threshold"
                    for i in range(ncanal):
                        pplt.show_image_pauli_to_file_set(PI_AUX, IM[:, :, i], image_name_dpb[i])
                # Print the evidences fusion
                    pplt.show_image_pauli_to_file_set(PI_AUX, ROC, image_name_ROC_dpb)
                    pplt.show_image_pauli_to_file_set(PI_AUX, ROC_THRESHOLD, image_name_ROC_dpb_threshold)
    elif opcao == 5:
        if opcao_subscene == 1:
        # Plot a pdf gray to each channel
            image_name_dpb_gray = np.zeros(ncanal, dtype='U256')
            image_name_dpb_gray[0] = "la_img1_april_2009_Chh_gray"
            image_name_dpb_gray[1] = "la_img1_april_2009_Chv_gray"
            image_name_dpb_gray[2] = "la_img1_april_2009_Cvv_gray"
            image_name_dpb_gray[3] = "la_img1_april_2009_span_gray"
            image_name_dpb_gray[4] = "la_img1_april_2009_RatioHHHVRoi01_gray"
            image_name_dpb_gray[5] = "la_img1_april_2009_RatioHHVVRoi01_gray"
            image_name_dpb_gray[6] = "la_img1_april_2009_RatioHVVVRoi01_gray"
            image_name_dpb_gray[7] = "la_img1_april_2009_RatioHVHHRoi01_gray"
            image_name_dpb_gray[8] = "la_img1_april_2009_RatioVVHVRoi01_gray"
            image_name_dpb_gray[9] = "la_img1_april_2009_RatioVVHHRoi01_gray"
            #
            pplt.show_image_to_file(img[:, :, 0], nrows, ncols, image_name_dpb_gray[0])
            pplt.show_image_to_file(img[:, :, 1], nrows, ncols, image_name_dpb_gray[1])
            pplt.show_image_to_file(img[:, :, 2], nrows, ncols, image_name_dpb_gray[2])
            pplt.show_image_to_file(img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2] , nrows, ncols, image_name_dpb_gray[3])
            channel_1 = 0
            channel_2 = 1
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[4])
            channel_1 = 0
            channel_2 = 2
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[5])
            channel_1 = 1
            channel_2 = 2
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[6])
            channel_1 = 1
            channel_2 = 0
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[7])
            channel_1 = 2
            channel_2 = 1
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[8])
            channel_1 = 2
            channel_2 = 0
            pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[9])
            # Plot a pdf to each channel
            # Crop image to print
            # # Plot a pdf to each channel
            print("Select Full Image or Crop Image:")
            print("10. Full image")
            print("20. Crop image - To data P band image is not necessary to choose - Insert 10")
            opcao_crop=int(input("Type the option:"))
            if opcao_crop == 10:
                # Define name to each channel
                #PI_AUX= pplt.image_contrast_brightness(PI_AUX)
                image_name_dpb = np.zeros(ncanal, dtype='U256')
                image_name_dpb[0] = "la_img1_april_2009_ChhRoi01"
                image_name_dpb[1] = "la_img1_april_2009_ChvRoi01"
                image_name_dpb[2] = "la_img1_april_2009_CvvRoi01"
                image_name_dpb[3] = "la_img1_april_2009_SpanRoi01"
                image_name_dpb[4] = "la_img1_april_2009_RatioHHHVRoi01"
                image_name_dpb[5] = "la_img1_april_2009_RatioHHVVRoi01"
                image_name_dpb[6] = "la_img1_april_2009_RatioHVVVRoi01"
                image_name_dpb[7] = "la_img1_april_2009_RatioHVHHRoi01"
                image_name_dpb[8] = "la_img1_april_2009_RatioVVHVRoi01"
                image_name_dpb[9] = "la_img1_april_2009_RatioVVHHRoi01"
                # Define name to evidences fusion
                image_name_ROC_dpb = "la_img1_april_2009_FusionRocRoi01"
                for i in range(ncanal):
                    pplt.show_image_pauli_to_file_set(PI_AUX, IM[:, :, i], image_name_dpb[i])
            # Print the evidences fusion
                pplt.show_image_pauli_to_file_set(PI_AUX, ROC, image_name_ROC_dpb)
            else:
                IM_crop  = pb.crop_image_la_img1(IM, xlim, ylim)
                #ROC_crop = pb.crop_image_2d_la_imag1(ROC, xlim, ylim)
                ROC_crop = pb.crop_image_2d_la_imag1(ROC, xlim, ylim)
                ROC_crop_threshold = pb.crop_image_2d_la_imag1(ROC_THRESHOLD, xlim, ylim)
                PI_crop  = pb.crop_image_la_img1(PI_AUX, xlim, ylim)
                #PI_crop  = pplt.image_contrast_brightness(PI_crop)
                # Define name to each channel
                image_name_crop = np.zeros(ncanal, dtype='U256')
                image_name_crop[0] = "la_img1_april_2009_ChhRoi01_crop"
                image_name_crop[1] = "la_img1_april_2009_ChvRoi01_crop"
                image_name_crop[2] = "la_img1_april_2009_CvvRoi01_crop"
                image_name_crop[3] = "la_img1_april_2009_SpanRoi01_crop"
                image_name_crop[4] = "la_img1_april_2009_RatioHHHVRoi01_crop"
                image_name_crop[5] = "la_img1_april_2009_RatioHHVVRoi01_crop"
                image_name_crop[6] = "la_img1_april_2009_RatioHVVVRoi01_crop"
                image_name_crop[7] = "la_img1_april_2009_RatioHVHHRoi01_crop"
                image_name_crop[8] = "la_img1_april_2009_RatioVVHVRoi01_crop"
                image_name_crop[9] = "la_img1_april_2009_RatioVVHHRoi01_crop"
                #
                image_name_crop_gray = np.zeros(ncanal, dtype='U256')
                image_name_crop_gray[0] = "la_img1_april_2009_ChhRoi01_crop_gray"
                image_name_crop_gray[1] = "la_img1_april_2009_ChvRoi01_crop_gray"
                image_name_crop_gray[2] = "la_img1_april_2009_CvvRoi01_crop_gray"
                image_name_crop_gray[3] = "la_img1_april_2009_SpanRoi01_crop_gray"
                image_name_crop_gray[4] = "la_img1_april_2009_RatioHHHVRoi01_crop_gray"
                image_name_crop_gray[5] = "la_img1_april_2009_RatioHHVVRoi01_crop_gray"
                image_name_crop_gray[6] = "la_img1_april_2009_RatioHVVVRoi01_crop_gray"
                image_name_crop_gray[7] = "la_img1_april_2009_RatioHVHHRoi01_crop_gray"
                image_name_crop_gray[8] = "la_img1_april_2009_RatioVVHVRoi01_crop_gray"
                image_name_crop_gray[9] = "la_img1_april_2009_RatioVVHHRoi01_crop_gray"
                # Define name to evidences fusion
                image_name_ROC_crop = "la_img1_april_2009_FusionRocRoi01_crop"
                image_name_ROC_crop_gray = "la_img1_april_2009_FusionRocRoi01_crop_gray"
                image_name_ROC_crop_gray_threshold = "la_img1_april_2009_FusionRocRoi01_crop_gray_threshold"
                #
                pplt.show_image_pauli_gray_to_file_set(PI_crop, IM_crop[:, :, 0], "teste_gray_evidence_color")
                for i in range(ncanal):
                    pplt.show_image_pauli_to_file_set(PI_crop, IM_crop[:, :, i] , image_name_crop[i])
                    pplt.show_image_pauli_gray_to_file_set(PI_crop, IM_crop[:, :, i], image_name_crop_gray[i])
                # Print the evidences fusion
                pplt.show_image_pauli_to_file_set(PI_crop, ROC_crop, image_name_ROC_crop)
                pplt.show_image_pauli_gray_to_file_set(PI_crop, ROC_crop, image_name_ROC_crop_gray)
                pplt.show_image_pauli_gray_to_file_set(PI_crop, ROC_crop_threshold, image_name_ROC_crop_gray_threshold)
        else:
            # Plot a pdf gray to each channel
                image_name_dpb_gray = np.zeros(ncanal, dtype='U256')
                image_name_dpb_gray[0] = "la_img1_may_2015_Chh_gray"
                image_name_dpb_gray[1] = "la_img1_may_2015_Chv_gray"
                image_name_dpb_gray[2] = "la_img1_may_2015_Cvv_gray"
                image_name_dpb_gray[3] = "la_img1_may_2015_span_gray"
                image_name_dpb_gray[4] = "la_img1_may_2015_RatioHHHVRoi01_gray"
                image_name_dpb_gray[5] = "la_img1_may_2015_RatioHHVVRoi01_gray"
                image_name_dpb_gray[6] = "la_img1_may_2015_RatioHVVVRoi01_gray"
                image_name_dpb_gray[7] = "la_img1_may_2015_RatioHVHHRoi01_gray"
                image_name_dpb_gray[8] = "la_img1_may_2015_RatioVVHVRoi01_gray"
                image_name_dpb_gray[9] = "la_img1_may_2015_RatioVVHHRoi01_gray"

                #
                pplt.show_image_to_file(img[:, :, 0], nrows, ncols, image_name_dpb_gray[0])
                pplt.show_image_to_file(img[:, :, 1], nrows, ncols, image_name_dpb_gray[1])
                pplt.show_image_to_file(img[:, :, 2], nrows, ncols, image_name_dpb_gray[2])
                pplt.show_image_to_file(img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2] , nrows, ncols, image_name_dpb_gray[3])
                channel_1 = 0
                channel_2 = 1
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[4])
                channel_1 = 0
                channel_2 = 2
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[5])
                channel_1 = 1
                channel_2 = 2
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[6])
                channel_1 = 1
                channel_2 = 0
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[7])
                channel_1 = 2
                channel_2 = 1
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[8])
                channel_1 = 2
                channel_2 = 0
                pplt.show_image_ratio_intensities_to_files(img, nrows, ncols, img_rt, channel_1, channel_2, image_name_dpb_gray[9])
                # Plot a pdf to each channel
                # Crop image to print
                # # Plot a pdf to each channel
                print("Select Full Image or Crop Image:")
                print("10. Full image")
                print("20. Crop image - To data P band image is not necessary to choose - Insert 10")
                opcao_crop=int(input("Type the option:"))
                if opcao_crop == 10:
                    # Define name to each channel
                    #PI_AUX= pplt.image_contrast_brightness(PI_AUX)
                    image_name_dpb = np.zeros(ncanal, dtype='U256')
                    image_name_dpb[0] = "la_img1_may_2015_ChhRoi01"
                    image_name_dpb[1] = "la_img1_may_2015_ChvRoi01"
                    image_name_dpb[2] = "la_img1_may_2015_CvvRoi01"
                    image_name_dpb[3] = "la_img1_may_2015_SpanRoi01"
                    image_name_dpb[4] = "la_img1_may_2015_RatioHHHVRoi01"
                    image_name_dpb[5] = "la_img1_may_2015_RatioHHVVRoi01"
                    image_name_dpb[6] = "la_img1_may_2015_RatioHVVVRoi01"
                    image_name_dpb[7] = "la_img1_may_2015_RatioHVHHRoi01"
                    image_name_dpb[8] = "la_img1_may_2015_RatioVVHVRoi01"
                    image_name_dpb[9] = "la_img1_may_2015_RatioVVHHRoi01"
                    #efine name to evidences fusion
                    #mage_name_ROC_dpb = "la_img1_may_2015_FusionRocRoi01"
                    for i in range(ncanal):
                        pplt.show_image_pauli_to_file_set(PI_AUX, IM[:, :, i], image_name_dpb[i])
                # Print the evidences fusion
                    pplt.show_image_pauli_to_file_set(PI_AUX, ROC, image_name_ROC_dpb)
                else:
                    IM_crop  = pb.crop_image_la_img1(IM, xlim, ylim)
                    #ROC_crop = pb.crop_image_2d_la_imag1(ROC, xlim, ylim)
                    ROC_crop = pb.crop_image_2d_la_imag1(ROC, xlim, ylim)
                    ROC_crop_threshold = pb.crop_image_2d_la_imag1(ROC_THRESHOLD, xlim, ylim)
                    PI_crop  = pb.crop_image_la_img1(PI_AUX, xlim, ylim)
                    #
                    image_name_crop = np.zeros(ncanal, dtype='U256')
                    image_name_crop[0] = "la_img1_may_2015_ChhRoi01_crop"
                    image_name_crop[1] = "la_img1_may_2015_ChvRoi01_crop"
                    image_name_crop[2] = "la_img1_may_2015_CvvRoi01_crop"
                    image_name_crop[3] = "la_img1_may_2015_SpanRoi01_crop"
                    image_name_crop[4] = "la_img1_may_2015_RatioHHHVRoi01_crop"
                    image_name_crop[5] = "la_img1_may_2015_RatioHHVVRoi01_crop"
                    image_name_crop[6] = "la_img1_may_2015_RatioHVVVRoi01_crop"
                    image_name_crop[7] = "la_img1_may_2015_RatioHVHHRoi01_crop"
                    image_name_crop[8] = "la_img1_may_2015_RatioVVHVRoi01_crop"
                    image_name_crop[9] = "la_img1_may_2015_RatioVVHHRoi01_crop"
                    # Define name to evidences fusion
                    #
                    image_name_crop_gray = np.zeros(ncanal, dtype='U256')
                    image_name_crop_gray[0] = "la_img1_may_2015_ChhRoi01_crop_gray"
                    image_name_crop_gray[1] = "la_img1_may_2015_ChvRoi01_crop_gray"
                    image_name_crop_gray[2] = "la_img1_may_2015_CvvRoi01_crop_gray"
                    image_name_crop_gray[3] = "la_img1_may_2015_SpanRoi01_crop_gray"
                    image_name_crop_gray[4] = "la_img1_may_2015_RatioHHHVRoi01_crop_gray"
                    image_name_crop_gray[5] = "la_img1_may_2015_RatioHHVVRoi01_crop_gray"
                    image_name_crop_gray[6] = "la_img1_may_2015_RatioHVVVRoi01_crop_gray"
                    image_name_crop_gray[7] = "la_img1_may_2015_RatioHVHHRoi01_crop_gray"
                    image_name_crop_gray[8] = "la_img1_may_2015_RatioVVHVRoi01_crop_gray"
                    image_name_crop_gray[9] = "la_img1_may_2015_RatioVVHHRoi01_crop_gray"
                    # Define name to evidences fusion
                    image_name_ROC_crop = "la_img1_may_2015_FusionRocRoi01_crop"
                    image_name_ROC_crop_gray = "la_may_2015_FusionRocRoi01_crop_gray"
                    image_name_ROC_crop_gray_threshold = "la_may_2015_FusionRocRoi01_crop_gray_threshold"
                    #
                    #pplt.show_image_pauli_gray_to_file_set(PI_crop, IM_crop[:, :, 0], "teste_gray_evidence_color")
                    for i in range(ncanal):
                        pplt.show_image_pauli_to_file_set(PI_crop, IM_crop[:, :, i] , image_name_crop[i])
                        pplt.show_image_pauli_gray_to_file_set(PI_crop, IM_crop[:, :, i], image_name_crop_gray[i])
                    # Print the evidences fusion
                    pplt.show_image_pauli_to_file_set(PI_crop, ROC_crop, image_name_ROC_crop)
                    pplt.show_image_pauli_gray_to_file_set(PI_crop, ROC_crop, image_name_ROC_crop_gray)
                    pplt.show_image_pauli_gray_to_file_set(PI_crop, ROC_crop_threshold, image_name_ROC_crop_gray_threshold)
                    #
