#!/usr/bin/ipython
import numpy as np
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, Tophat2DKernel
from math import *
import sys
from astropy.io import fits
import string
import os

##########
#SUPRHT  #
##########
# 2015: DEVELOPPED BY ARTHUR BRENA
# 2017: REVISED by L. MONTIER
#This program extracts the structures (filaments) of a given intensity map (.fits) and gives several outputs: 
#the extracted filaments map, the angle distribution of the filaments (theta_RHT) and its sigma (sigma_theta_RHT)

#print'_____________________________________________________________________________________________________________________________________________________________________\n|  ________________________________________________________________________________________________________________________________________________________________  |\n| |       _________________     _____       _____    ____________        _________________    ____________        _____       _____    _________________    _____  | |\n| |      /                /    /    /      /    /   /            \      /                /   /            \      /    /      /    /   /                /   /__  /  | |\n| |     /    ____________/    /    /      /    /   /    ________  \    /    ____________/   /    ________  \    /    /      /    /   /_____      _____/    __/ /   | |\n| |    /    /__________      /    /      /    /   /    /       /  /   /    /____           /    /       /  /   /    /______/    /         /     /         /_  /    | |\n| |    \___________    \    /    /      /    /   /     \______/  /   /    _____/          /     \______/  /   /    _______     /         /     /        ___/ /     | |\n| |   ____________/    /   /    /______/    /   /    /\_________/   /    /___________    /    /\     ____/   /    /      /    /         /     /         \___/      | |\n| |  /                /   /                /   /    /              /                /   /    /  \    \      /    /      /    /         /     /                     | |\n| | /________________/   /________________/   /____/              /________________/   /____/    \____\    /____/      /____/         /_____/                      | |\n| |________________________________________________________________________________________________________________________________________________________________| |\n|____________________________________________________________________________________________________________________________________________________________________|\n'

######################
#KERNEL INITIALISATION
######################
'''
ks = kernel_size is the diameter of the kernel inside which we apply the Hough Transform
bw = bar_width defines the width of the line on which we integrate the intensity
nd = N-Drizz is the number of sub-pixels per line (or column) used to compute the approximate intersection of the bar with the current pixel
nt = number of angles used to span the range [0,180] degree

Prioritise given parameters to generate a new kernel. If none given, search for the file name given. 

If no arguments input, look for the default file and, if it doesn't exist, create it.
'''
def init_kernel(kf='kernel_supRHT.fits', ks=None, bw=None, nd=None, nt=None):
    if os.path.isfile(kf) and ks != None and nd != None and bw != None and nt != None:
        kfile=fits.open(kf)
        khead=kfile[0].header
        ksf= khead['KERNSIZE']
        bwf= khead['BARWIDTH']
        ndf= khead['NDRIZZ']
        ntf= khead['NTHETA']
        if ksf == ks and ndf == nd and bwf == bw and ntf == nt:
            kernel = kfile[0].data
        else:
            kernel = kernelset(ks, bw, nd, nt)
            if (kernel == 0).all():
                #print 'Aborted'
                return -1
            khead['KERNSIZE'] = ks
            khead['BARWIDTH'] = bw
            khead['NDRIZZ'] = nd
            khead['NTHETA'] = nt
            hdu = fits.PrimaryHDU(data = kernel, header = khead)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(kf, overwrite=True)
            
    elif ks != None and nd != None and bw != None and nt != None and not os.path.isfile(kf):
        kernel = kernelset(ks, bw, nd, nt)
        if (kernel == 0).all():
            #print 'Aborted'
            return -1
        khead = fits.Header()
        khead['KERNSIZE'] = ks
        khead.comments['KERNSIZE'] = "Size of the kernel's side"
        khead['BARWIDTH'] = bw
        khead.comments['BARWIDTH'] = "Width of Kernel Bar"
        khead['NDRIZZ'] = nd
        khead.comments['NDRIZZ'] = "Number of sub-pixels/line in 1 pix"
        khead['NTHETA'] = nt
        khead.comments['NTHETA'] = "Number of sub-angles in the range [0-180] deg"
        hdu = fits.PrimaryHDU(data = kernel, header = khead)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(kf, overwrite=True)

    else:
        print ('Reading Kernel file: ' + kf)
        if os.path.isfile(kf):
            kfile=fits.open(kf)
            kernel=kfile[0].data
            khead=kfile[0].header
            ks= khead['KERNSIZE']
            bw= khead['BARWIDTH']
            nd= khead['NDRIZZ']
            nt= khead['NTHETA']
        else:
            print ('File not found, using default values')
            ks,bw,nd,nt = 55, 3, 101, 180
            kernel = kernelset(ks, bw, nd, nt)
            if (kernel ==0).all():
                #print 'Aborted'
                return -1
            khead = fits.Header()
            khead['KERNSIZE'] = ks
            khead.comments['KERNSIZE'] = "Size of the kernel's side"
            khead['BARWIDTH'] = bw
            khead.comments['BARWIDTH'] = "Width of Kernel Bar"
            khead['NDRIZZ'] = nd
            khead.comments['NDRIZZ'] = "Number of sub-pixels/line in 1 pix"
            khead['NTHETA'] = nt
            khead.comments['NTHETA'] = "Number of sub-angles in [0-180]"
            hdu = fits.PrimaryHDU(data = kernel, header = khead)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(kf, overwrite=True)
    print (kf)
    kfile=fits.open(kf)
    khead2=kfile[0].header
    print (repr(khead2))
    return kernel, ks, bw, nd, nt
   


################         
#KERNEL CREATION
################
'''
Generates the kernel, a (kernel_size x kernel_size x ntheta) matrix
'''
def kernelset(kernel_size, bar_width, ndrizz, ntheta):
    if (np.arctan((1.*bar_width)/kernel_size) < (np.pi / ntheta / 2.)):   
        print ('\n Warning: kernel size ('+str(kernel_size)+'), bar width ('+str(bar_width)+'), and sub-angle number ('+str(ntheta)+') are not compatible with a full coverage of the image  !! Process Aborted')
        return np.zeros((int(kernel_size),int(kernel_size), int(ntheta)))
    print ('Generating Kernel with arguments: kernel size = '+str(kernel_size)+', bar width = '+str(bar_width)+', sub-pixel number = '+str(ndrizz)+', sub-angle number  = '+str(ntheta)+' :')
    kernel=np.zeros((kernel_size, kernel_size, ntheta))
    ii=np.zeros((ndrizz,ndrizz))
    jj=np.zeros((ndrizz,ndrizz))
    for i in range(ndrizz):
        for j in range(ndrizz):
            ii[j][i]=(i-(ndrizz/2))
            jj[j][i]=(j-(ndrizz/2))
    ii = ii / ndrizz 
    jj = jj / ndrizz 
    for i in range(ntheta):
        theta=[(i+90.)*np.pi/ntheta]
        ro=np.zeros((ndrizz,ndrizz))
        for j in range(kernel_size):
            for k in range(kernel_size):
                ro= 1*((j-(kernel_size/2)+ii)*np.cos(theta)+(k-(kernel_size/2)+jj)*np.sin(theta))
                circle=(((j-(kernel_size/2)+ii)**2+(k-(kernel_size/2)+jj)**2) <= (kernel_size/2.)**2 )
                kernel[k][j][i]=1.*np.sum((ro<=(bar_width/2.))*(ro>=-(bar_width/2.))*circle)/(ndrizz**2)
        bar=int(i*(20./ntheta))
        percent=int(i*(100./ntheta))
        sys.stdout.write('\r [{1}{2}] {0}%'.format(percent, '='*bar, ' '*(20-bar)))
        sys.stdout.flush()
    print ('\nKernel ready!')
    return kernel





########################
#ROLLING HOUGH TRANSFORM
########################
'''

Apply the Hough transform by convolving the kernel matrix with the selected map

hthets histogram gives for each pixel the contribution for every angle bin

'''
def suprht(filename, kernel, kernel_size, bar_width, ntheta, frac, sigma, smr, path):
    fileh=filename
    rootbeg=fileh.rfind('/')
    rootend=fileh.rfind('.fits')

    root=fileh[rootbeg+1:rootend]
    print ('code:\n'+root)
    huf=fits.open(fileh, ext=1)
    inmap=huf[0].data
    hheader=huf[0].header
    print (inmap.shape)
    #IMAGE CROPING
    ###############
    threshold=frac*(kernel_size*bar_width)
    bitmask=mask_bit(inmap, smr)
    if sigma != None:
        op2=mask_red(inmap, stdev=sigma)
    else:
        op2=np.ones_like(inmap, dtype=bool)
    edge_crop=np.zeros_like(inmap, dtype=bool)
    print ("value:", (kernel_size/2)+1,len(edge_crop[0])-((kernel_size/2)+1),(kernel_size/2)+1)
    edge_crop[int((kernel_size/2)+1):len(edge_crop[0])-int((kernel_size/2)+1),int(kernel_size/2)+1:len(edge_crop[1])-int((kernel_size/2)+1)]=True
    op2=op2*edge_crop
    opi=(np.where((op2)>0))[1]
    opj=(np.where((op2)>0))[0]
    npix = np.sum(op2)
    hthets=np.zeros((npix,ntheta))
    hthets2=np.zeros((npix,ntheta))
    pix=np.zeros((kernel_size,kernel_size))

    #HOUGH TRANSFORM
    ################
    print ('Rolling Hough Transform in progress...')
    for i in range(npix):
        pix=(bitmask*op2)[opj[i]-int(kernel_size/2):opj[i]+int(kernel_size/2)+1,opi[i]-int(kernel_size/2):opi[i]+int(kernel_size/2)+1]
        # option 1
        hthets[i] = np.sum(kernel*np.repeat(pix[:, :, np.newaxis], ntheta, axis=2), axis=(0,1))
        # option 2
        #hthets[i] = np.sum([pix*kernel[:,:,j] for j in range(ntheta)], axis=(1,2))
        # option 3
        #for j in range(ntheta):
        #    hthets[i][j] = np.sum([pix*kernel[:,:,j], axis=(0,1))
        # Computing bar
        bar=int((i+1)/(npix/20.))
        percent=int(100.*(i+1)/npix)
        sys.stdout.write('\r [{1}{2}] {0}%'.format(percent, '='*bar, ' '*(20-bar)))
        sys.stdout.flush()
    threshold2=(hthets>threshold)
    threshold3=(hthets<threshold)
    hthets2[threshold3]=0
    hthets2[threshold2]=hthets[threshold2] - threshold
    dtheta = np.pi/ntheta
    thets = np.arange(-np.pi/2., np.pi/2., dtheta)

    #COMPUTING RHT STOKES PARAMETERS, THETA_RHT, SIGMA_THETA_RHT AND FILAMENTS MAP
    ###############################################################
    print ('\nComputing...')
    xpic = hheader["NAXIS1"]
    ypic = hheader["NAXIS2"]
    UNRHT = np.zeros((ypic, xpic), np.float_)
    QNRHT = np.zeros((ypic, xpic), np.float_)
    UNRHTsq = np.zeros((ypic, xpic), np.float_)
    QNRHTsq = np.zeros((ypic, xpic), np.float_)
    INTNRHT = np.zeros((ypic, xpic), np.float_)
    QNRHT[opj,opi] = np.dot(hthets2, np.cos(2*thets))
    UNRHT[opj,opi] = np.dot(hthets2, np.sin(2*thets))
    QNRHTsq[opj,opi] = np.dot(hthets2, np.cos(2*thets)**2)
    UNRHTsq[opj,opi] = np.dot(hthets2, np.sin(2*thets)**2)
    INTNRHT[opj,opi] = np.sum(hthets2, axis=1)
    theta = 0.5*np.arctan2(UNRHT, QNRHT)*180/np.pi
    sigtheta = np.sqrt((QNRHT**2 * UNRHTsq +  UNRHT**2 * QNRHTsq) / (4 * (QNRHT**2 + UNRHT**2)**2.) )*180/np.pi
    nanpos=np.isnan(sigtheta)
    p=(sigtheta<0.01)
    if np.sum(nanpos)!=0:
        sigtheta[nanpos]=0
    if np.sum(p)!=0:
        sigtheta[p]=0

    #SAVING FITS
    ############
    thetahdu = fits.PrimaryHDU(theta)
    filetheta=path+root+'_SUPRHT_THETA_K'+str(kernel_size)+'_BAR'+str(bar_width)+'.fits'
    new_filename= path+root+'_SUPRHT_K'+str(kernel_size)+'_BAR'+str(bar_width)+'.fits'
    filestheta=path+root+'_SUPRHT_SIGTHETA_K'+str(kernel_size)+'_BAR'+str(bar_width)+'.fits'
    thetahdu.writeto(filetheta)
    sthetahdu = fits.PrimaryHDU(sigtheta)
    sthetahdu.writeto(filestheta)
    fits_create(new_filename, opi, opj, hthets, kernel_size, bar_width, smr, frac, 'False', backproj=INTNRHT)
    return new_filename, filetheta, filestheta



################
#GLOBAL FUNCTION
################

'''

Starts the entire program

To use this function, you need at least a map path, the other arguments will get default values

However, you can fine tune all the optional arguments, here is a little description of these:

    -you can point a scpecific, already generated kernel with the argument kernel_file

    -kernel_size, minipix_size and bar_width are the same arguments as in the function init_kernel

    -with histogram_fraction, you can specify the fraction of the hthets histogram that is taken into account

    -sigma_gauss_edge_smooth can be set to reduce the original field in order to avoid edge effect, the greater sigma will be, the sharper will be the cropping
     (developped for Herschel fields)

    -smooth_rad_bitmap determines the width of the TopHat kernel used to generate the bitmap

    -the path keyword can be used to indicate a different path where to save output .fits files

'''

def main(filename,kernel_file='kernel_supRHT.fits', kernel_size=None, bar_width=None, ndrizz=None, ntheta=None, histogram_fraction=0.7, sigma_gauss_edge_smooth=None, smooth_rad_bitmap=11, path=''):
    res = init_kernel(kernel_file, kernel_size, bar_width, ndrizz, ntheta)
    print ("res:", res);
    if res == -1:
    	return ;    	
    else:	
    	kernel, ks, bw, nd, nt = res
    if filename.endswith('.fits'):
        delete(filename,ks, bw, nd, nt, path)
        new_filename, filetheta, filestheta = suprht(filename,kernel, ks, bw, nt, histogram_fraction, sigma_gauss_edge_smooth, smooth_rad_bitmap, path)
        file_created(new_filename, filetheta, filestheta)
    else:
        raise ValueError('Only .fits files are supported')




#RE-EXTRACTING RHT STOKES PARAMETERS, THETA_RHT, SIGMA_THETA_RHT AND FILAMENTS MAP
##################################################################################

'''
Given an already generated supeRHT file and the original map, you can recalculate Q_RHT, U_RHT, THETA_RHT, SIMA_THETA_RHT and INTRHT(filaments map) with another frac
'''

def QUINT_extraction(fileh, file_SUPERHT, frac):
    threshold=frac*(kernel_size*bar_width)
    huf=fits.open(fileh, ext=1)
    inmap=huf[0].data
    hheader=huf[0].header
    hth=fits.open(file_SUPERHT)
    datahth=hth[1].data
    hthets=datahth['hthets']
    opi=datahth['hi']
    opj=datahth['hj']
    op2=mask_red(inmap)
    threshold2=(hthets>threshold)
    threshold3=(hthets<threshold)
    hthets[threshold3]=0
    hthets[threshold2]=hthets[threshold2] - threshold
    ntheta = 180
    dtheta = np.pi/ntheta
    thets = np.arange(-np.pi/2., np.pi/2., dtheta)
    xpic = hheader['NAXIS1']
    ypic = hheader['NAXIS1']
    UNRHT = np.zeros((ypic, xpic), np.float_)
    QNRHT = np.zeros((ypic, xpic), np.float_)
    UNRHTsq = np.zeros((ypic, xpic), np.float_)
    QNRHTsq = np.zeros((ypic, xpic), np.float_)
    INTNRHT = np.zeros((ypic, xpic), np.float_)
    print ('Computing...')
    QNRHT[opj,opi] = np.dot(hthets, np.cos(2*thets))
    UNRHT[opj,opi] = np.dot(hthets, np.sin(2*thets))
    QNRHTsq[opj,opi] = np.dot(hthets, np.cos(2*thets)**2)
    UNRHTsq[opj,opi] = np.dot(hthets, np.sin(2*thets)**2)
    INTNRHT[opj,opi] = np.sum(hthets, axis=1)
    theta = 0.5*np.arctan2(UNRHT, QNRHT)*180/np.pi
    sigtheta = np.sqrt((QNRHT**2 * UNRHTsq +  UNRHT**2 * QNRHTsq) / (4 * (QNRHT**2 + UNRHT**2)**2.) )*180/np.pi
    nanpos=np.isnan(sigtheta)
    p=(sigtheta<0.01)
    if np.sum(nanpos)!=0:
        sigtheta[nanpos]=0
    if np.sum(p)!=0:
        sigtheta[p]=0
    return INTNRHT, theta, sigtheta, QNRHT, UNRHT, np.sum(hthets, axis=0), thets


#RE-COMPUTING FILAMENTS MAP
###########################

'''
Same function as before but only recalculating the filments map
'''

def INT_extraction(fileh, file_SUPERHT, frac):
    threshold=frac*(kernel_size*bar_width)
    huf=fits.open(fileh, ext=1)
    inmap=huf[0].data
    hheader=huf[0].header
    hth=fits.open(file_SUPERHT)
    datahth=hth[1].data
    hthets=datahth['hthets']
    opi=datahth['hi']
    opj=datahth['hj']
    op2=mask_red(inmap)
    threshold2=(hthets>threshold)
    threshold3=(hthets<threshold)
    hthets[threshold3]=0
    hthets[threshold2]=hthets[threshold2] - threshold
    xpic = hheader['NAXIS1']
    ypic = hheader['NAXIS1']
    INTNRHT = np.zeros((ypic, xpic), np.float_)
    print ('Computing...')
    INTNRHT[opj,opi] = np.sum(hthets, axis=1)
    return INTNRHT

#FIELD EDGE CROPPING FUNCTION
#############################

def mask_red(inmap, stdev=25):
    pos=(inmap>0)
    post=pos*1
    gauss=Gaussian2DKernel(stdev, x_size=(stdev*2)+1, y_size=(stdev*2)+1)
    op=convolve_fft(post,gauss, crop=True, normalize_kernel=True)
    op2=(op>0.95)
    return op2


#BITMAP GENERATOR
#################

def mask_bit(inmap, smooth_rad):
    tophat=Tophat2DKernel(smooth_rad)
    inmap_conv=convolve_fft(inmap, tophat)
    bitmask=((inmap-inmap_conv)>0)
    return bitmask

#DELETING FUNCTION
##################

'''

Delete previous file if you're trying to create a new one with the same name

'''

def delete(filename, kernel_size, bar_width, ndrizz, ntheta, path):
    filenam=filename
    rootbeg=filenam.rfind('/')
    rootend=filenam.rfind('.fits')
    root=filenam[rootbeg+1:rootend]
    filetheta=path+root+'_SUPERHT_THETA_K'+str(kernel_size)+'_BAR'+str(bar_width)+'.fits'
    new_filename= path+root+'_SUPERHT_K'+str(kernel_size)+'_BAR'+str(bar_width)+'.fits'
    filestheta=path+root+'_SUPERHT_SIGTHETA_K'+str(kernel_size)+'_BAR'+str(bar_width)+'.fits'
    for f in [new_filename, filestheta, filetheta]:
        try:
            os.remove(f)
        except:
            continue

#CHEKING FILE FUNCTION
######################

'''

checks if the files have succesfully been created at the end of the processing


'''

def file_created(frht, ftrht, fstrht):
    fc='Files succesfully saved:\t'
    if os.path.isfile(frht):
        fc=fc+'SUPERHT.fits\t'
    if os.path.isfile(ftrht):
        fc=fc+'SUPERHT_THETA.fits\t'
    if os.path.isfile(fstrht):
        fc=fc+'SUPERHT_SIGTHETA.fits\t'
    print (fc)

#CREATING FITS FUNCTION
#######################

'''

creates the .fits file with the filaments map, hthets histogram, the coordinates of the hthets pixels (hi, hj) and the header

'''

def fits_create(xyt_filename, hi, hj, hthets, wlen, bw, smr, frac, original, backproj=None):
        Hi = fits.Column(name='hi', format='1I', array=hi)
        Hj = fits.Column(name='hj', format='1I', array=hj)
        ntheta = hthets.shape[1]
        Hthets = fits.Column(name='hthets', format=str(int(ntheta))+'E', array=hthets)
        cols = fits.ColDefs([Hi, Hj, Hthets])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        prihdr = fits.Header()
        prihdr['WLEN'] = wlen 
        prihdr['BW'] = bw
        prihdr['SMR'] = smr
        prihdr['FRAC'] = frac
        prihdr['ORIGINAL'] = original
        prihdr['NTHETA'] = ntheta
        prihdu = fits.PrimaryHDU(data=backproj, header=prihdr)
        thdulist = fits.HDUList([prihdu, tbhdu])
        thdulist.writeto(xyt_filename, output_verify='silentfix', overwrite=True, checksum=True)
