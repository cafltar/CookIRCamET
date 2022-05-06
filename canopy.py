import sys
import numpy as np
import cv2
import time

from constants import *
from time import sleep
from datetime import datetime
import os
import numpy as np
import logging 

def kb(theta):
    return (np.sqrt(XE ** 2 + (np.tan(thetar)) ** 2)) / (XE + 1.774 * (XE + 1.182) ** -0.733)

def taudir(thetas , psis , hc , wc , row , LAI , rhosoil ): 
    #def taudir to compute transmittance of DIRECT beam radiation through the canopy
    #using procedure of Campbell and Norman (1998), Chapter 15 (CN98)

    #Thetas = Solar zenith angle (rad)
    #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    #   and 90 degrees for perpendicular orientation (rad)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #LAI = Leaf area index, field (m2 m-2)
    #XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
    #Zeta = Leaf absorptivity (usually 0.85 for VIS, 0.15 for NIR)
    #rhosoil = Soil reflectance (for Pullman clay loam, ~0.15 for VIS, ~0.25 for NIR)

    #PLFi   #Path length fraction of continuous ellipse canopy
    #               MRFi   #Multiple row function of continuous ellipse canopy
    #Kdir   #Extinction coefficient for direct beam radiation
    #                rhohor      #Reflectivity for horizontal leafs (VIS or NIR)
    #rhodir      #Adjusted canopy reflectivity as a function of Kdir and rhohor (VIS or NIR)

    PLFi = PLF(thetas, psis, hc, wc, row)
    MRFi = MRF(thetas, psis, hc, wc, row)
    Kdir = kb(thetas) #CN98, 15.4, p. 251
    rhohor = (1 - ((Zeta) ** 0.5)) / (1 + ((Zeta) ** 0.5))    #CN98, 15.7, p. 255
    rhodir = 2 * Kdir * rhohor / (Kdir + 1)     #CN98, 15.8, p. 257

    #Convert field LAI to local LAI
    LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
    LAIL = LAI * row / wc

    return (((rhodir ** 2) - 1) * np.exp(-(Zeta ** 0.5) * Kdir * LAIL * PLFi * MRFi)) / (((rhodir * rhosoil) - 1) + rhodir * (rhodir - rhosoil) * np.exp(-2 * (Zeta ** 0.5) * Kdir * LAIL * PLFi * MRFi))    #CN98, 15.11, p. 257
    
#def taudiff to compute transmittance of DIFFUSE radiation through the canopy
#by integrating taudir over all solar zenith and azimuth angles
def taudiff(hc , wc , row , LAI , rhosoil): 
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #LAI = Leaf area index, field (m2 m-2)
    #XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
    #Zeta = Leaf absorptivity (usually 0.85 for VIS, 0.15 for NIR)
    #rhosoil = Soil reflectance (for Pullman clay loam, ~0.15 for VIS, ~0.25 for NIR)
    psis = pi / 12      #Numerical integration carried out in 15 degree increments
    tau_diff = 0

    while psis <= pi / 2:
        thetas = pi / 12    #Numerical integration carried out in 15 degree increments
        while thetas < pi / 2:
            TDIR = taudir(thetas, psis, hc, wc, row, LAI, rhosoil)
            tau_diff = taudiff + 2 * (2 / pi) * TDIR * np.sin(thetas) * np.cos(thetas) * (pi / 12) * (pi / 12)
            thetas = thetas + pi / 12
        psis = psis + pi / 12
    return tau_diff                                                   

def rhocsdir(thetas , psis , hc , wc , row , LAI , rhosoil ): 
    #Thetas = Solar zenith angle (rad)
    #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    #   and 90 degrees for perpendicular orientation (rad)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #LAI = Leaf area index, field (m2 m-2)
    #XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
    #Zeta = Leaf absorptivity (usually 0.85 for VIS, 0.15 for NIR)
    #rhosoil = Soil reflectance (for Pullman clay loam, ~0.15 for VIS, ~0.25 for NIR)
    
    PLFi = PLF(thetas, psis, hc, wc, row)
    MRFi = MRF(thetas, psis, hc, wc, row)
    Kdir = kb(thetas) #CN98, 15.4, p. 251
    rhohor = (1 - ((Zeta) ** 0.5)) / (1 + ((Zeta) ** 0.5))    #CN98, 15.7, p. 255
    rhodir = 2 * Kdir * rhohor / (Kdir + 1)     #CN98, 15.8, p. 257
    
    #Convert field LAI to local LAI
    LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
    LAIL = LAI * row / wc
    
    Xidir = ((rhodir - rhosoil) / (rhodir * rhosoil - 1)) * np.exp(-2 * (Zeta ** 0.5) * Kdir * LAIL * PLFi * MRFi)

    rhocsdir = (rhodir + Xidir) / (1 + Xidir * rhodir)  #CN98, 15.9, p. 257

                                                    
def rhocsdiff(hc , wc , row , LAI , rhosoil ):
    #def rhocsdiff to compute hemispherical-hemispherical reflectance of DIFFUSE radiation from the canopy
    #by integrating rhocsdir over all solar zenith and azimuth angles

    #hc = Canopy height (m)
    #wc = canopy width (m)
    #LAI = Leaf area index, field (m2 m-2)
    #XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
    #Zeta = Leaf absorptivity (usually 0.85 for VIS, 0.15 for NIR)
    #rhosoil = Soil reflectance (for Pullman clay loam, ~0.15 for VIS, ~0.25 for NIR)
    #Initialize variables
    psis = pi / 12      #Numerical integration carried out in 15 degree increments
    rhocs_diff_tmp = 0

    while psis <= pi / 2:
        thetas = pi / 12    #Numerical integration carried out in 15 degree increments
        while thetas < pi / 2:
            RDIR = rhocsdir(thetas, psis, hc, wc, row, LAI, rhosoil)
            rhocs_diff_tmp = rhocsdiff_tmp + 2 * (2 / pi) * RDIR * np.sin(thetas) * np.cos(thetas) * (pi / 12) * (pi / 12)
            thetas = thetas + pi / 12
        psis = psis + pi / 12
    return rhocs_diff_tmp

def MRF(thetas , psis , hc , wc , row ): 

    #def MRF (multiple row factor) to calculate the number of crop rows
    #along the path of a sunbeam, where crop rows are modeled as ellipses.

    #Thetas = Solar zenith angle (rad)
    #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    #   and 90 degrees for perpendicular orientation (rad)
    #hc = Canopy height (m)
    #wc = canopy width (m)

    # bc     #Vertical axis of crop ellipse (m)
    # ac    #Horizonal axis of crop ellipse (m)
    # thetasp    #Solar zenith angle projected perpendicular to crop row (rad)

    # Xscr   #Horizontal distance from canopy ellipse origin to tangent of sunray
    #along thetaspcr (m)
    # Yscr   #Vertical distance from canopy ellipse origin to tangent of sunray
    #along thetaspcr (m)
    # thetaspcr  #Critical perpendicular solar zenith angle, where greater angles
    #result in adjacent row shading (rad)

    # Xscr2   #Horizontal distance from canopy ellipse origin to tangent of sunray
    #along thetaspcr2, for the next row from Xscr (m)
    # Yscr2   #Vertical distance from canopy ellipse origin to tangent of sunray
    #along thetaspcr2, for the next row from Yscr (m)
    # thetaspcr2  #Critical perpendicular solar zenith angle, where greater angles
    #result in adjacent row shading, for the next row from thetascr (rad)


    if thetas > 89 * pi / 180:
        return 0

    #Constrain wc to row spacing
    wc = min(wc,(row - 0.01))
    bc = hc / 2
    ac = wc / 2
    thetasp = np.atan((np.tan(thetas)) * np.abs(np.sin(psis)))
    n = 0
    thetaspcr2 = 0

    while thetasp >= thetaspcr2:
        n = n + 1
        Xscr = 2 * ((ac) ** 2) / (n * row)   #Xscr is positive
        Yscr = np.sqrt(((bc ** 2) * Xscr * (n * row - 2 * Xscr)) / 2 / ((ac) ** 2)) #Yscr is positive
        thetaspcr = np.atan((n * row - 2 * Xscr) / (2 * Yscr)) #thetascr is positive

        Xscr2 = 2 * ((ac) ** 2) / ((n + 1) * row) #Xscr is positive
        Yscr2 = np.sqrt(((bc ** 2) * Xscr2 * ((n + 1) * row - 2 * Xscr2)) / 2 / ((ac) ** 2)) #Yscr is positive
        thetaspcr2 = np.atan(((n + 1) * row - 2 * Xscr2) / (2 * Yscr2)) #thetascr is positive

        MRFtmp = n + ((thetasp - thetaspcr) / (thetaspcr2 - thetaspcr))

        if MRFtmp < 1: MRFtmp = 1
        return MRFtmp
    
def PLF(thetas , psis , hc , wc , row ): 
    #def PLF (Path Length Fraction) to calculate the path length of a sunray
    #relative to NADIR of an elliptical canopy

    #Thetas = Solar zenith angle (rad)
    #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    #   and 90 degrees for perpendicular orientation (rad)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    
    # bc     #Vertical axis of crop ellipse (m)
    # ac    #Horizonal axis of crop ellipse (m)
    # thetasp    #Solar zenith angle projected perpendicular to crop row (rad)
    
    # Xp   #Horizontal distance from canopy ellipse origin to point of entry of
    #sunray that passes through canopy ellipse origin (m)
    # Yp   #Vertical distance from canopy ellipse origin to point of entry of
    #sunray that passes through canopy ellipse origin (m)
    # Zp   #Axial distance from canopy ellipse origin to point of entry of
    #sunray that passes through canopy ellipse origin (m)

    if thetas > 89 * pi / 180:
        return 0
    
    #Constrain wc to row spacing
    wc = min(wc,(row - 0.01))

    #Constrain psis to > 0
    psis = max(psis,0.01)

    bc = hc / 2
    ac = wc / 2
    thetasp = np.atan((np.tan(thetas)) * np.abs(np.sin(psis)))
    Yp = ac * bc / np.sqrt((bc ** 2) * ((np.tan(thetasp)) ** 2) + (ac ** 2))
    Xp = Yp * np.tan(thetasp)
    Zp = Xp / np.abs(np.tan(psis))
    return (np.sqrt(Xp ** 2 + Yp ** 2 + Zp ** 2)) / bc

def fcsolar(thetas , psis , hc , wc , row ): 
    #def fcsolar to estimate the fraction of the total surface covered by the canopy normal
    #to the solar zenith angle, where the canopy is modelled as an ellipse.
    #Both leaves and substrate may be visible within the canopy area.
    
    #Thetas = Solar zenith angle (rad)
    #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    #   and 90 degrees for perpendicular orientation (rad)
    #hc = Canopy height (m)
    #wc = canopy width (m)

    if thetas > 89 * pi / 180:
        return 0
    
    #Constrain wc to row spacing
    wc = min(wc,(row - 0.01))

    # bc     #Vertical axis of crop ellipse (m)
    # ac    #Horizonal axis of crop ellipse (m)
    # thetasp    #Solar zenith angle projected perpendicular to crop row (rad)
    
    # Xscr   #Horizontal distance from canopy ellipse origin to tangent of sunray
    #along thetaspcr (m)
    # Yscr   #Vertical distance from canopy ellipse origin to tangent of sunray
    #along thetaspcr (m)
    # thetaspcr  #Critical perpendicular solar zenith angle, where greater angles
    #result in adjacent row shading (rad)

    bc = hc / 2
    ac = wc / 2
    thetasp = np.atan((np.tan(thetas)) * np.abs(np.sin(psis)))
    
    Xscr = 2 * ((ac) ** 2) / (row)   #Xscr is positive
    Yscr = np.sqrt(((bc ** 2) * Xscr * (row - 2 * Xscr)) / 2 / ((ac) ** 2)) #Yscr is positive
    thetaspcr = np.atan((row - 2 * Xscr) / (2 * Yscr)) #thetascr is positive

    if thetasp >= thetaspcr:
        return 1
    # Xs     #Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # Ys     #Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)

    Xs = ac / np.sqrt(1 + (bc ** 2) / ((ac) ** 2) * ((np.tan(thetasp)) ** 2))    #Xs is positive
    Ys = (bc ** 2) / ((ac) ** 2) * Xs * np.tan(thetasp) #Ys is positive

    return (2 * Xs + 2 * Ys * np.tan(thetasp)) / row

def fdhc(hc , wc , row , Pr , Vr ): 
    #def fdhc to calculate the downward hemispherical view factor of canopy
    #of a row crop (e.g., canopy viewed by an inverted radiometer)

    #hc = Crop canopy height (m)
    #wc = Crop canopy width (m)
    #row = Crop row spacing (m)
    #Vr = Vertical distance of radiometer from soil surface (m)
    #Pr = Horizontal, perpendicular distance from radiometer to row center (m)

    # ac     #Crop canopy major semiaxis
    # bc     #Crop canopy minor semiaxis
    # pi 
    # psir  #Azimuth view angle element of radiometer (radians)
    # fdhsoil  #Downward hemispherical view factor of soil
    # thetar1  #Zenith view angle of radiometer to ellipse tangent, left of ellipse (radians)
    # thetar2  #Zenith view angle of radiometer to ellipse tangent, right of ellipse (radians)
    # i As Integer #Loop counter for multiple rows
    # NR As Integer    #Minimum number of interrows where soil is visible to radiometer
    # tanthetarcr   #np.tangent of maximum zenith view angle of radiomoeter-soil view factor (radians)

    ac = hc / 2
    bc = wc / 2
    psir = pi / 72
    fdhsoil = 0
    if wc >= row:
        return 1 - fdhsoil

    while psir < pi / 2:
        tanthetarcr = row * ((1 - 4 * (bc ** 2) / (row ** 2)) ** 0.5) / (2 * ac * np.sin(psir))
        NR = np.ceil(((Vr * tanthetarcr * np.sin(psir)) / row), 0) + 2
        for i in np.arange(-NR,NR,1):
            thetar1 = np.atan(quartic1(ac, bc, psir, (row * (i + 1) - Pr), Vr))
            thetar2 = np.atan(quartic2(ac, bc, psir, (row * i - Pr), Vr))
            if thetar1 > thetar2:
                fdhsoil = fdhsoil + (2 / pi) * (1 / pi) * (thetar1 - thetar2) * (pi / 72)
            else:
                pass
            psir = psir + pi / 72
    return 1 - fdhsoil

def constrainpsi(rawpsi):

    #Constrain 45 < rawpsi < 90
    if np.abs(rawpsi) < 45 * pi / 180:
        psir = pi / 2 - (np.abs(rawpsi))
    else:
        if np.abs(rawpsi) > 135 * pi / 180:
            psir = (np.abs(rawpsi)) - pi / 2
        else:
            if np.abs(rawpsi) > 90 * pi / 180:
                psir = pi - np.abs(rawpsi)
            else:
                psir = np.abs(rawpsi)
    return psir

def quartic1IRT(X1 , Y1 , ac , bc ): 
    #Compute the 1st root of a quartic equation of form
    #Ax**4 + Bx**3 + Cx**2 + Dx + E = 0
    #Compute coefficients of quartic
    A = (Y1 ** 2) * (bc ** 2) - (bc ** 4)
    b = -2 * X1 * Y1 * (bc ** 2)
    c = (Y1 ** 2) * (ac ** 2) + (X1 ** 2) * (bc ** 2) - 2 * (ac ** 2) * (bc ** 2)
    d = -2 * X1 * Y1 * (ac ** 2)
    E = (X1 ** 2) * (ac ** 2) - (ac ** 4)

    if np.abs(X1) < 0.001:     #Assume quadratic equation
        return -np.sqrt((-c + np.sqrt(c ** 2 - 4 * A * E)) / (2 * A))
        
    #Compute quartic
    alpha = -((3 * b ** 2) / (8 * A ** 2)) + c / A
    beta = (b ** 3) / (8 * A ** 3) - (b * c) / (2 * A ** 2) + d / A
    gamma = -(3 * b ** 4) / (256 * A ** 4) + (c * b ** 2) / (16 * A ** 3) - (b * d) / (4 * A ** 2) + E / A
    P = -((alpha ** 2) / 12) - gamma
    Q = -(alpha ** 3) / (108) + (alpha * gamma / 3) - (beta ** 2) / 8
    r = Q / 2 + np.sqrt((Q ** 2) / 4 + (P ** 3) / 27)
    if r < 0:
        u = -((np.abs(r)) ** (1 / 3))
    else:
        u = r ** (1 / 3)

    # Uy 
    if u = 0:
        Uy = 0
    else:
        Uy = P / (3 * u)
                 
    y = -5 / 6 * alpha - u + Uy
    W = np.sqrt(np.abs(alpha + 2 * y))

    # ZZ 
    if W = 0:
        ZZ = np.abs(3 * alpha + 2 * y)
        return  -(b / (4 * A)) + (W - np.sqrt(ZZ)) / 2 - y #(y subtracted to smooth curve)
    else:
        ZZ = -(3 * alpha + 2 * y + 2 * beta / W)
        if ZZ < 0:
            ZZ = np.abs(3 * alpha + 2 * y - 2 * beta / W)
            return  -(b / (4 * A)) + (-W - np.sqrt(ZZ)) / 2
        else:
            return  -(b / (4 * A)) + (W - np.sqrt(ZZ)) / 2

def quartic2IRT(X1 , Y1 , ac , bc ):
    #Compute the 2nd root of a quartic equation of form
    #Ax**4 + Bx**3 + Cx**2 + Dx + E = 0

    #Compute coefficients of quartic
    A = (Y1 ** 2) * (bc ** 2) - (bc ** 4)
    b = -2 * X1 * Y1 * (bc ** 2)
    c = (Y1 ** 2) * (ac ** 2) + (X1 ** 2) * (bc ** 2) - 2 * (ac ** 2) * (bc ** 2)
    d = -2 * X1 * Y1 * (ac ** 2)
    E = (X1 ** 2) * (ac ** 2) - (ac ** 4)
                 
    if np.abs(X1) < 0.001:     #Assume quadratic equation
        return np.sqrt((-c + np.sqrt(c ** 2 - 4 * A * E)) / (2 * A))
    #Compute quartic
    alpha = -((3 * b ** 2) / (8 * A ** 2)) + c / A
    beta = (b ** 3) / (8 * A ** 3) - (b * c) / (2 * A ** 2) + d / A
    gamma = -(3 * b ** 4) / (256 * A ** 4) + (c * b ** 2) / (16 * A ** 3) - (b * d) / (4 * A ** 2) + E / A
    P = -((alpha ** 2) / 12) - gamma
    Q = -(alpha ** 3) / (108) + (alpha * gamma / 3) - (beta ** 2) / 8
    r = Q / 2 + np.sqrt((Q ** 2) / 4 + (P ** 3) / 27)
    if r < 0:
        u = -((np.abs(r)) ** (1 / 3))
    else:
        u = r ** (1 / 3)

    # Uy 
    if u = 0:
        Uy = 0
    else:
        Uy = P / (3 * u)
    
    y = -5 / 6 * alpha - u + Uy
    W = np.sqrt(np.abs(alpha + 2 * y))
                 
    # ZZ 
    if W = 0:
        ZZ = np.abs(3 * alpha + 2 * y)
        return -(b / (4 * A)) + (-W + np.sqrt(ZZ)) / 2 - y #(y subtracted to smooth curve)
    else:
        ZZ = -(3 * alpha + 2 * y + 2 * beta / W)
        if ZZ < 0:
            ZZ = np.abs(3 * alpha + 2 * y - 2 * beta / W)
            return -(b / (4 * A)) + (-W + np.sqrt(ZZ)) / 2
        else:
            return -(b / (4 * A)) + (W + np.sqrt(ZZ)) / 2
