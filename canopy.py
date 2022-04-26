import sys
import numpy as np
import cv2
import time
import utils, canopy
from time import sleep
from datetime import datetime
import os
import numpy as np
import logging 

def taudir(thetas , psis , hc , wc , row , LAI , XE , Zeta , rhosoil ): 
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

    PLFi = canopy.PLF(thetas, psis, hc, wc, row)
    MRFi = canopy.MRF(thetas, psis, hc, wc, row)
    Kdir = ((XE ** 2 + (np.tan(thetas)) ** 2) ** 0.5) / (XE + 1.774 * (XE + 1.182) ** -0.733) #CN98, 15.4, p. 251
    rhohor = (1 - ((Zeta) ** 0.5)) / (1 + ((Zeta) ** 0.5))    #CN98, 15.7, p. 255
    rhodir = 2 * Kdir * rhohor / (Kdir + 1)     #CN98, 15.8, p. 257

    #Convert field LAI to local LAI
    LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
    LAIL = LAI * row / wc

    return (((rhodir ** 2) - 1) * np.exp(-(Zeta ** 0.5) * Kdir * LAIL * PLFi * MRFi)) / (((rhodir * rhosoil) - 1) + rhodir * (rhodir - rhosoil) * np.exp(-2 * (Zeta ** 0.5) * Kdir * LAIL * PLFi * MRFi))    #CN98, 15.11, p. 257
    
#def taudiff to compute transmittance of DIFFUSE radiation through the canopy
#by integrating taudir over all solar zenith and azimuth angles
def taudiff(hc , wc , row , LAI , XE , Zeta , rhosoil): 
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
            TDIR = taudir(thetas, psis, hc, wc, row, LAI, XE, Zeta, rhosoil)
            tau_diff = taudiff + 2 * (2 / pi) * TDIR * np.sin(thetas) * np.cos(thetas) * (pi / 12) * (pi / 12)
            thetas = thetas + pi / 12
        psis = psis + pi / 12
    return tau_diff                                                   

def rhocsdir(thetas , psis , hc , wc , row , LAI , XE , Zeta , rhosoil ): 
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
    Kdir = ((XE ** 2 + (np.tan(thetas)) ** 2) ** 0.5) / (XE + 1.774 * (XE + 1.182) ** -0.733) #CN98, 15.4, p. 251
    rhohor = (1 - ((Zeta) ** 0.5)) / (1 + ((Zeta) ** 0.5))    #CN98, 15.7, p. 255
    rhodir = 2 * Kdir * rhohor / (Kdir + 1)     #CN98, 15.8, p. 257
    
    #Convert field LAI to local LAI
    LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
    LAIL = LAI * row / wc
    
    Xidir = ((rhodir - rhosoil) / (rhodir * rhosoil - 1)) * np.exp(-2 * (Zeta ** 0.5) * Kdir * LAIL * PLFi * MRFi)

    rhocsdir = (rhodir + Xidir) / (1 + Xidir * rhodir)  #CN98, 15.9, p. 257

                                                    
def rhocsdiff(hc , wc , row , LAI , XE , Zeta , rhosoil ):
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
            RDIR = rhocsdir(thetas, psis, hc, wc, row, LAI, XE, Zeta, rhosoil)
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

def fveg(XE , thetar , rawpsi , hc , wc , row , LAI , Pr , Vr , FOV , thetas , psis ): 
    #def fveg to compute the fraction of vegetation appearing in a radiometer footprint
    #where the crop rows are modeled as continuous ellipses.
    #LADFOPT = Option for leaf angle distribution function (LADF), where
    #           (1 = Ellipsoidal, 2 = Beta)
    #For ellipsoidal LADF option:
    #Xe = Ratio of horizontal to vertical projected leaves (spherical Xe = 1)
    #For Beta LADF option:
    #tAVG = mean of normalized leaf angle t (e.g., 0.5 for symmetric PDF#s)
    #tVAR = variance of normalized leaf angle t

    #thetar = Radiometer view zenith angle (radians)
    #rawpsi = Azimuth angel of crop row relative to radiometer view angle,
    #   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    #   to crop row (radians)
    #hc = Canopy height (m)
    #wc = Canopy width (m)
    #row = Crop row spacing (m)
    #LAI = Leaf area index (m2 m-2)

    #Pr = Perpendicular distance of radiometer from canopy row center (m)
    #Vr = Vertical height of radiometer relative to soil (m)
    #FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
    #thetas = Solar zenith angle (rad)
    #psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    #       and 90 degrees for perpendicular orientation (rad)

    #Variables used in this function
    # KBR    #Extinction coefficient of radiometer viewing canopy
    # PFR     #Path length fraction of radiometer viewing continuous ellipse
    # MFR     #Multiple row function of radiometer viewing continuous ellipses
    # psir   #radiometer azimuth angle relative to crop row, constrained to 45-90 deg
    # ER     #Extinction of radiometer view path through canopy
    # fcs1   #Fraction of solid sunlit continuous ellispe appearing in radiometer footprint
    # fcs2   #Fraction of solid shaded continuous ellispe appearing in radiometer footprint

    psir = constrainpsi(rawpsi)
    BR = (np.sqrt(XE ** 2 + (np.tan(thetar)) ** 2)) / (XE + 1.774 * (XE + 1.182) ** -0.733)
    PFR = PLF(thetar, psir, hc, wc, row)
    MFR = MRF(thetar, psir, hc, wc, row)

    ER = np.exp(-KBR * row / wc * LAI * PFR * MFR)

    fcs1 = fcs(1, Pr, row, thetar, rawpsi, hc, wc, Vr, FOV, thetas, psis)
    fcs2 = fcs(2, Pr, row, thetar, rawpsi, hc, wc, Vr, FOV, thetas, psis)

    return (fcs1 + fcs2) * (1 - ER)

def fcs(OPT, Pr , row , theta , rawpsi , hc , wc , Dv , FOV , thetas , psis ) :

    #def fcs to compute the fraction of sunlit or shaded canopy or soil
    #appearing in the elliptical footprint of a radiometer, where the canopy
    #is modelled as ELLIPSE

    #OPT = 1 for sunlit canopy
    #      2 for shaded canopy
    #      3 for sunlit soil
    #      4 for shaded soil

    #Pr = Perpendicular distance of radiometer from canopy row center (m)
    #row = Crop row spacing (m)
    #Theta = Zenith angle of radiometer (radians)
    #Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
    #   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    #   to crop row (radians)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #Dv = Vertical height of radiometer relative to soil (m)
    #FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
    #thetas = Solar zenith angle (radians)
    #psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
    #crop row and pi/2 is looking perpendicular to crop row (radians)

    #Compute major (ar) and minor (br) axes of radiometer footprint

    # ar 
    # OA 
    # br 

    ar = Dv / 2 * (np.tan(theta + np.atan(1 / 2 / FOV)) - np.tan(theta - np.atan(1 / 2 / FOV)))
    OA = Dv * np.tan(theta - np.atan(1 / 2 / FOV))
    br = (np.sqrt(((ar + OA) ** 2) + (Dv ** 2))) / (2 * FOV)

    # Aell     #Total elliptical area of radiometer footprint (m2)
    Aell = ar * br * pi
    
    psi = constrainpsi(rawpsi)
                
    #Compute T1
    # T1     #Distance from radiometer ellipse footprint center to line
    #   tangent to radiometer ellipse for given azimuth and zenith angles (m).
    if (np.abs(psi - 90 * pi / 180) < 0.01):   #90 degree radiometer azimuth
        T1 = ar
    else:
        if np.abs(psi) < 0.01:     #ZERO degree radiometer azimuth
            T1 = br
        else:                        #Radiometer azimuth between ZERO and 90 degrees
            # xtr1 
            # ytr1 
            xtr1 = ar / np.sqrt(1 + (br ** 2) / (ar ** 2) / ((np.tan(psi)) ** 2))
            ytr1 = (br ** 2) / (ar ** 2) * xtr1 / np.tan(psi)
            T1 = xtr1 + ytr1 / np.tan(psi)
            
    #Determine number of rows appearing in radiometer footprint, make it an odd number,
    #and add TWO extra rows either side to account for adjacent row shading

    NR = 2 * (np.ceil((2 * T1 * np.sin(psi) / row), 0)) + 1# + 4#

    # thetasp   #Solar zenith angle perpendicular to canopy (rad)
    # OC     #Horizontal distance from radiometer to center of radiometer footprint (m)
    thetasp = -thetas * np.sin(psis)

    # MIN 
    if np.abs(np.tan(rawpsi)) < 1:
        MIN = np.tan(pi / 2 - psi)
    else:
        MIN = 1
        
    OC = Dv / 2 * (np.tan(theta + np.atan(1 / 2 / FOV)) + np.tan(theta - np.atan(1 / 2 / FOV)))

    Pnr = np.zeros(NR)
    H = np.zeros((NR,6))
    ac = np.zeros((NR,6))
    # N1 As Integer
    for N1 in range(0,NR): #Row number

        Pnr[N1] = Pr + (row / 2) * (2 * N1 - NR - 1)

        H[N1, 0] = Heighte1(Pnr[N1], theta, rawpsi, hc, wc, Dv, FOV)
        H[N1, 1] = Heighte2(Pnr[N1], theta, rawpsi, hc, wc, Dv, FOV)
        H[N1, 2] = Heighte3(Pnr[N1], row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
        H[N1, 3] = Heighte4(Pnr[N1], row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
        H[N1, 4] = Heighte5(Pnr[N1], row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
        H[N1, 5] = Heighte6(Pnr[N1], row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)

        #    tantheta(N1, 1) = (H(N1, 1) + OC * MIN) / Dv
        #    tantheta(N1, 2) = (H(N1, 2) + OC * MIN) / Dv
        #    tantheta(N1, 3) = (H(N1, 3) + OC * MIN) / Dv
        #    tantheta(N1, 4) = (H(N1, 4) + OC * MIN) / Dv
        #    tantheta(N1, 5) = (H(N1, 5) + OC * MIN) / Dv
        #    tantheta(N1, 6) = (H(N1, 6) + OC * MIN) / Dv

        if H[N1, 3] > H[N1, 0]:
            H[N1, 3] = H[N1, 0] #H1 obscurs H4 (no shaded soil visible on near-side)
            H[N1, 2] = H[N1, 0] #H1 obscurs H3 (no shaded canopy visible on near-side)
        if H[N1, 5] < H[N1, 1]:
            H[N1, 5] = H[N1, 1] #H2 obscurs H6 (no shaded soil visible on far-side)
            H[N1, 4] = H[N1, 1] #H2 obscurs H5 (no shaded canopy visible on far-side)

        #    If H(N1, 4) > H(N1, 1) Then H(N1, 4) = H(N1, 1) #H1 obscurs H4 (no shaded soil visible on near-side)
        #    If H(N1, 3) < H(N1, 1) Then H(N1, 3) = H(N1, 1) #H1 obscurs H3 (no shaded canopy visible on near-side)
        #    If H(N1, 6) < H(N1, 2) Then H(N1, 6) = H(N1, 2) #H2 obscurs H6 (no shaded soil visible on far-side)
        #    If H(N1, 5) > H(N1, 2) Then H(N1, 5) = H(N1, 2) #H2 obscurs H5 (no shaded canopy visible on far-side)
    
        # N2 As Integer
        for N2 in range(1,NR):   #Account for adjacent rows obscuring chord locations
            if H[(N2 - 1), 1] > H[N2, 3]: H[N2, 3] = H[(N2 - 1), 1]
            #Far side of canopy boundary in row N2-1 obscurs near side of sunlit-shaded soil boundary in row N2
            if H[(N2 - 1), 1] > H[N2, 0]: H[N2, 0] = H[(N2 - 1), 1]
            #Far side of canopy boundary in row N2-1 obscurs near side of canopy in row N2
            if H[(N2 - 1), 1] > H[N2, 2]: H[N2, 1] = H[(N2 - 1), 1]
            #Far side of canopy boundary in row N2-1 obscurs near side of sunlit-shaded canopy boundary in row N2
            if H[N2, 0] < H[(N2 - 1), 5]: H[(N2 - 1), 5] = H[N2, 0]
            #Near side of canopy in row N2 obscurs far side of sunlit-shaded soil boundary in row N2-1
            if H[N2, 0] < H[(N2 - 1), 1]: H[(N2 - 1), 1] = H[N2, 0]
            #Near side of canopy in row N2 obscurs far side of canopy in row N2-1
            if H[N2, 0] < H[(N2 - 1), 4]: H[(N2 - 1), 4] = H[N2, 0]
            #Near side of canopy in row N2 obscurs far side of sunlit-shaded canopy boundary in row N2-1
    
    #Build array of chord areas
    
    # N3 As Integer
    for N3 in range(0,NR):
        # y As Integer
        for y in range(0,6):  #N3 Chord numbers 1 to 6
            if H([N3, y] > 0:
                ac[N3, y] = Chord(psi, ar, br, np.abs(H[N3, y]))
            else:
                ac[N3, y] = Aell - Chord(psi, ar, br, np.abs(H[N3, y]))
        

    #Compute areas of sunlit and shaded soil and canopy visible to radiometer

    # N4 As Integer
    for N4 in range(1,NR - 1):
        fc[N4, 0] = (ac[N4, 2] - ac[N4, 1]) #Sunlit canopy
        #If fc(N4, 1) < 0 Then fc(N4, 1) = 0
        fc[N4, 1] = (ac[N4, 0] - ac[N4, 2]) + ac[N4, 4] - ac[N4, 1] #Shaded canopy
        #If fc(N4, 2) < 0 Then fc(N4, 2) = 0
        fc[N4, 2] = (ac[(N4 - 1), 5] - ac[N4, 3] + ac[N4, 5] - ac[(N4 + 1), 3]) * 0.5 #Sunlit soil
        #If fc(N4, 3) < 0 Then fc(N4, 3) = 0      
        fc[N4, 3] = ac[N4, 3] - ac[N4, 0] + ac[N4, 1] - ac[N4, 5] #Shaded soil
        #If fc(N4, 4) < 0 Then fc(N4, 4) = 0
    fcr[0] = 0
    fcr[1] = 0
    fcr[2] = 0
    fcr[3] = 0
    #Sum areas for each row
    # N5 As Integer
    for N5 in range(1,(NR - 1)):
        fcr[0] = fc[N5, 0] + fcr[0]
        fcr[1] = fc[N5, 1] + fcr[1]
        fcr[2] = fc[N5, 2] + fcr[2]
        fcr[3] = fc[N5, 3] + fcr[3]

    #20 fcs = H(3, OPT)

    fcs = (fcr[OPT]) / Aell
    return fcs
                 
def Heighte1(Pr , theta , rawpsi , hc , wc , Dv , FOV ): 
    #Heighte1 = Distance from center of footprint of radiometer to chord shadow
    #   cast by near-edge of row crop canopy, where crop canopy is modelled as ELLIPSE (m)
    #Pr = Perpendicular distance of radiometer from canopy row center (m)
    #Theta = Zenith angle of radiometer (radians)
    #Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
    #   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    #   to crop row (radians)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #Dv = Vertical height of radiometer relative to soil (m)
    #FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")

    #Constrain 45 < psi < 90
    # psi
                 
    psi = constrainpsi(rawpsi)

    # ac     #Horizonal axis of elliptical canopy (m)
    # bc     #Vertical axis of elliptical canopy (m)
    # X1     #Horizonal distance from canopy ellipse origin to radiometer (m)
    # Y1     #Vertical distance from canopy ellipse origin to radiometer (m)
    ac = wc / 2 / np.sin(psi)
    bc = hc / 2
    X1 = Pr / np.sin(psi)
    Y1 = Dv - bc

    #Find tantheta1, the inverse slope of tangent line from radiometer to near edge of canopy
    # tantheta1 
    tantheta1 = quartic1IRT(X1, Y1, ac, bc)

    # OC     #Horizontal distance from radiometer to center of radiometer footprint (m)
    OC = Dv / 2 * (np.tan(theta + np.atan(1 / 2 / FOV)) + np.tan(theta - np.atan(1 / 2 / FOV)))

    # MIN 
    if np.abs(np.tan(rawpsi)) < 1:
        MIN = np.tan(pi / 2 - psi)
    else:
        MIN = 1
        
    return Dv * tantheta1 - OC * MIN
                 
def Heighte2(Pr , theta , rawpsi , hc , wc , Dv , FOV): 

    #Heighte2 = Distance from center of footprint of radiometer to chord shadow
    #   cast by far-edge of row crop canopy, where crop canopy is modelled as ELLIPSE (m)
    #Pr = Perpendicular distance of radiometer from canopy row center (m)
    #Theta = Zenith angle of radiometer (radians)
    #Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
    #   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    #   to crop row (radians)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #Dv = Vertical height of radiometer relative to soil (m)
    #FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
                 
    #Constrain 45 < psi < 90
    # psi
    psi = constrainpsi(rawpsi)

    # ac     #Horizonal axis of elliptical canopy (m)
    # bc     #Vertical axis of elliptical canopy (m)
    # X1     #Horizonal distance from canopy ellipse origin to radiometer (m)
    # Y1     #Vertical distance from canopy ellipse origin to radiometer (m)
    ac = wc / 2 / np.sin(psi)
    bc = hc / 2
    X1 = Pr / np.sin(psi)
    Y1 = Dv - bc

    #Find tantheta2, the inverse slope of tangent line from radiometer to far edge of canopy
    # tantheta2 
    tantheta2 = quartic2IRT(X1, Y1, ac, bc)

    # OC     #Horizontal distance from radiometer to center of radiometer footprint (m)
    OC = Dv / 2 * (np.tan(theta + np.atan(1 / 2 / FOV)) + np.tan(theta - np.atan(1 / 2 / FOV)))
    # MIN 
    if np.abs(np.tan(rawpsi)) < 1:
        MIN = np.tan(pi / 2 - psi)
    else:
        MIN = 1

    return Dv * tantheta2 - OC * MIN
                 
def Heighte3(Pr , row , theta , rawpsi , hc , wc , Dv , FOV , thetas , psis ):
    #Heighte3 = Distance from center of footprint of radiometer to chord projected by
    #sunlit-shaded boundary on near-side of canopy, where crop canopy is modelled as ELLIPSE (m)
    #Pr = Perpendicular distance of radiometer from canopy row center (m)
    #row = Crop row spacing (m)
    #Theta = Zenith angle of radiometer (radians)
    #Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
    #   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    #   to crop row (radians)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #Dv = Vertical height of radiometer relative to soil (m)
    #FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
    #thetas = Solar zenith angle (radians)
    #psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
    #crop row and pi/2 is looking perpendicular to crop row (radians)
                
    #Constrain 45 < psi < 90
    # psi
    
    psi = constrainpsi(rawpsi)
    # bc     #Vertical axis of elliptical canopy (m)
    # thetasp   #Solar zenith angle perpendicular to canopy (rad)
    # Xs     #Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # Ys     #Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # X3     #Horizontal distance from radiometer to ground-projected
    #sunlit-shaded boundary on canopy (m)
    bc = hc / 2
    thetasp = -thetas * np.sin(psis)
    Xs = wc / 2 / np.sqrt(1 + (bc ** 2) / ((wc / 2) ** 2) * ((np.tan(thetasp)) ** 2))    #Xs is positive
    Ys = -(bc ** 2) / ((wc / 2) ** 2) * Xs * np.tan(thetasp) #Ys is positive or negative
    #Determine critical perpendicular solar zenith angle,
    #beyond which results in adjacent row shading
    # Xscr   #Horizontal distance from canopy ellipse origin to tangent of sunray
    #along thetasp (m)
    # Yscr   #Vertical distance from canopy ellipse origin to tangent of sunray
    #along thetasp (m)
    # thetaspcr      #Critical perpendicular solar zenith angle

    Xscr = 2 * ((wc / 2) ** 2) / row     #Xscr is positive
    Yscr = -np.sqrt(((bc ** 2) * Xscr * (row - 2 * Xscr)) / 2 / ((wc / 2) ** 2))  #Yscr is negative
    thetaspcr = np.atan(-((wc / 2) ** 2) * Yscr / (bc ** 2) / Xscr)   #thetascr is positive
                 
    if thetasp > thetaspcr:     #Shadows cast by adjacent rows and H3 is raised
        
        m3 = 1 / np.tan(thetasp)
        b3 = -Ys - m3 * (row - Xs)
        AA = (bc ** 2) + ((wc / 2) ** 2) * (m3 ** 2)
        BB = 2 * m3 * b3 * ((wc / 2) ** 2)
        CC = ((wc / 2) ** 2) * (b3 ** 2) - ((wc / 2) ** 2) * (bc ** 2)
        Xs3 = (-BB + np.sqrt((BB ** 2) - 4 * AA * CC)) / (2 * AA)    #Positive root (negative root taken for H5)
        Ys3 = m3 * Xs3 + b3
        X3 = Dv / np.sin(psi) * ((Pr - Xs3) / (Dv - bc - Ys3))

    else:    #Compute X3 as normal (no shading by adjacent row)
        X3 = Dv / np.sin(psi) * ((Pr - Xs) / (Dv - bc - Ys))

    # OC     #Horizontal distance from radiometer to center of radiometer footprint (m)
    OC = Dv / 2 * (np.tan(theta + np.atan(1 / 2 / FOV)) + np.tan(theta - np.atan(1 / 2 / FOV)))
            
    if np.abs(np.tan(rawpsi)) < 1:
        MIN = np.tan(pi / 2 - psi)
    else:
        MIN = 1
    
    return X3 - OC * MIN
                 
def Heighte4(Pr , row , theta , rawpsi , hc , wc , Dv , FOV , thetas , psis ): 
    #Heighte4 = Distance from center of footprint of radiometer to chord projected by
    #sunlit-shaded soil boundary on near-side of canopy, where crop canopy is modelled as ELLIPSE (m)
    #Pr = Perpendicular distance of radiometer from canopy row center (m)
    #row = Crop row spacing (m)
    #Theta = Zenith angle of radiometer (radians)
    #Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
    #   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    #   to crop row (radians)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #Dv = Vertical height of radiometer relative to soil (m)
    #FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
    #thetas = Solar zenith angle (radians)
    #psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
    #crop row and pi/2 is looking perpendicular to crop row (radians)
                
                
    #Constrain 45 < psi < 90
    # psi 
    psi = constrainpsi(rawpsi)
                 
    # bc     #Vertical axis of elliptical canopy (m)
    # thetasp   #Solar zenith angle perpendicular to canopy (rad)
    # Xs     #Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # Ys     #Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # X4     #Horizontal distance from radiometer to ground-projected
    #sunlit-shaded soil boundary on near-side of canopy (m)
    bc = hc / 2
    thetasp = -thetas * np.sin(psis)
    Xs = wc / 2 / np.sqrt(1 + (bc ** 2) / ((wc / 2) ** 2) * ((np.tan(thetasp)) ** 2))
    Ys = -(bc ** 2) / ((wc / 2) ** 2) * Xs * np.tan(thetasp)
    X4 = (Pr - Xs + (np.tan(thetasp)) * (bc + Ys)) / np.sin(psi)
                 
    # OC     #Horizontal distance from radiometer to center of radiometer footprint (m)
    OC = Dv / 2 * (np.tan(theta + np.atan(1 / 2 / FOV)) + np.tan(theta - np.atan(1 / 2 / FOV)))
                 
    # MIN 
    if np.abs(np.tan(rawpsi)) < 1:
        MIN = np.tan(pi / 2 - psi)
    else:
        MIN = 1

    return X4 - OC * MIN

def Heighte5(Pr , row , theta , rawpsi , hc , wc , Dv , FOV , thetas , psis ): 
    #Heighte5 = Distance from center of footprint of radiometer to chord projected by
    #sunlit-shaded boundary on far-side of canopy, where crop canopy is modelled as ELLIPSE (m)
    #Pr = Perpendicular distance of radiometer from canopy row center (m)
    #row = Crop row spacing (m)
    #Theta = Zenith angle of radiometer (radians)
    #Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
    #   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    #   to crop row (radians)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #Dv = Vertical height of radiometer relative to soil (m)
    #FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
    #thetas = Solar zenith angle (radians)
    #psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
    #crop row and pi/2 is looking perpendicular to crop row (radians)
                
    #Constrain 45 < psi < 90
    # psi    
    psi = constrainpsi(rawpsi)

    # bc     #Vertical axis of elliptical canopy (m)
    # thetasp   #Solar zenith angle perpendicular to canopy (rad)
    # Xs     #Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # Ys     #Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # X5     #Horizontal distance from radiometer to ground-projected
                                             #sunlit-shaded boundary on canopy (m)
    bc = hc / 2
    thetasp = -thetas * np.sin(psis)
    Xs = -wc / 2 / np.sqrt(1 + (bc ** 2) / ((wc / 2) ** 2) * ((np.tan(thetasp)) ** 2)) #Xs is negative (positive for H3)
    Ys = -(bc ** 2) / ((wc / 2) ** 2) * Xs * np.tan(thetasp) #Ys is positive or negative

    #Determine critical perpendicular solar zenith angle,
    #beyond which results in adjacent row shading
    # Xscr   #Horizontal distance from canopy ellipse origin to tangent of sunray
    #along thetasp (m)
    # Yscr   #Vertical distance from canopy ellipse origin to tangent of sunray
    #along thetasp (m)
    # thetaspcr      #Critical perpendicular solar zenith angle

    Xscr = -2 * ((wc / 2) ** 2) / row     #Xscr is negative (positive for H3)
    Yscr = -np.sqrt(((bc ** 2) * Xscr * (row + 2 * Xscr)) / -2 / ((wc / 2) ** 2))  #Yscr is negative; note -2 and row+2*Xscr
    thetaspcr = np.atan(-((wc / 2) ** 2) * Yscr / (bc ** 2) / Xscr)   #thetascr is negative

    if thetasp < thetaspcr:     #Shadows cast by adjacent rows and H5 is raised.
                 #NOTE: thetascr is negative (thetascr for H3 is positive)
        
        m5 = 1 / np.tan(thetasp)
        b5 = -Ys + m5 * (row + Xs) #NOTE: signs are negative for H3
        AA = (bc ** 2) + ((wc / 2) ** 2) * (m5 ** 2)
        BB = 2 * m5 * b5 * ((wc / 2) ** 2)
        CC = ((wc / 2) ** 2) * (b5 ** 2) - ((wc / 2) ** 2) * (bc ** 2)
        Xs5 = (-BB - np.sqrt((BB ** 2) - 4 * AA * CC)) / (2 * AA)    #NOTE: Negative root (H3 positive)
        Ys5 = m5 * Xs5 + b5
        X5 = Dv / np.sin(psi) * ((Pr - Xs5) / (Dv - bc - Ys5))

    else:    #Compute X5 as normal (no shading by adjacent row)

        X5 = Dv / np.sin(psi) * ((Pr - Xs) / (Dv - bc - Ys))

    # OC     #Horizontal distance from radiometer to center of radiometer footprint (m)
    OC = Dv / 2 * (np.tan(theta + np.atan(1 / 2 / FOV)) + np.tan(theta - np.atan(1 / 2 / FOV)))
                 
    if np.abs(np.tan(rawpsi)) < 1:
        MIN = np.tan(pi / 2 - psi)
    else:
        MIN = 1

    return X5 - OC * MIN
                 
def Heighte6(Pr , row , theta , rawpsi , hc , wc , Dv , FOV , thetas , psis ):
    #Heighte6 = Distance from center of footprint of radiometer to chord projected by
    #sunlit-shaded soil boundary on far-side of canopy, where crop canopy is modelled as ELLIPSE (m)
    #Pr = Perpendicular distance of radiometer from canopy row center (m)
    #row = Crop row spacing (m)
    #Theta = Zenith angle of radiometer (radians)
    #Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
    #   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    #   to crop row (radians)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #Dv = Vertical height of radiometer relative to soil (m)
    #FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
    #thetas = Solar zenith angle (radians)
    #psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
    #crop row and pi/2 is looking perpendicular to crop row (radians)
    #Constrain 45 < psi < 90
    # psi
    psi = constrainpsi(rawpsi)
                 
    # bc     #Vertical axis of elliptical canopy (m)
    # thetasp   #Solar zenith angle perpendicular to canopy (rad)
    # Xs     #Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # Ys     #Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
    # X6     #Horizontal distance from radiometer to ground-projected
    #sunlit-shaded soil boundary on near-side of canopy (m)
    bc = hc / 2
    thetasp = -thetas * np.sin(psis)
    Xs = -wc / 2 / np.sqrt(1 + (bc ** 2) / ((wc / 2) ** 2) * ((np.tan(thetasp)) ** 2))    #NOTE: negative sign (H4 positive)
    Ys = -(bc ** 2) / ((wc / 2) ** 2) * Xs * np.tan(thetasp)
    X6 = (Pr - Xs + (np.tan(thetasp)) * (bc + Ys)) / np.sin(psi)

    # OC     #Horizontal distance from radiometer to center of radiometer footprint (m)
    OC = Dv / 2 * (np.tan(theta + np.atan(1 / 2 / FOV)) + np.tan(theta - np.atan(1 / 2 / FOV)))

     
    if np.abs(np.tan(rawpsi)) < 1:
        MIN = np.tan(pi / 2 - psi)
    else:
        MIN = 1

    return X6 - OC * MIN

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

def Chord(rawpsi , MAJOR , MINOR , H ):               
    #def chord to compute the chord area of an ellipse
    # X      #Axis used to compute chord area where H crosses (m)
    # y      #Axis perpendicular to X (m)
    # psi    #Angle of chord wrt Y-axis, constrained to 45 < psi < 90 (rad)
                 
    #Constrain 45 < psi < 90, specify X and Y
    
    if np.abs(rawpsi) < 45 * pi / 180:
        psi = pi / 2 - (np.abs(rawpsi))
        X = MINOR
        y = MAJOR
    else:
        if np.abs(rawpsi) > 135 * pi / 180:
            psi = (np.abs(rawpsi)) - pi / 2
            X = MINOR
            y = MAJOR
        else:
            if np.abs(rawpsi) > 90 * pi / 180:
                psi = pi - np.abs(rawpsi)
                X = MAJOR
                y = MINOR
            else:
                psi = np.abs(rawpsi)
                X = MAJOR
                y = MINOR
    #Compute distance from ellipse center to tangent (maximum possible H to form a chord)
    ## ec  #Ellipse eccentricity (m)
    ## asin1  #Parameter used to compute inverse sine
    ## alpha1     #Internal angle (rad)
    ## HH1     #Radius of internal construction circle (m)
    ## T1     #Distance from ellipse center to tangent (m)

    #ec = np.sqrt(1 - (X ** 2) / (y ** 2))
    #asin1 = ec * np.sin(psi + pi / 2)
    #alpha1 = np.atan(asin1 / np.sqrt(-asin1 * asin1 + 1))
    #HH1 = y * (np.sin(pi / 2 - psi - alpha1)) / (np.sin(psi + pi / 2))
    #T1 = ec * y + HH1 / np.sin(psi)

    #Compute T1
    # T1     #Distance from radiometer ellipse footprint center to line
    #   tangent to radiometer ellipse for given azimuth and zenith angles (m).

    if (np.abs(psi - 90 * pi / 180) < 0.01):   #90 degree radiometer azimuth
        T1 = y
        if H > T1:  #Chord outside of Ellipse
            return 0
            
        else:            #Case 2 with equilateral triangle
            return X * y * np.atan((np.sqrt(y ** 2 - H ** 2)) / H) - H * X / y * np.sqrt(y ** 2 - H ** 2)
            
    else:
        if np.abs(psi) < 0.01:     #ZERO degree radiometer azimuth
            T1 = X
            if H > T1:  #Chord outside of Ellipse
                return 0
                
            else:            #Case 4 with equilateral triangle
                return X * y * np.atan((np.sqrt(X ** 2 - H ** 2)) / H) - H * y / X * np.sqrt(X ** 2 - H ** 2)
                 
        else: #Radiometer azimuth between ZERO and 90 degrees
            xtr1 = y / np.sqrt(1 + (X ** 2) / (y ** 2) / ((np.tan(psi)) ** 2))
            ytr1 = (X ** 2) / (y ** 2) * xtr1 / np.tan(psi)
            T1 = xtr1 + ytr1 / np.tan(psi)
    #Determine if H cuts a chord through ellipse
    if H > T1:
        return 0
        

    if np.abs(X - y) < 0.001: #Ellipse is a circle
        hc = H * np.sin(psi)
        yy = hc / X
        acosyy = np.atan(-yy / np.sqrt(-yy * yy + 1)) + 2 * np.atan(1)
        return (X ** 2) * acosyy - hc * np.sqrt(X ** 2 - hc ** 2)
        
    #Chord = Chord area (m2)
    #psi = Azimuth angle (radians)
    #Y = Major axis (m)
    #X = Minor axis (m)
    #H = Distance from ellipse center to chord along major axis Y (m)
    #C = Case number, where
    #CASE 1: X > H*tan(psi); H < Y
    #CASE 2: X < H*tan(psi); H < Y
    #CASE 3: X < H*tan(psi); H > Y
    #CASE 4: X > H*tan(psi); H > Y

    #Determine chord case
    # CC As Integer
    if H < y:
        if X > H * np.tan(psi):
            CC = 1#
        else:
            CC = 2#
    else:
        if X > H * np.tan(psi):
            CC = 4#
        else:
            CC = 3#

    #See diagrams for variables used below
    # aq     #Used in quadratic equation for q
    # bq     #Used in quadratic equation for q
    # cq     #Used in quadratic equation for q
    # Q 
    # s 
    # alpha 
    # A 

    # at     #Used in quadratic equation for t
    # bt     #Used in quadratic equation for t
    # ct     #Used in quadratic equation for t
    #Declare sector and triangle variables, where Chord = Aes - Aet
    # Aes    #Area enclosed by sector (m2)
    # Aet    #Area enclosed by triangle (m2)
                 
    if CC = 1:
        aq = (y ** 2) / (X ** 2) + (1 / np.tan(psi)) ** 2
        bq = 2 * H / np.tan(psi)
        cq = (H ** 2) - (y ** 2)
        Q = (-bq + np.sqrt((bq ** 2) - 4 * aq * cq)) / (2 * aq)
        s = Q / np.tan(psi)
        alpha = np.atan(Q / (H + s))
        A = np.sqrt((Q ** 2) + ((H + s) ** 2))

        at = (X ** 2) / (y ** 2) + (np.tan(psi)) ** 2
        bt = 2 * H * ((np.tan(psi)) ** 2)   #negative for Cases 2 & 3
        ct = (H ** 2) * ((np.tan(psi)) ** 2) - (X ** 2)
        t = (-bt + np.sqrt((bt ** 2) - 4 * at * ct)) / (2 * at)
        u = (H + t) * np.tan(psi)  #-t for Cases 2 & 3
        beta = np.atan(t / u)       #Inverse for Cases 2 & 3
        b = np.sqrt((t ** 2) + (u ** 2))
        Aes = X * y / 2 * (np.atan(y / X * np.tan(alpha)) + np.atan(X / y * np.tan(beta)) + pi / 2)
        Aet = A * b / 2 * np.sin(alpha + beta + pi / 2)
        return Aes - Aet
        

    if CC = 2:
        aq = (y ** 2) / (X ** 2) + (1 / np.tan(psi)) ** 2
        bq = 2 * H / np.tan(psi)
        cq = (H ** 2) - (y ** 2)
        Q = (-bq + np.sqrt((bq ** 2) - 4 * aq * cq)) / (2 * aq)
        s = Q / np.tan(psi)
        alpha = np.atan(Q / (H + s))
        A = np.sqrt((Q ** 2) + ((H + s) ** 2))
        at = (X ** 2) / (y ** 2) + (np.tan(psi)) ** 2
        bt = -2 * H * ((np.tan(psi)) ** 2)
        ct = (H ** 2) * ((np.tan(psi)) ** 2) - (X ** 2)
        t = (-bt - np.sqrt((bt ** 2) - 4 * at * ct)) / (2 * at)
        u = (H - t) * np.tan(psi)  #+t for case 1
        beta = np.atan(u / t)       #Inverse for case 1
        b = np.sqrt((t ** 2) + (u ** 2))
        Aes = X * y / 2 * (np.atan(y / X * np.tan(alpha)) + np.atan(y / X * np.tan(beta)))
        Aet = A * b / 2 * np.sin(alpha + beta)
        return Aes - Aet
        
    if CC = 3:
        aq = (y ** 2) / (X ** 2) + (1 / np.tan(psi)) ** 2 #tan for Cases 1 & 2
        bq = -2 * H / np.tan(psi)                  #positive and tan for Cases 1 & 2
        cq = (H ** 2) - (y ** 2)
        Q = (-bq - np.sqrt((bq ** 2) - 4 * aq * cq)) / (2 * aq)
        s = Q / np.tan(psi)            #tan for Cases 1 & 2
        alpha = np.atan(Q / (H - s))    #+s for Cases 1 & 2
        A = np.sqrt((Q ** 2) + ((H - s) ** 2))   #+s for Cases 1 & 2
        at = (X ** 2) / (y ** 2) + (np.tan(psi)) ** 2
        bt = -2 * H * ((np.tan(psi)) ** 2)
        ct = (H ** 2) * ((np.tan(psi)) ** 2) - (X ** 2)
        t = (-bt - np.sqrt((bt ** 2) - 4 * at * ct)) / (2 * at)
        u = (H - t) * np.tan(psi)  #+t for case 1
        beta = np.atan(u / t)       #Inverse for case 1
        b = np.sqrt((t ** 2) + (u ** 2))
        Aes = X * y / 2 * (np.atan(y / X * np.tan(beta)) - np.atan(y / X * np.tan(alpha)))
        Aet = A * b / 2 * np.sin(beta - alpha)
        return Aes - Aet
        
    if CC = 4:
        aq = (y ** 2) / (X ** 2) + (1 / np.tan(psi)) ** 2 #tan for Cases 1 & 2
        bq = -2 * H / np.tan(psi)                  #positive and tan for Cases 1 & 2
        cq = (H ** 2) - (y ** 2)
        Q = (-bq - np.sqrt((bq ** 2) - 4 * aq * cq)) / (2 * aq)
        s = Q / np.tan(psi)            #tan for Cases 1 & 2
        alpha = np.atan(Q / (H - s))    #+s for Cases 1 & 2
        A = np.sqrt((Q ** 2) + ((H - s) ** 2))   #+s for Cases 1 & 2
        at = (X ** 2) / (y ** 2) + (np.tan(psi)) ** 2
        bt = 2 * H * ((np.tan(psi)) ** 2)   #negative for Cases 2 & 3
        ct = (H ** 2) * ((np.tan(psi)) ** 2) - (X ** 2)
        t = (-bt + np.sqrt((bt ** 2) - 4 * at * ct)) / (2 * at)
        u = (H + t) * np.tan(psi)  #-t for Cases 2 & 3
        beta = np.atan(t / u)       #Inverse for Cases 2 & 3
        b = np.sqrt((t ** 2) + (u ** 2))
        Aes = X * y / 2 * (np.atan(X / y * np.tan(beta)) + pi / 2 - np.atan(y / X * np.tan(alpha)))
        Aet = A * b / 2 * np.sin(-alpha + beta + pi / 2)

        return Aes - Aet        
