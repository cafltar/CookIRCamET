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

class canopy:
    def __init__(self,inputs_obj,solar_obj):
        self.io = inputs_obj
        self.sol = solar_obj
        self.thetas = self.sol.solarzenith#from running solar class methods solarzenith - deg
        self.psis = self.sol.solarazimuth #deg
        self.thetar = self.io.radiometer_zenith#from input parameters
        self.psir = self.io.radiometer_azimuth
        
    def kb(self,theta):
        #canopy extinction coefficient - 
        #using procedure of Campbell and Norman (1998), Chapter 15 (CN98)
        return (np.sqrt(self.io.XE ** 2 + (np.tan(theta)) ** 2)) / (self.io.XE + 1.774 * (self.io.XE + 1.182) ** -0.733)

    def fsis_calc(self):
        #fsis - interrow shaded fraction (Colaizzi 2016)
        self.fsis = (self.io.f_soil_shade + self.io.f_res_shade)/(self.io.f_soil_shade + self.io.f_res_shade + self.io.f_soil_sun + self.io.f_res_sun)
    def fdhc_calc(self):
        # fdhc = downward hemispherical view factor of canopy
        #the downward hemispherical view factor of the canopy is that which is sunlit+shaded
        self.fdhc = self.io.f_veg_shade + self.io.f_veg_shade
        
    def mrf_plf_calc(self):
        ac = self.io.hc / 2    # Vertical semiaxis of crop ellipse (m)
        bc = self.io.wc / 2    # Horizontal semiaxis of crop ellipse (m)
        # Calculate psird, the radiometer azimuth relative to a crop row,
        # constrain 0 < psird < 90 deg, where 0 deg is parallel and 90 deg
        # is perdendicular to the crop row
        # Constrain 0 =< Rowdir < +180
        row_dirc = acosd(cosd(self.io.row_dir))
        # Calculate azimuthsunrow, the solar azimuth relative to a crop row,
        # constrain 0 < azimuthsunrow < 90 deg
        asr = abs(asind(sind(self.sol.solarazimuth - row_dirc)))
        # azimuthsunrow cannot equal 0 or +90 to avoid domain errors
        asr[(asr==0)|(asr==90)] = 1e-12
        azimuthsunrow = asr
        psird = abs(asind(sind(self.io.rad_azimuth - row_dirc)))

        # psird cannot equal 0 or 90 deg to avoid domain errors.
        psird[(psird==0)|(psird==90)] = 1e-12

        # Convert from degrees to radians
        psir = np.deg2rad(psirdc)
        thetar = np.deg2rad(thetard)

        # thetarp = Directional radiometer zenith angle projected perpendicular
        # to crop row thetarp can be positive or negative (rad)
        thetarp = np.atan((np.tan(thetar)) * np.sin(psir))
        thetasp = np.atan(tand(self.sol.solarzenith) * sind(azimuthsunrow))

        n = 20 # Limit maximum rows to 20
        thetaspMR = thetasp * np.ones(length(thetasp),n)
        thetarpMR = thetarp * np.ones(length(thetasp),n)
        rowarray = np.ones((length(thetasp),n))*np.arange(1,n)

        # XcrMR = Horizontal distance from canopy ellipse origin to tangent of
        # sunray or directional radiometer view along thetaspcr or thetarpcr,
        # respectively, for multiple rows (n). XscrMR is positive (m)
        XcrMR = 2 * ((bc) ** 2) / (rowarray * self.io.row_width)

        # YcrMR = Vertical distance from canopy ellipse origin to tangent of
        # sunray or directional radiometer view along thetaspcr or thetarpcr,
        # respectively, for multiple rows (n). YcrMR is positive (m)
        YcrMR = np.sqrt(((ac ** 2)*XcrMR*(rowarray * self.io.row_width - 2*XcrMR)) /2/(bc**2))

        # thetapcrMR = Critical perpendicular solar zenith angle, where greater
        # angles result in adjacent row shading for multiple rows,
        # or critical perpendicular directional radiometer view angle, where greater
        # angles result in ajacent rows obscuring view for multiple rows.
        # thetapcrMR is positive (rad)
        thetapcrMR = np.atan((rowarray * self.io.row_width - 2*XcrMR)/(2*YcrMR))

        testarrays = rowarray*(thetaspMR > thetapcrMR)
        testarrayr = rowarray*(thetarpMR > thetapcrMR)

        # Calculate MRFs (solar) and MRFr (directional radiometer)
        # For solar, MultiRow >= 1 daylight only
        MRFs = (self.sol.solarzenith<89) * max(1,testarrays.max(axis=1))
        MRFr = max(1, testarrayr.max(axis=1))

        # Calculate PLFs (solar) and PLFr (directional radiometer)
        Ysp = ac * bc /np.sqrt((ac ** 2) * ((np.tan(thetasp)) ** 2) + (bc ** 2))
        Xsp = Ysp* np.tan(thetasp)
        Zsp = Xsp / np.abs(np.tan(azimuthsunrow))
        PLFs = (self.sol.solarzenith<89) * (np.sqrt(Xsp ** 2 + Ysp ** 2 + Zsp ** 2)) / ac

        Yrp = ac * bc / np.sqrt((ac ** 2) * ((np.tan(thetarp)) ** 2) + (bc ** 2))
        Xrp = Yrp * np.tan(thetarp)
        Zrp = Xrp / np.abs(np.tan(psir))
        PLFr = (np.sqrt(Xrp ** 2 + Yrp ** 2 + Zrp ** 2)) / ac
        
        self.mrf_s = MRFs
        self.mrf_r = MRFr

        self.plf_s = PLFs
        self.plf_r = PLFr

    def taudir(self):
        # DIRECT BEAM TRANSMITTANCE AND REFLECTANCE
        # Kbs from solar class
        # Calculate direct beam canopy reflectance for visible and near infrared
        # spectra (CN98, eq. 15.7, p. 255 CN98 eq. 15.8, p. 257)
        rhohorv = (1 - ((self.io.zeta_vis) ** 0.5)) / (1 + ((self.io.zeta_vis) ** 0.5))
        rhohorn = (1 - ((self.io.zeta_nir) ** 0.5)) / (1 + ((self.io.zeta_nir) ** 0.5))
        Kbs = kb(np.deg2rad(self.thetas))
        Kbr = kb(np.deg2rad(self.thetar))
        
        rhodirv = 2 * Kbs * rhohorv / (Kbs + 1)
        rhodirn = 2 * Kbs * rhohorn / (Kbs + 1)
        
        # Calculate direct beam canopy transmittance for visible and near infrared
        # spectra (CN98, eq. 15.11, p. 257)
        self.taudir_vis = (((rhodirv ** 2) - 1) * np.exp(-(self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * self.plf_s * self.mrf_s)) / (((rhodirv * self.io.alb_vis) - 1) + rhodirv * (rhodirv - self.io.alb_vis) * np.exp(-2 * (self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * self.plf_s * self.mrf_s))
        self.taudir_nir = (((rhodirn ** 2) - 1) * np.exp(-(self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * self.plf_s * self.mrf_s)) / (((rhodirn * self.io.alb_nir) - 1) + rhodirn * (rhodirn - self.io.alb_nir) * np.exp(-2 * (self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * self.plf_s * self.mrf_s))

        # Calculate direct beam canopy reflectance for visible and near infrared
        # spectra (CN98, eq. 15.9, p. 257)
        xidirv = ((rhodirv - self.io.self.io.alb_vis)/ (rhodirv * self.io.self.io.alb_vis - 1)) * np.exp(-2 * (self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * self.plf_s * self.mrf_s)
        xidirn = ((rhodirn - self.io.self.io.alb_nir) / (rhodirn * self.io.self.io.alb_nir - 1)) * exp(-2 * (self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * self.plf_s * self.mrf_s)
        self.rhocsdir_vis = (rhodirv + xidirv) / (1 + xidirv * rhodirv)
        self.rhocsdir_nir = (rhodirn + xidirn) / (1 + xidirn * rhodirn)
       
    def taudiff(self): 
        #def taudiff to compute transmittance of DIFFUSE radiation through the canopy
        #by integrating taudir over all solar zenith and azimuth angles
        # ***********************************************************************

        # DIFFUSE TRANSMITTANCE AND REFLECTANCE

        # Calculate multiple row factor for a solar beam through the canopy
        # for each solar zenith and azimuth element (Colaizzi et al. 2012)
        psisi = np.arange(5,85,5).reshape(1,-1) # Solar azimuth vector
        thetasi = np.ones((len(self.io.hc),len(psisi))) * psisi # Solar zenith array (2D)
        # Solar azimuth array (3D)
        psisi3 = np.ones(*thetasi.shape,1) * psisi.reshape(1, -1,1)
        # Projected solar zenith array (3D)
        tanthetaspi = ((tand(thetasi[:,:,np.newaxis])) * sind(psisi3))
        # Multiple row factor for solar beam (3D)
        MRFsi = max(1, np.floor(tanthetaspi * self.io.hc / self.io.row_width))

        # Calculate path length fraction for a solar beam through the canopy
        # for each solar zenith and azimuth element (Colaizzi et al. 2012) (3D)
        Yspi = ac * bc / np.sqrt((ac ** 2) * (tanthetaspi ** 2) + (bc ** 2))
        Xspi = Yspi * tanthetaspi
        Zspi = Xspi / abs(tand(psisi3))
        PLFsi = (np.sqrt(Xspi ** 2 + Yspi ** 2 + Zspi ** 2)) / ac

        # Calculate direct beam extinction coefficient for each solar zenith
        # element (CN98, 15.4, p. 251) (2D, independent of solar azimuth)
        Kbsi = kb(thetasi)
        # Calculate direct beam canopy reflectance for visible and near infrared
        # spectra for each solar zenith element (CN98, eq. 15.7, p. 255
        # CN98 eq. 15.8, p. 257) (2D, independent of solar azimuth)
        rhodirvi = 2 * Kbsi * rhohorv / (Kbsi + 1)
        rhodirni = 2 * Kbsi * rhohorn / (Kbsi + 1)

        # Calculate direct beam canopy transmittance for visible and near infrared
        # spectra for each solar zenith and azimuth element (CN98, eq. 15.11, p. 257)
        taudirvi = (((rhodirvi ** 2) - 1) * np.exp(-(self.io.zeta_vis ** 0.5) * Kbsi * self.io.LAIL * PLFsi * MRFsi)) / (((rhodirvi * self.io.alb_vis) - 1) + rhodirvi * (rhodirvi - self.io.alb_vis) * np.exp(-2 * (self.io.zeta_vis ** 0.5) * Kbsi * self.io.LAIL * PLFsi * MRFsi))
        taudirni = (((rhodirni ** 2) - 1) * np.exp(-(self.io.zeta_nir ** 0.5) * Kbsi * self.io.LAIL * PLFsi * MRFsi)) / (((rhodirni * self.io.alb_nir) - 1) + rhodirni * (rhodirni - self.io.alb_nir) * np.exp(-2 * (self.io.zeta_nir ** 0.5) * Kbsi * self.io.LAIL * PLFsi * MRFsi))

        # Calculate diffuse canopy transmittance for visible and near infrared
        # spectra by integrating taudirvi and taudirni, repsectively
        taudirvi1 = 4 * (1 / 90) * (1 / 90) * ((sind(thetasi)) ** 2) * ((cosd(thetasi)) ** 2) * taudirvi
        taudirni1 = 4 * (1 / 90) * (1 / 90) * ((sind(thetasi)) ** 2) * ((cosd(thetasi)) ** 2) * taudirni
        self.taudiff_vis = np.sum(taudirvi1, axis=[1, 2])
        self.taudiff_nir = np.sum(taudirni1, [1, 2])


        # Calculate direct beam canopy + soil reflectance for visible and near infrared
        # spectra for each solar zenith and azimuth element (CN98, eq. 15.9, p. 257)
        xidirvi = ((rhodirvi - self.io.alb_vis) / (rhodirvi * self.io.alb_vis - 1)) * np.exp(-2 * (self.io.zeta_vis ** 0.5) * Kbsi * self.io.LAIL * PLFsi * MRFsi)
        xidirni = ((rhodirni - self.io.alb_nir) / (rhodirni * self.io.alb_nir - 1)) * np.exp(-2 * (self.io.zeta_nir ** 0.5) * Kbsi * self.io.LAIL * PLFsi * MRFsi)
        rhocsdirvi = (rhodirvi + xidirvi) / (1 + xidirvi * rhodirvi)
        rhocsdirni = (rhodirni + xidirni) / (1 + xidirni * rhodirni)

        # Calculate diffuse canopy reflectance for visible and near infrared
        # spectra by integrating rhocsdirvi and rhocsdirni, repsectively
        rhocsdirvi1 = 4 * (5 / 85) * (5 / 85) * ((sind(thetasi)) ** 2) * ((cosd(thetasi)) ** 2) * rhocsdirvi
        rhocsdirni1 = 4 * (5 / 85) * (5 / 85) * ((sind(thetasi)) ** 2) * ((cosd(thetasi)) ** 2) * rhocsdirni
        self.rhocsdiff_vis = np.sum(rhocsdirvi1, [1, 2])
        self.rhocsdiff_nir = np.sum(rhocsdirni1, [1, 2])
 
