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
        self.thetar = self.io.radiometer_zenith#from running solar class methods
        self.psir = self.io.radiometer_azimuth
        
    def kb(self,theta):
        #canopy extinction coefficient - 
        #using procedure of Campbell and Norman (1998), Chapter 15 (CN98)
        return (np.sqrt(self.io.XE ** 2 + (np.tan(theta)) ** 2)) / (self.io.XE + 1.774 * (self.io.XE + 1.182) ** -0.733)

    def fcsolar_calc(self):
        #fcsolar = Planar solar view factor of crop row vegetation,
        solarzenith = min(89,self.thetas) # Constrain solarzenith < 89 deg
        fcs1 = solarzenith < 89 # fcs = 0 unless sun above horizon

        # Constrain 0 =< Rowdir < +180
        rowdirc = acosd(cosd(io.row_dir))
        # Calculate azimuthsunrow, the solar azimuth relative to a crop row,
        # constrain 0 < azimuthsunrow < 90 deg
        asr = abs(asind(sind(self.psis - rowdirc)))
        # azimuthsunrow cannot equal 0 or +90 to avoid domain errors
        azimuthsunrow = asr
        azimuthsunrow[asr==0] = azimuthsunrow[asr==0]+1e-9
        azimuthsunrow[asr==90] = azimuthsunrow[asr==90]+1e-9 

        # Calculations
        ac = hc / 2    # Vertical semiaxis of crop ellipse (m)
        bc = wc / 2    # Horizontal semiaxis of crop ellipse (m)
        # thetasp = Solar zenith angle projected perpendicular to crop row
        thetasp = atand(tand(solarzenith) * sind(azimuthsunrow))

        # Xscr = Horizontal distance from canopy ellipse origin to tangent of sunray
        # along thetaspcr (m)
        # Yscr = Vertical distance from canopy ellipse origin to tangent of sunray
        # along thetaspcr (m)
        # thetaspcr = Critical perpendicular solar zenith angle, where greater angles
        # result in adjacent row shading (rad)

        Xscr = 2 * ((bc) ** 2) / (self.io.row_width)   # Xscr is positive
        Yscr = np.sqrt(((ac ** 2)*Xscr*(self.io.row_width - 2*Xscr)) /2/((bc)**2))
        # Yscr is positive
        thetaspcr = np.atan((self.io.row_width - 2*Xscr)/(2*Yscr)) # thetascr is positive

        # Constrain fcs = 1 for low sun elevation
        fcs2 = abs(thetasp) >= thetaspcr

        # Xs = Horizontal distance from canopy ellipse origin to
        # tangent of sunray along thetasp (m)
        # Ys = Vertical distance from canopy ellipse origin to
        # tangent of sunray along thetasp (m)

        Xs = bc/sqrt(1+(ac**2)/((bc)**2)*((np.tan(thetasp))**2)) # Xs is positive
        Ys = (ac**2)/((bc)**2)*Xs*np.tan(thetasp) # Ys is positive or negative

        fcs3 = min(1,((2*Xs + 2*Ys*np.tan(thetasp))/self.io.row_width))
        fcsolar = fcs1*max(fcs2,fcs3)

        self.fcsolar = fcsolar
        
    def mrf_calc(self):
        self.mrf_s =
        self.mrf_r =

    def plf_calc(self):
        self.plf_s =
        self.plf_r =

    def fdhc_calc(self):
        # OUTPUT
        # fdhc = downward hemispherical view factor of canopy

        # Create 3D arrays to integrate azimuth (0 to 90 deg) by number of
        # interrows visible to radiometer

        # Build azimuth element array from 0 to 90 degrees, step psiele
        psiele = 5 # Differential azimuth element (deg)
        psiarray = np.ones(size(ac)) * (psiele:psiele:(90-psiele))
        # Calculate number of perpendicular interrows visible below the radiometer
        NRh = ceil(Vh * (sqrt(1 - 4 * (bc ** 2) / (self.io.row_width ** 2))) / (2 * ac))
        NRhmax = min(10,max(NRh)) # Limit array to 10 crop rows
        # Make array with dimensions (:,:,NRhmax)
        AA = np.arange(1,NRhmax)
        BB = reshape(AA, 1, 1, [])
        NRharray = (np.ones(length(psiarray),width(psiarray),NRhmax)) * BB - 1

        # Calculate quartic coefficients to be used in quartic function:
        #   (Ax^4 + Bx^3 + Cx^2 + Dx + E = 0)
        X1 = (self.io.row_width * (NRharray) - Ph) ./ sind(psiarray)
        Y1 = Vh - ac
        bcr = bc / sind(psiarray)

        A = (Y1 ** 2) * (ac ** 2) - (ac ** 4)
        B = -2 * X1 * Y1 * (ac ** 2)
        C = (Y1 ** 2) * (bcr ** 2) + (X1 ** 2) * (ac ** 2) - 2 * (bcr ** 2) * (ac ** 2)
        D = -2 * X1 * Y1 * (bcr ** 2)
        E = (X1 ** 2) * (bcr ** 2) - (bcr ** 4)

        # Call quartic function. Can test roots by uncommenting code below.
        [quartic1, quartic2] = quartic(A, B, C, D, E)

        # Calculate arctan of quartic roots 1 and 2
        # reduce arrays by 1 in third dimension and shift
        thetah1 = atand(quartic1(:,:,2:NRhmax))
        thetah2 = atand(quartic2(:,:,1:(NRhmax-1)))

        # Calculate soil view factor.
        fdhsoil = max(0, ((2/180)*(1/180) * (psiele) * (thetah1 - thetah2)))

        # Factor of 2 is for symmetrical interrows viewed by hemispherical transect.
        fdhc = 1 - 2*sum(fdhsoil, [2, 3])
        #the downward hemispherical view factor of the canopy is that which is sunlit+shaded
        self.fdhc =

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
        taudirVIS = (((rhodirv ** 2) - 1) * np.exp(-(self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs)) / (((rhodirv * SoilAlbvis) - 1) + rhodirv * (rhodirv - SoilAlbvis) * np.exp(-2 * (self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs))
        taudirNIR = (((rhodirn ** 2) - 1) * np.exp(-(self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs)) / (((rhodirn * SoilAlbnir) - 1) + rhodirn * (rhodirn - SoilAlbnir) * np.exp(-2 * (self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs))

        # Calculate direct beam canopy reflectance for visible and near infrared
        # spectra (CN98, eq. 15.9, p. 257)
        xidirv = ((rhodirv - SoilAlbvis) / (rhodirv * SoilAlbvis - 1)) * exp(-2 * (self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs)
        xidirn = ((rhodirn - SoilAlbnir) / (rhodirn * SoilAlbnir - 1)) * exp(-2 * (self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs)
        rhocsdirVIS = (rhodirv + xidirv) / (1 + xidirv * rhodirv)
        rhocsdirNIR = (rhodirn + xidirn) / (1 + xidirn * rhodirn)

        # ***********************************************************************

        # DIFFUSE TRANSMITTANCE AND REFLECTANCE

        # Calculate multiple row factor for a solar beam through the canopy
        # for each solar zenith and azimuth element (Colaizzi et al. 2012)
        thetasi = ones(size(hc)) * (5:5:85) # Solar zenith array (2D)
        psisi = (5:5:85) # Solar azimuth vector
        # Solar azimuth array (3D)
        psisi3 = ones(size(thetasi)) * reshape(psisi, 1, 1,[])
        # Projected solar zenith array (3D)
        tanthetaspi = ((tand(thetasi)) * sind(psisi3))
        # Multiple row factor for solar beam (3D)
        MRFsi = max(1, floor(tanthetaspi * hc / self.io.row_width))

        # Calculate path length fraction for a solar beam through the canopy
        # for each solar zenith and azimuth element (Colaizzi et al. 2012) (3D)
        Yspi = ac * bc / sqrt((ac ** 2) * (tanthetaspi ** 2) + (bc ** 2))
        Xspi = Yspi * tanthetaspi
        Zspi = Xspi / abs(tand(psisi3))
        PLFsi = (sqrt(Xspi ** 2 + Yspi ** 2 + Zspi ** 2)) / ac

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
                            taudirvi = (((rhodirvi ** 2) - 1) * ...
                                            exp(-(Leafabsv ** 0.5) * Kbsi * LAIL * PLFsi * MRFsi)) / ...
                                (((rhodirvi * SoilAlbvis) - 1) + rhodirvi * (rhodirvi - SoilAlbvis) * ...
                                     exp(-2 * (Leafabsv ** 0.5) * Kbsi * LAIL * PLFsi * MRFsi))
                                taudirni = (((rhodirni ** 2) - 1) * ...
                                                exp(-(Leafabsn ** 0.5) * Kbsi * LAIL * PLFsi * MRFsi)) / ...
                                    (((rhodirni * SoilAlbnir) - 1) + rhodirni * (rhodirni - SoilAlbnir) * ...
                                         exp(-2 * (Leafabsn ** 0.5) * Kbsi * LAIL * PLFsi * MRFsi))

                                    # Calculate diffuse canopy transmittance for visible and near infrared
                                    # spectra by integrating taudirvi and taudirni, repsectively
                                    taudirvi1 = 4 * (1 / 90) * (1 / 90) * ...
                                        ((sind(thetasi)) ** 2) * ((cosd(thetasi)) ** 2) * taudirvi
                                        taudirni1 = 4 * (1 / 90) * (1 / 90) * ...
                                            ((sind(thetasi)) ** 2) * ((cosd(thetasi)) ** 2) * taudirni
                                            taudiffVIS = sum(taudirvi1, [2, 3])
                                            taudiffNIR = sum(taudirni1, [2, 3])

                                            # Calculate direct beam canopy + soil reflectance for visible and near infrared
                                            # spectra for each solar zenith and azimuth element (CN98, eq. 15.9, p. 257)
                                            xidirvi = ((rhodirvi - SoilAlbvis) / (rhodirvi * SoilAlbvis - 1)) * ...
                                                exp(-2 * (Leafabsv ** 0.5) * Kbsi * LAIL * PLFsi * MRFsi)
                                                xidirni = ((rhodirni - SoilAlbnir) / (rhodirni * SoilAlbnir - 1)) * ...
                                                    exp(-2 * (Leafabsn ** 0.5) * Kbsi * LAIL * PLFsi * MRFsi)
                                                    rhocsdirvi = (rhodirvi + xidirvi) / (1 + xidirvi * rhodirvi)
                                                    rhocsdirni = (rhodirni + xidirni) / (1 + xidirni * rhodirni)

                                                    # Calculate diffuse canopy reflectance for visible and near infrared
                                                    # spectra by integrating rhocsdirvi and rhocsdirni, repsectively
                                                    rhocsdirvi1 = 4 * (5 / 85) * (5 / 85) * ...
                                                        ((sind(thetasi)) ** 2) * ((cosd(thetasi)) ** 2) * rhocsdirvi
                                                        rhocsdirni1 = 4 * (5 / 85) * (5 / 85) * ...
                                                            ((sind(thetasi)) ** 2) * ((cosd(thetasi)) ** 2) * rhocsdirni
                                                            rhocsdiffVIS = sum(rhocsdirvi1, [2, 3])
                                                            rhocsdiffNIR = sum(rhocsdirni1, [2, 3])
        
        #def taudir to compute transmittance of DIRECT beam radiation through the canopy - C&N 98
        self.taudir_vis =
        self.taudir_nir =
    def taudiff(self): 
        #def taudiff to compute transmittance of DIFFUSE radiation through the canopy
        #by integrating taudir over all solar zenith and azimuth angles
        self.taudiff_vis =
        self.taudiff_nir =
    def rhocsdir(self): 
        #def rhocsdiff to compute reflectance of DIRECT radiation through the canopy - C&N 98
        self.rhocsdir_vis =
        self.rhocsdir_nir =
    def rhocsdiff(self):
        #def rhocsdiff to compute reflectance of DIFFUSE radiation through the canopy
        #by integrating all solar zenith and azimuth angles
        self.rhocsdiff_vis =
        self.rhocsdiff_nir =

        function [quartic1, quartic2] = quartic(A, B, C, D, E)
        # Function quartic to calculate the first root (quartic1) and
        # second root (quartic2) of a quartic equation of form:
        # Ax^4 + Bx^3 + Cx^2 + Dx + E = 0.
        #
        # If user has Symbolic Toolbox, this function could be replaced with:
        # # syms A B C D E x
        # # eqn = A*x^4 + B*x^3 + C*x^2 + D*x + E == 0
        # # S = solve(eqn)
        # or similar code.
        #
        # Paul D. Colaizzi, USDA-ARS, Bushland, Texas, USA
        # ***********************************************************************

        # Check for quadratic equation
        quad = ((abs(B)<eps)&(abs(D)<eps))

        alpha = -((3 * B ** 2) / (8 * A ** 2)) + C / A
        beta = (B ** 3) / (8 * A ** 3) - (B * C) / (2 * A ** 2) + D / A
        gamma = -(3 * B ** 4) / (256 * A ** 4) + (C * B ** 2) / (16 * A ** 3) - ...
            (B * D) / (4 * A ** 2) + E / A
            P = -((alpha ** 2) / 12) - gamma
            Q = -(alpha ** 3) / (108) + (alpha * gamma / 3) - (beta ** 2) / 8
            r = Q / 2 + sqrt((Q ** 2) / 4 + (P ** 3) / 27)

            u = zeros(size(r))
            u = (r<0) * (-((abs(r)) ** (1/3))) + (r>=0)* (r ** (1/3))

            Uy = zeros(size(u))
            Uy = (u~=0) * P / (3 * (u+(u==0)*eps))

            y = -5 / 6 * alpha - u + Uy
            W = sqrt(abs(alpha + 2 * y))

            ZZo = zeros(size(y))
            ZZo = (W==0) * (abs(3 * alpha + 2 * y)) + ...
                (W~=0) * (-(3 * alpha + 2 * y + 2 * beta / (W+(W==0)*eps)))
                ZZ = (W==0) * (abs(3 * alpha + 2 * y)) + ...
                    ((ZZo<0)&(W~=0)) * abs(3 * alpha + 2 * y - 2 * beta / (W+(W==0)*eps)) ...
                        + (ZZo>=0) * ZZo

                        quartic1 = ((W==0)&(quad==0)) * (-(B / (4 * A)) + (W - sqrt(ZZ)) / 2 - y) + ...
                            ((W~=0)&(ZZ<0)&(quad==0)) * (-(B / (4 * A)) + (-W - sqrt(ZZ)) / 2) + ...
                                ((W~=0)&(ZZ>=0)&(quad==0)) * (-(B / (4 * A)) + (W - sqrt(ZZ)) / 2) + ...
                                    (quad) * -sqrt((-C + sqrt(C ** 2 - 4 * A * E)) / (2 * A))

                                    quartic2 = ((W==0)&(quad==0)) * (-(B / (4 * A)) + (-W + sqrt(ZZ)) / 2 - y) + ...
                                        ((W~=0)&(ZZ<0)&(quad==0)) * (-(B / (4 * A)) + (-W + sqrt(ZZ)) / 2) + ...
                                            ((W~=0)&(ZZ>=0)&(quad==0)) * (-(B / (4 * A)) + (W + sqrt(ZZ)) / 2) + ...
                                                (quad) * sqrt((-C + sqrt(C ** 2 - 4 * A * E)) / (2 * A))

                                                # y subtracted in first terms of quartic1 and quartic2 to smooth curves
                                                # If abs(Xc) < 0.001, then B = C =~ 0 and quad = 1, and assume quadratic equation.

                                                clearvars alpha beta gamma P Q r u Uy y W ZZo ZZ

                                                end
