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
        % Calculate psird, the radiometer azimuth relative to a crop row,
        % constrain 0 < psird < 90 deg, where 0 deg is parallel and 90 deg
        % is perdendicular to the crop row
        psird = abs(asind(sind(radazimuth - Rowdirc)));
        % psird cannot equal 0 or 90 deg to avoid domain errors.
        psirdc = psird + ((psird==0)|(psird==90)) .* eps;

        % Convert from degrees to radians
        psir = psirdc .* pi ./ 180;
        thetar = thetard .* pi ./ 180;

        % thetarp = Directional radiometer zenith angle projected perpendicular
        % to crop row; thetarp can be positive or negative (rad)
        thetarp = atan((tan(thetar)) .* sin(psir));

        n = 20; % Limit maximum rows to 20
        thetaspMR = thetasp .* ones(length(thetasp),n);
        thetarpMR = thetarp .* ones(length(thetasp),n);
        rowarray = ones(length(thetasp),n).*(1:1:n);

        % XcrMR = Horizontal distance from canopy ellipse origin to tangent of
        % sunray or directional radiometer view along thetaspcr or thetarpcr,
        % respectively, for multiple rows (n). XscrMR is positive (m)
        XcrMR = 2 .* ((bc) .^ 2) ./ (rowarray .* Rowsp);

        % YcrMR = Vertical distance from canopy ellipse origin to tangent of
        % sunray or directional radiometer view along thetaspcr or thetarpcr,
        % respectively, for multiple rows (n). YcrMR is positive (m)
        YcrMR = sqrt(((ac .^ 2).*XcrMR.*(rowarray .* Rowsp - 2.*XcrMR)) ./2./((bc).^2));

        % thetapcrMR = Critical perpendicular solar zenith angle, where greater
        % angles result in adjacent row shading for multiple rows,
        % or critical perpendicular directional radiometer view angle, where greater
        % angles result in ajacent rows obscuring view for multiple rows.
        % thetapcrMR is positive (rad)
        thetapcrMR = atan((rowarray .* Rowsp - 2.*XcrMR)./(2.*YcrMR));

        testarrays = rowarray.*(thetaspMR > thetapcrMR);
        testarrayr = rowarray.*(thetarpMR > thetapcrMR);

        % Calculate MRFs (solar) and MRFr (directional radiometer)
        % For solar, MultiRow >= 1 daylight only
        MRFs = (solarzenith<89) .* max(1,(max(testarrays,[],2)));
        MRFr = max(1, (max(testarrayr,[],2)));

        % Calculate PLFs (solar) and PLFr (directional radiometer)
        Ysp = ac .* bc ./ sqrt((ac .^ 2) .* ((tan(thetasp)) .^ 2) + (bc .^ 2));
        Xsp = Ysp .* tan(thetasp);
        Zsp = Xsp ./ abs(tan(azimuthsunrow));
        PLFs = (solarzenith<89) .* (sqrt(Xsp .^ 2 + Ysp .^ 2 + Zsp .^ 2)) ./ ac;

        Yrp = ac .* bc ./ sqrt((ac .^ 2) .* ((tan(thetarp)) .^ 2) + (bc .^ 2));
        Xrp = Yrp .* tan(thetarp);
        Zrp = Xrp ./ abs(tan(psir));
        PLFr = (sqrt(Xrp .^ 2 + Yrp .^ 2 + Zrp .^ 2)) ./ ac;
        
        self.mrf_s =
        self.mrf_r =

        self.plf_s =
        self.plf_r =

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
        #def rhocsdiff to compute reflectance of DIRECT radiation through the canopy - C&N 98
        self.rhocsdir_vis =
        self.rhocsdir_nir =

        # Calculate direct beam canopy transmittance for visible and near infrared
        # spectra (CN98, eq. 15.11, p. 257)
        taudirv = (((rhodirv ** 2) - 1) * np.exp(-(self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs)) / (((rhodirv * self.io.alb_vis) - 1) + rhodirv * (rhodirv - self.io.alb_vis) * np.exp(-2 * (self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs))
        taudivn = (((rhodirn ** 2) - 1) * np.exp(-(self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs)) / (((rhodirn * self.io.alb_nir) - 1) + rhodirn * (rhodirn - self.io.alb_nir) * np.exp(-2 * (self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs))

        # Calculate direct beam canopy reflectance for visible and near infrared
        # spectra (CN98, eq. 15.9, p. 257)
        xidirv = ((rhodirv - self.io.alb_vis)/ (rhodirv * self.io.alb_vis - 1)) * np.exp(-2 * (self.io.zeta_vis ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs)
        xidirn = ((rhodirn - self.io.alb_nir) / (rhodirn * self.io.alb_nir - 1)) * exp(-2 * (self.io.zeta_nir ** 0.5) * Kbs * self.io.LAIL * PLFs * MRFs)
        rhocsdirv = (rhodirv + xidirv) / (1 + xidirv * rhodirv)
        rhocsdirn = (rhodirn + xidirn) / (1 + xidirn * rhodirn)



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
        # ***********************************************************************

        # DIFFUSE TRANSMITTANCE AND REFLECTANCE

        # Calculate multiple row factor for a solar beam through the canopy
        # for each solar zenith and azimuth element (Colaizzi et al. 2012)
        thetasi = np.ones(len(hc)) * np.arange(5,85,5) # Solar zenith array (2D)
        psisi = np.arange(5,85,5) # Solar azimuth vector
        # Solar azimuth array (3D)
        psisi3 = np.ones(thetasi.shape) * reshape(psisi, 1, 1,[])
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

        self.taudiff_vis =
        self.taudiff_nir =


    def rhocsdir(self): 
    def rhocsdiff(self):
        #def rhocsdiff to compute reflectance of DIFFUSE radiation through the canopy
        #by integrating all solar zenith and azimuth angles
        self.rhocsdiff_vis =
        self.rhocsdiff_nir =

    def quartic(self,A, B, C, D, E)
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
