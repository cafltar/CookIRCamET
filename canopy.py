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

#def taudiff to compute transmittance of DIFFUSE radiation through the canopy
#by integrating taudir over all solar zenith and azimuth angles
                def taudiff(hc , wc , row , 
                                 LAI , XE , Zeta , rhosoil ) 

                #hc = Canopy height (m)
#wc = canopy width (m)
                #LAI = Leaf area index, field (m2 m-2)
#XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
                #Zeta = Leaf absorptivity (usually 0.85 for VIS, 0.15 for NIR)
#rhosoil = Soil reflectance (for Pullman clay loam, ~0.15 for VIS, ~0.25 for NIR)

                 Pi     #Pi
Pi = 3.14159265358979

 thetas     #Solar zenith angle (radians)
                 psis   #Solar azimuth angle
 PLFi   #Path length fraction of elliptical canopy
                 MRFi   #Multiple row function of elliptical canopy
 Kdir   #Extinction coefficient for direct beam radiation
                 TDIR     #Transmittance for direct beam radiation

#Initialize variables
                psis = Pi / 12      #Numerical integration carried out in 15 degree increments
taudiff = 0

Do While psis <= Pi / 2
    thetas = Pi / 12    #Numerical integration carried out in 15 degree increments
                    Do While thetas < Pi / 2
                            TDIR = taudir(thetas, psis, hc, wc, row, LAI, XE, Zeta, rhosoil)
                                    taudiff = taudiff + 2 * (2 / Pi) * TDIR * Sin(thetas) * Cos(thetas) * (Pi / 12) * (Pi / 12)
                                            thetas = thetas + Pi / 12
                                                Loop
                                                    psis = psis + Pi / 12
                                                    Loop

                                                    
                                                    #def rhocsdir to compute directional-hemispherical reflectance of DIRECT beam radiation from the canopy
#using procedure of Campbell and Norman (1998), Chapter 15 (CN98)
                                                    def rhocsdir(thetas , psis , hc , wc , row , 
                                                                      LAI , XE , Zeta , rhosoil ) 

                                                    #Thetas = Solar zenith angle (rad)
#Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
                                                    #   and 90 degrees for perpendicular orientation (rad)
#hc = Canopy height (m)
                                                    #wc = canopy width (m)
#LAI = Leaf area index, field (m2 m-2)
                                                    #XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
#Zeta = Leaf absorptivity (usually 0.85 for VIS, 0.15 for NIR)
                                                    #rhosoil = Soil reflectance (for Pullman clay loam, ~0.15 for VIS, ~0.25 for NIR)

 PLFi   #Path length fraction of continuous ellipse canopy
                                                     MRFi   #Multiple row function of continuous ellipse canopy
 Kdir   #Extinction coefficient for direct beam radiation
                                                     rhohor      #Reflectivity for horizontal leafs (VIS or NIR)
 rhodir      #Adjusted canopy reflectivity as a function of Kdir and rhohor (VIS or NIR)
                                                     Xidir       #Second-order reflectance from soil contributing to canopy reflectance
                            #for direct (beam) radiation (VIS or NIR)

                                                    PLFi = PLF(thetas, psis, hc, wc, row)
                                                    MRFi = MRF(thetas, psis, hc, wc, row)
                                                    Kdir = ((XE ^ 2 + (Tan(thetas)) ^ 2) ^ 0.5) / (XE + 1.774 * (XE + 1.182) ^ -0.733) #CN98, 15.4, p. 251
rhohor = (1 - ((Zeta) ^ 0.5)) / (1 + ((Zeta) ^ 0.5))    #CN98, 15.7, p. 255
                                                    rhodir = 2 * Kdir * rhohor / (Kdir + 1)     #CN98, 15.8, p. 257

#Convert field LAI to local LAI
                                                     LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
LAIL = LAI * row / wc

Xidir = ((rhodir - rhosoil) / (rhodir * rhosoil - 1)) * 
Exp(-2 * (Zeta ^ 0.5) * Kdir * LAIL * PLFi * MRFi)

rhocsdir = (rhodir + Xidir) / (1 + Xidir * rhodir)  #CN98, 15.9, p. 257

                                                    
                                                    #def rhocsdiff to compute hemispherical-hemispherical reflectance of DIFFUSE radiation from the canopy
#by integrating rhocsdir over all solar zenith and azimuth angles
                                                    def rhocsdiff(hc , wc , row , 
                                                                       LAI , XE , Zeta , rhosoil ) 

                                                    #hc = Canopy height (m)
#wc = canopy width (m)
                                                    #LAI = Leaf area index, field (m2 m-2)
#XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
                                                    #Zeta = Leaf absorptivity (usually 0.85 for VIS, 0.15 for NIR)
#rhosoil = Soil reflectance (for Pullman clay loam, ~0.15 for VIS, ~0.25 for NIR)

                                                     Pi     #Pi
Pi = 3.14159265358979

 thetas     #Solar zenith angle (radians)
                                                     psis   #Solar azimuth angle
 PLFi   #Path length fraction of elliptical canopy
                                                     MRFi   #Multiple row function of elliptical canopy
 Kdir   #Extinction coefficient for direct beam radiation
                                                     RDIR     #Reflectance for direct beam radiation

#Initialize variables
                                                    psis = Pi / 12      #Numerical integration carried out in 15 degree increments
rhocsdiff = 0

Do While psis <= Pi / 2
    thetas = Pi / 12    #Numerical integration carried out in 15 degree increments
                                                        Do While thetas < Pi / 2
                                                                RDIR = rhocsdir(thetas, psis, hc, wc, row, LAI, XE, Zeta, rhosoil)
                                                                        rhocsdiff = rhocsdiff + 2 * (2 / Pi) * RDIR * Sin(thetas) * Cos(thetas) * (Pi / 12) * (Pi / 12)
                                                                                thetas = thetas + Pi / 12
                                                                                    Loop
                                                                                        psis = psis + Pi / 12
                                                                                        Loop

Attribute VB_Name = "Module3"
Option Explicit
Function MRF(thetas As Double, psis As Double, hc As Double, wc As Double, row As Double) As Double

#Function MRF (multiple row factor) to calculate the number of crop rows
#along the path of a sunbeam, where crop rows are modeled as ellipses.

#Thetas = Solar zenith angle (rad)
#Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
#   and 90 degrees for perpendicular orientation (rad)
#hc = Canopy height (m)
#wc = canopy width (m)

# bc As Double    #Vertical axis of crop ellipse (m)
# ac As Double   #Horizonal axis of crop ellipse (m)
# thetasp As Double   #Solar zenith angle projected perpendicular to crop row (rad)

# Xscr As Double  #Horizontal distance from canopy ellipse origin to tangent of sunray
#along thetaspcr (m)
# Yscr As Double  #Vertical distance from canopy ellipse origin to tangent of sunray
#along thetaspcr (m)
# thetaspcr As Double #Critical perpendicular solar zenith angle, where greater angles
#result in adjacent row shading (rad)

# Xscr2 As Double  #Horizontal distance from canopy ellipse origin to tangent of sunray
#along thetaspcr2, for the next row from Xscr (m)
# Yscr2 As Double  #Vertical distance from canopy ellipse origin to tangent of sunray
#along thetaspcr2, for the next row from Yscr (m)
# thetaspcr2 As Double #Critical perpendicular solar zenith angle, where greater angles
#result in adjacent row shading, for the next row from thetascr (rad)

#Assign value to pi
# Pi As Double
Pi = 3.14159265358979

If thetas > 89 * Pi / 180 Then
    MRF = 0
    GoTo 20
End If

#Constrain wc to row spacing
If wc >= row Then wc = (row - 0.01)

bc = hc / 2
ac = wc / 2
thetasp = Atn((Tan(thetas)) * Abs(Sin(psis)))

# n As Integer    #Loop counter
n = 0
thetaspcr2 = 0

Do While thetasp >= thetaspcr2

    n = n + 1
    Xscr = 2 * ((ac) ^ 2) / (n * row)   #Xscr is positive
    Yscr = Sqr(((bc ^ 2) * Xscr * (n * row - 2 * Xscr)) / 2 / ((ac) ^ 2)) #Yscr is positive
    thetaspcr = Atn((n * row - 2 * Xscr) / (2 * Yscr)) #thetascr is positive

        Xscr2 = 2 * ((ac) ^ 2) / ((n + 1) * row) #Xscr is positive
    Yscr2 = Sqr(((bc ^ 2) * Xscr2 * ((n + 1) * row - 2 * Xscr2)) / 2 / ((ac) ^ 2)) #Yscr is positive
            thetaspcr2 = Atn(((n + 1) * row - 2 * Xscr2) / (2 * Yscr2)) #thetascr is positive

Loop

MRF = n + ((thetasp - thetaspcr) / (thetaspcr2 - thetaspcr))

If MRF < 1 Then MRF = 1

20 End Function

Function PLF(thetas As Double, psis As Double, hc As Double, wc As Double, row As Double) As Double

#Function PLF (Path Length Fraction) to calculate the path length of a sunray
            #relative to NADIR of an elliptical canopy

#Thetas = Solar zenith angle (rad)
            #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
#   and 90 degrees for perpendicular orientation (rad)
            #hc = Canopy height (m)
#wc = canopy width (m)

            # bc As Double    #Vertical axis of crop ellipse (m)
# ac As Double   #Horizonal axis of crop ellipse (m)
            # thetasp As Double   #Solar zenith angle projected perpendicular to crop row (rad)

# Xp As Double  #Horizontal distance from canopy ellipse origin to point of entry of
            #sunray that passes through canopy ellipse origin (m)
# Yp As Double  #Vertical distance from canopy ellipse origin to point of entry of
            #sunray that passes through canopy ellipse origin (m)
# Zp As Double  #Axial distance from canopy ellipse origin to point of entry of
            #sunray that passes through canopy ellipse origin (m)

#Assign value to pi
            # Pi As Double
            Pi = 3.14159265358979

            If thetas > 89 * Pi / 180 Then
                PLF = 0
                    GoTo 20
                    End If

                    #Constrain wc to row spacing
If wc >= row Then wc = (row - 0.01)

#Constrain psis to > 0
                    If psis <= 0 Then psis = 0.01

                    bc = hc / 2
                    ac = wc / 2
                    thetasp = Atn((Tan(thetas)) * Abs(Sin(psis)))
                    Yp = ac * bc / Sqr((bc ^ 2) * ((Tan(thetasp)) ^ 2) + (ac ^ 2))
                    Xp = Yp * Tan(thetasp)
                    Zp = Xp / Abs(Tan(psis))
                    PLF = (Sqr(Xp ^ 2 + Yp ^ 2 + Zp ^ 2)) / bc

                    20 End Function

                    Function fcsolar(thetas As Double, psis As Double, hc As Double, wc As Double, row As Double) As Double
                    #Function fcsolar to estimate the fraction of the total surface covered by the canopy normal
#to the solar zenith angle, where the canopy is modelled as an ellipse.
                    #Both leaves and substrate may be visible within the canopy area.

#Thetas = Solar zenith angle (rad)
                    #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
#   and 90 degrees for perpendicular orientation (rad)
                    #hc = Canopy height (m)
#wc = canopy width (m)

                    #Assign value to pi
# Pi As Double
Pi = 3.14159265358979

If thetas > 89 * Pi / 180 Then
    fcsolar = 0
    GoTo 20
End If

#Constrain wc to row spacing
                    If wc >= row Then wc = (row - 0.01)

                    # bc As Double    #Vertical axis of crop ellipse (m)
# ac As Double   #Horizonal axis of crop ellipse (m)
                    # thetasp As Double   #Solar zenith angle projected perpendicular to crop row (rad)

# Xscr As Double  #Horizontal distance from canopy ellipse origin to tangent of sunray
                    #along thetaspcr (m)
# Yscr As Double  #Vertical distance from canopy ellipse origin to tangent of sunray
                    #along thetaspcr (m)
# thetaspcr As Double #Critical perpendicular solar zenith angle, where greater angles
                    #result in adjacent row shading (rad)

bc = hc / 2
ac = wc / 2
thetasp = Atn((Tan(thetas)) * Abs(Sin(psis)))
    
Xscr = 2 * ((ac) ^ 2) / (row)   #Xscr is positive
                    Yscr = Sqr(((bc ^ 2) * Xscr * (row - 2 * Xscr)) / 2 / ((ac) ^ 2)) #Yscr is positive
thetaspcr = Atn((row - 2 * Xscr) / (2 * Yscr)) #thetascr is positive

                    If thetasp >= thetaspcr Then
                        fcsolar = 1
                            GoTo 20
                            End If

                            # Xs As Double    #Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
# Ys As Double    #Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)

                            Xs = ac / Sqr(1 + (bc ^ 2) / ((ac) ^ 2) * ((Tan(thetasp)) ^ 2))    #Xs is positive
Ys = (bc ^ 2) / ((ac) ^ 2) * Xs * Tan(thetasp) #Ys is positive

                            fcsolar = (2 * Xs + 2 * Ys * Tan(thetasp)) / row

                            20 End Function

                            Function fdhc(hc As Double, wc As Double, row As Double, Pr As Double, Vr As Double) As Double
                            #Function fdhc to calculate the downward hemispherical view factor of canopy
#of a row crop (e.g., canopy viewed by an inverted radiometer)

                            #hc = Crop canopy height (m)
#wc = Crop canopy width (m)
                            #row = Crop row spacing (m)
#Vr = Vertical distance of radiometer from soil surface (m)
                            #Pr = Horizontal, perpendicular distance from radiometer to row center (m)

# ac As Double    #Crop canopy major semiaxis
                            # bc As Double    #Crop canopy minor semiaxis
# Pi As Double
# psir As Double #Azimuth view angle element of radiometer (radians)
                            # fdhsoil As Double #Downward hemispherical view factor of soil
# thetar1 As Double #Zenith view angle of radiometer to ellipse tangent, left of ellipse (radians)
                            # thetar2 As Double #Zenith view angle of radiometer to ellipse tangent, right of ellipse (radians)
# i As Integer #Loop counter for multiple rows
                            # NR As Integer    #Minimum number of interrows where soil is visible to radiometer
# tanthetarcr As Double  #Tangent of maximum zenith view angle of radiomoeter-soil view factor (radians)

                            ac = hc / 2
                            bc = wc / 2
                            Pi = 3.14159265358979
                            psir = Pi / 72
                            fdhsoil = 0
                            If wc >= row Then GoTo 20

                            Do While psir < Pi / 2
                                tanthetarcr = row * ((1 - 4 * (bc ^ 2) / (row ^ 2)) ^ 0.5) / (2 * ac * Sin(psir))
                                    NR = Application.WorksheetFunction.RoundUp(((Vr * tanthetarcr * Sin(psir)) / row), 0) + 2#
                                        i = -NR
                                            For i = -NR To NR Step 1
                                                    thetar1 = Atn(quartic1(ac, bc, psir, (row * (i + 1) - Pr), Vr))
                                                            thetar2 = Atn(quartic2(ac, bc, psir, (row * i - Pr), Vr))
                                                                    If thetar1 > thetar2 Then
                                                                                fdhsoil = fdhsoil + (2 / Pi) * (1 / Pi) * (thetar1 - thetar2) * (Pi / 72)
                                                                                        Else
                                                                                                    GoTo 10
                                                                                                            End If
                                                                                                            10  Next i
                                                                                                                psir = psir + Pi / 72
                                                                                                                Loop

                                                                                                                20 fdhc = 1 - fdhsoil

                                                                                                                End Function


                                                                                                                
Attribute VB_Name = "Module4"
Option Explicit
Function fveg(XE As Double, thetar As Double, rawpsi As Double, hc As Double, _
              wc As Double, row As Double, LAI As Double, Pr As Double, Vr As Double, _
              FOV As Double, thetas As Double, psis As Double) As Double
'Function fveg to compute the fraction of vegetation appearing in a radiometer footprint
'where the crop rows are modeled as continuous ellipses.

'LADFOPT = Option for leaf angle distribution function (LADF), where
'           (1 = Ellipsoidal, 2 = Beta)
'For ellipsoidal LADF option:
    'Xe = Ratio of horizontal to vertical projected leaves (spherical Xe = 1)
'For Beta LADF option:
    'tAVG = mean of normalized leaf angle t (e.g., 0.5 for symmetric PDF's)
    'tVAR = variance of normalized leaf angle t

                                             'thetar = Radiometer view zenith angle (radians)
'rawpsi = Azimuth angel of crop row relative to radiometer view angle,
                                             '   where zero is looking parallel to crop row and pi/2 is looking perpendicular
'   to crop row (radians)
                                             'hc = Canopy height (m)
'wc = Canopy width (m)
                                             'row = Crop row spacing (m)
'LAI = Leaf area index (m2 m-2)

                                             'Pr = Perpendicular distance of radiometer from canopy row center (m)
'Vr = Vertical height of radiometer relative to soil (m)
                                             'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
'thetas = Solar zenith angle (rad)
                                             'psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
'       and 90 degrees for perpendicular orientation (rad)

                                             'Variables used in this function
Dim KBR As Double   'Extinction coefficient of radiometer viewing canopy
                                             Dim PFR As Double    'Path length fraction of radiometer viewing continuous ellipse
Dim MFR As Double    'Multiple row function of radiometer viewing continuous ellipses
                                             Dim psir As Double  'radiometer azimuth angle relative to crop row, constrained to 45-90 deg
Dim ER As Double    'Extinction of radiometer view path through canopy
                                             Dim fcs1 As Double  'Fraction of solid sunlit continuous ellispe appearing in radiometer footprint
Dim fcs2 As Double  'Fraction of solid shaded continuous ellispe appearing in radiometer footprint

                                             'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

KBR = (Sqr(XE ^ 2 + (Tan(thetar)) ^ 2)) / _
                (XE + 1.774 * (XE + 1.182) ^ -0.733)

'Constrain 45 < rawpsi < 90
                                             If Abs(rawpsi) < 45 * Pi / 180 Then
                                                 psir = Pi / 2 - (Abs(rawpsi))
                                             Else
                                                 If Abs(rawpsi) > 135 * Pi / 180 Then
                                                     psir = (Abs(rawpsi)) - Pi / 2
                                                 Else
                                                     If Abs(rawpsi) > 90 * Pi / 180 Then
                                                         psir = Pi - Abs(rawpsi)
                                                     Else
                                                         psir = Abs(rawpsi)
                                                     End If
                                                 End If
                                             End If

                                             PFR = PLF(thetar, psir, hc, wc, row)
                                             MFR = MRF(thetar, psir, hc, wc, row)

                                             ER = Exp(-KBR * row / wc * LAI * PFR * MFR)

                                             fcs1 = fcs(1, Pr, row, thetar, rawpsi, hc, wc, Vr, FOV, thetas, psis)
                                             fcs2 = fcs(2, Pr, row, thetar, rawpsi, hc, wc, Vr, FOV, thetas, psis)

                                             fveg = (fcs1 + fcs2) * (1 - ER)

                                             End Function
                                             Function fcs(OPT As Integer, Pr As Double, row As Double, theta As Double, rawpsi As Double, _
                                                          hc As Double, wc As Double, Dv As Double, FOV As Double, thetas As Double, psis As Double) As Double

                                             'Function fcs to compute the fraction of sunlit or shaded canopy or soil
'appearing in the elliptical footprint of a radiometer, where the canopy
                                             'is modelled as ELLIPSE

'OPT = 1 for sunlit canopy
                                             '      2 for shaded canopy
'      3 for sunlit soil
                                             '      4 for shaded soil

'Pr = Perpendicular distance of radiometer from canopy row center (m)
                                             'row = Crop row spacing (m)
'Theta = Zenith angle of radiometer (radians)
                                             'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
'   where zero is looking parallel to crop row and pi/2 is looking perpendicular
                                             '   to crop row (radians)
'hc = Canopy height (m)
                                             'wc = canopy width (m)
'Dv = Vertical height of radiometer relative to soil (m)
                                             'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
'thetas = Solar zenith angle (radians)
                                             'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
'crop row and pi/2 is looking perpendicular to crop row (radians)

                                             'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

'Compute major (ar) and minor (br) axes of radiometer footprint

                                             Dim ar As Double
                                             Dim OA As Double
                                             Dim br As Double

                                             ar = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) - Tan(theta - Atn(1 / 2 / FOV)))
                                             OA = Dv * Tan(theta - Atn(1 / 2 / FOV))
                                             br = (Sqr(((ar + OA) ^ 2) + (Dv ^ 2))) / (2 * FOV)

                                             Dim Aell As Double    'Total elliptical area of radiometer footprint (m2)
Aell = ar * br * Pi

'Constrain 45 < psi < 90
                                             Dim psi As Double
                                             If Abs(rawpsi) < 45 * Pi / 180 Then
                                                 psi = Pi / 2 - (Abs(rawpsi))
                                             Else
                                                 If Abs(rawpsi) > 135 * Pi / 180 Then
                                                     psi = (Abs(rawpsi)) - Pi / 2
                                                 Else
                                                     If Abs(rawpsi) > 90 * Pi / 180 Then
                                                         psi = Pi - Abs(rawpsi)
                                                     Else
                                                         psi = Abs(rawpsi)
                                                     End If
                                                 End If
                                             End If

                                             'Compute T1
Dim T1 As Double    'Distance from radiometer ellipse footprint center to line
                                             '   tangent to radiometer ellipse for given azimuth and zenith angles (m).
If (Abs(psi - 90 * Pi / 180) < 0.01) Then   '90 degree radiometer azimuth
                                                 T1 = ar
                                                 GoTo 10
                                             Else
                                                 If Abs(psi) < 0.01 Then     'ZERO degree radiometer azimuth
        T1 = br
        GoTo 10
    Else                        'Radiometer azimuth between ZERO and 90 degrees
                                                     Dim xtr1 As Double
                                                     Dim ytr1 As Double
                                                     xtr1 = ar / Sqr(1 + (br ^ 2) / (ar ^ 2) / ((Tan(psi)) ^ 2))
                                                     ytr1 = (br ^ 2) / (ar ^ 2) * xtr1 / Tan(psi)
                                                     T1 = xtr1 + ytr1 / Tan(psi)
                                                 End If
                                             End If

                                             'Determine number of rows appearing in radiometer footprint, make it an odd number,
'and add TWO extra rows either side to account for adjacent row shading
                                             10 Dim NR As Integer
                                             NR = 2 * (Application.WorksheetFunction.RoundUp((2 * T1 * Sin(psi) / row), 0)) + 1# + 4#

                                             Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
                                             thetasp = -thetas * Sin(psis)

                                             Dim MIN As Double
                                             If Abs(Tan(rawpsi)) < 1 Then
                                             MIN = Tan(Pi / 2 - psi)
                                             Else
                                             MIN = 1
                                             End If

                                             OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

                                             'Declare arrays
ReDim Pnr(NR) As Double
ReDim H(NR, 6) As Double
ReDim tantheta(NR, 6) As Double
ReDim ac(NR, 6) As Double
ReDim fc(NR, 4) As Double
Dim fcr(4) As Double

Dim N1 As Integer
For N1 = 1 To NR 'Row number

                                                 Pnr(N1) = Pr + (row / 2) * (2 * N1 - NR - 1)

                                                 H(N1, 1) = Heighte1(Pnr(N1), theta, rawpsi, hc, wc, Dv, FOV)
                                                 H(N1, 2) = Heighte2(Pnr(N1), theta, rawpsi, hc, wc, Dv, FOV)
                                                 H(N1, 3) = Heighte3(Pnr(N1), row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
                                                 H(N1, 4) = Heighte4(Pnr(N1), row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
                                                 H(N1, 5) = Heighte5(Pnr(N1), row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
                                                 H(N1, 6) = Heighte6(Pnr(N1), row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)

                                             '    tantheta(N1, 1) = (H(N1, 1) + OC * MIN) / Dv
'    tantheta(N1, 2) = (H(N1, 2) + OC * MIN) / Dv
                                             '    tantheta(N1, 3) = (H(N1, 3) + OC * MIN) / Dv
'    tantheta(N1, 4) = (H(N1, 4) + OC * MIN) / Dv
                                             '    tantheta(N1, 5) = (H(N1, 5) + OC * MIN) / Dv
'    tantheta(N1, 6) = (H(N1, 6) + OC * MIN) / Dv

                                                 If H(N1, 4) > H(N1, 1) Then
                                                     H(N1, 4) = H(N1, 1) 'H1 obscurs H4 (no shaded soil visible on near-side)
        H(N1, 3) = H(N1, 1) 'H1 obscurs H3 (no shaded canopy visible on near-side)
                                                 End If

                                                 If H(N1, 6) < H(N1, 2) Then
                                                     H(N1, 6) = H(N1, 2) 'H2 obscurs H6 (no shaded soil visible on far-side)
        H(N1, 5) = H(N1, 2) 'H2 obscurs H5 (no shaded canopy visible on far-side)
                                                 End If


                                             '    If H(N1, 4) > H(N1, 1) Then H(N1, 4) = H(N1, 1) 'H1 obscurs H4 (no shaded soil visible on near-side)
                                             '    If H(N1, 3) < H(N1, 1) Then H(N1, 3) = H(N1, 1) 'H1 obscurs H3 (no shaded canopy visible on near-side)
                                             '    If H(N1, 6) < H(N1, 2) Then H(N1, 6) = H(N1, 2) 'H2 obscurs H6 (no shaded soil visible on far-side)
                                             '    If H(N1, 5) > H(N1, 2) Then H(N1, 5) = H(N1, 2) 'H2 obscurs H5 (no shaded canopy visible on far-side)

                                             Next

                                             Dim N2 As Integer
                                             For N2 = 2 To NR   'Account for adjacent rows obscuring chord locations

    If H((N2 - 1), 2) > H(N2, 4) Then H(N2, 4) = H((N2 - 1), 2)
    'Far side of canopy boundary in row N2-1 obscurs near side of sunlit-shaded soil boundary in row N2
                                                 If H((N2 - 1), 2) > H(N2, 1) Then H(N2, 1) = H((N2 - 1), 2)
                                                 'Far side of canopy boundary in row N2-1 obscurs near side of canopy in row N2
    If H((N2 - 1), 2) > H(N2, 3) Then H(N2, 3) = H((N2 - 1), 2)
    'Far side of canopy boundary in row N2-1 obscurs near side of sunlit-shaded canopy boundary in row N2

                                                 If H(N2, 1) < H((N2 - 1), 6) Then H((N2 - 1), 6) = H(N2, 1)
                                                 'Near side of canopy in row N2 obscurs far side of sunlit-shaded soil boundary in row N2-1
    If H(N2, 1) < H((N2 - 1), 2) Then H((N2 - 1), 2) = H(N2, 1)
    'Near side of canopy in row N2 obscurs far side of canopy in row N2-1
                                                 If H(N2, 1) < H((N2 - 1), 5) Then H((N2 - 1), 5) = H(N2, 1)
                                                 'Near side of canopy in row N2 obscurs far side of sunlit-shaded canopy boundary in row N2-1
    
Next

'Build array of chord areas

                                             Dim N3 As Integer
                                             For N3 = 1 To NR
                                                 Dim y As Integer
                                                 For y = 1 To 6  'N3 Chord numbers 1 to 6
        If H(N3, y) > 0 Then
            ac(N3, y) = Chord(psi, ar, br, Abs(H(N3, y)))
        Else
            ac(N3, y) = Aell - Chord(psi, ar, br, Abs(H(N3, y)))
        End If
    Next
Next

'Compute areas of sunlit and shaded soil and canopy visible to radiometer

                                             Dim N4 As Integer
                                             For N4 = 2 To (NR - 1)

                                                 fc(N4, 1) = (ac(N4, 3) - ac(N4, 2)) 'Sunlit canopy
    'If fc(N4, 1) < 0 Then fc(N4, 1) = 0

                                                 fc(N4, 2) = (ac(N4, 1) - ac(N4, 3)) + ac(N4, 5) - ac(N4, 2) 'Shaded canopy
    'If fc(N4, 2) < 0 Then fc(N4, 2) = 0

                                                 fc(N4, 3) = (ac((N4 - 1), 6) - ac(N4, 4) + ac(N4, 6) - ac((N4 + 1), 4)) * 0.5 'Sunlit soil
    'If fc(N4, 3) < 0 Then fc(N4, 3) = 0

                                                 fc(N4, 4) = ac(N4, 4) - ac(N4, 1) + ac(N4, 2) - ac(N4, 6) 'Shaded soil
    'If fc(N4, 4) < 0 Then fc(N4, 4) = 0

                                             Next

                                             'Initialize fcr values
fcr(1) = 0
fcr(2) = 0
fcr(3) = 0
fcr(4) = 0

'Sum areas for each row
                                             Dim N5 As Integer
                                             For N5 = 2 To (NR - 1)

                                                 fcr(1) = fc(N5, 1) + fcr(1)
                                                 fcr(2) = fc(N5, 2) + fcr(2)
                                                 fcr(3) = fc(N5, 3) + fcr(3)
                                                 fcr(4) = fc(N5, 4) + fcr(4)

                                             Next
                                             '20 fcs = H(3, OPT)

fcs = (fcr(OPT)) / Aell

End Function

Function Heighte1(Pr As Double, theta As Double, rawpsi As Double, _
hc As Double, wc As Double, Dv As Double, FOV As Double) As Double

'Heighte1 = Distance from center of footprint of radiometer to chord shadow
                                             '   cast by near-edge of row crop canopy, where crop canopy is modelled as ELLIPSE (m)
'Pr = Perpendicular distance of radiometer from canopy row center (m)
                                             'Theta = Zenith angle of radiometer (radians)
'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
                                             '   where zero is looking parallel to crop row and pi/2 is looking perpendicular
'   to crop row (radians)
                                             'hc = Canopy height (m)
'wc = canopy width (m)
                                             'Dv = Vertical height of radiometer relative to soil (m)
'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")

                                             'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

'Constrain 45 < psi < 90
                                             Dim psi As Double
                                             If Abs(rawpsi) < 45 * Pi / 180 Then
                                                 psi = Pi / 2 - (Abs(rawpsi))
                                             Else
                                                 If Abs(rawpsi) > 135 * Pi / 180 Then
                                                     psi = (Abs(rawpsi)) - Pi / 2
                                                 Else
                                                     If Abs(rawpsi) > 90 * Pi / 180 Then
                                                         psi = Pi - Abs(rawpsi)
                                                     Else
                                                         psi = Abs(rawpsi)
                                                     End If
                                                 End If
                                             End If

                                             Dim ac As Double    'Horizonal axis of elliptical canopy (m)
Dim bc As Double    'Vertical axis of elliptical canopy (m)
                                             Dim X1 As Double    'Horizonal distance from canopy ellipse origin to radiometer (m)
Dim Y1 As Double    'Vertical distance from canopy ellipse origin to radiometer (m)
                                             ac = wc / 2 / Sin(psi)
                                             bc = hc / 2
                                             X1 = Pr / Sin(psi)
                                             Y1 = Dv - bc

                                             'Find tantheta1, the inverse slope of tangent line from radiometer to near edge of canopy
Dim tantheta1 As Double
tantheta1 = quartic1IRT(X1, Y1, ac, bc)

Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
                                             OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

                                             Dim MIN As Double
                                             If Abs(Tan(rawpsi)) < 1 Then
                                             MIN = Tan(Pi / 2 - psi)
                                             Else
                                             MIN = 1
                                             End If

                                             Heighte1 = Dv * tantheta1 - OC * MIN

                                             End Function
                                             Function Heighte2(Pr As Double, theta As Double, rawpsi As Double, _
                                                               hc As Double, wc As Double, Dv As Double, FOV As Double) As Double

                                             'Heighte2 = Distance from center of footprint of radiometer to chord shadow
'   cast by far-edge of row crop canopy, where crop canopy is modelled as ELLIPSE (m)
                                             'Pr = Perpendicular distance of radiometer from canopy row center (m)
'Theta = Zenith angle of radiometer (radians)
                                             'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
'   where zero is looking parallel to crop row and pi/2 is looking perpendicular
                                             '   to crop row (radians)
'hc = Canopy height (m)
                                             'wc = canopy width (m)
'Dv = Vertical height of radiometer relative to soil (m)
                                             'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")

'Assign value to pi
                                             Dim Pi As Double
                                             Pi = 3.14159265358979

                                             'Constrain 45 < psi < 90
Dim psi As Double
If Abs(rawpsi) < 45 * Pi / 180 Then
    psi = Pi / 2 - (Abs(rawpsi))
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psi = (Abs(rawpsi)) - Pi / 2
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psi = Pi - Abs(rawpsi)
        Else
            psi = Abs(rawpsi)
        End If
    End If
End If

Dim ac As Double    'Horizonal axis of elliptical canopy (m)
                                             Dim bc As Double    'Vertical axis of elliptical canopy (m)
Dim X1 As Double    'Horizonal distance from canopy ellipse origin to radiometer (m)
                                             Dim Y1 As Double    'Vertical distance from canopy ellipse origin to radiometer (m)
ac = wc / 2 / Sin(psi)
bc = hc / 2
X1 = Pr / Sin(psi)
Y1 = Dv - bc

'Find tantheta2, the inverse slope of tangent line from radiometer to far edge of canopy
                                             Dim tantheta2 As Double
                                             tantheta2 = quartic2IRT(X1, Y1, ac, bc)

                                             Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

Dim MIN As Double
If Abs(Tan(rawpsi)) < 1 Then
MIN = Tan(Pi / 2 - psi)
Else
MIN = 1
End If

Heighte2 = Dv * tantheta2 - OC * MIN

End Function
Function Heighte3(Pr As Double, row As Double, theta As Double, rawpsi As Double, _
hc As Double, wc As Double, Dv As Double, FOV As Double, _
thetas As Double, psis As Double) As Double

'Heighte3 = Distance from center of footprint of radiometer to chord projected by
                                             'sunlit-shaded boundary on near-side of canopy, where crop canopy is modelled as ELLIPSE (m)
'Pr = Perpendicular distance of radiometer from canopy row center (m)
                                             'row = Crop row spacing (m)
'Theta = Zenith angle of radiometer (radians)
                                             'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
'   where zero is looking parallel to crop row and pi/2 is looking perpendicular
                                             '   to crop row (radians)
'hc = Canopy height (m)
                                             'wc = canopy width (m)
'Dv = Vertical height of radiometer relative to soil (m)
                                             'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
'thetas = Solar zenith angle (radians)
                                             'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
'crop row and pi/2 is looking perpendicular to crop row (radians)

                                             'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

'Constrain 45 < psi < 90
                                             Dim psi As Double
                                             If Abs(rawpsi) < 45 * Pi / 180 Then
                                                 psi = Pi / 2 - (Abs(rawpsi))
                                             Else
                                                 If Abs(rawpsi) > 135 * Pi / 180 Then
                                                     psi = (Abs(rawpsi)) - Pi / 2
                                                 Else
                                                     If Abs(rawpsi) > 90 * Pi / 180 Then
                                                         psi = Pi - Abs(rawpsi)
                                                     Else
                                                         psi = Abs(rawpsi)
                                                     End If
                                                 End If
                                             End If

                                             Dim bc As Double    'Vertical axis of elliptical canopy (m)
Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
                                             Dim Xs As Double    'Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
Dim Ys As Double    'Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
                                             Dim X3 As Double    'Horizontal distance from radiometer to ground-projected
'sunlit-shaded boundary on canopy (m)
                                             bc = hc / 2
                                             thetasp = -thetas * Sin(psis)
                                             Xs = wc / 2 / Sqr(1 + (bc ^ 2) / ((wc / 2) ^ 2) * ((Tan(thetasp)) ^ 2))    'Xs is positive
Ys = -(bc ^ 2) / ((wc / 2) ^ 2) * Xs * Tan(thetasp) 'Ys is positive or negative

                                             'Determine critical perpendicular solar zenith angle,
'beyond which results in adjacent row shading
                                             Dim Xscr As Double  'Horizontal distance from canopy ellipse origin to tangent of sunray
'along thetasp (m)
                                             Dim Yscr As Double  'Vertical distance from canopy ellipse origin to tangent of sunray
'along thetasp (m)
                                             Dim thetaspcr As Double     'Critical perpendicular solar zenith angle

Xscr = 2 * ((wc / 2) ^ 2) / row     'Xscr is positive
                                             Yscr = -Sqr(((bc ^ 2) * Xscr * (row - 2 * Xscr)) / 2 / ((wc / 2) ^ 2))  'Yscr is negative
thetaspcr = Atn(-((wc / 2) ^ 2) * Yscr / (bc ^ 2) / Xscr)   'thetascr is positive

                                             If thetasp > thetaspcr Then     'Shadows cast by adjacent rows and H3 is raised
    
    Dim m3 As Double
    Dim b3 As Double
    Dim AA As Double
    Dim BB As Double
    Dim CC As Double
    Dim Xs3 As Double
    Dim Ys3 As Double
    
    m3 = 1 / Tan(thetasp)
    b3 = -Ys - m3 * (row - Xs)
    AA = (bc ^ 2) + ((wc / 2) ^ 2) * (m3 ^ 2)
    BB = 2 * m3 * b3 * ((wc / 2) ^ 2)
    CC = ((wc / 2) ^ 2) * (b3 ^ 2) - ((wc / 2) ^ 2) * (bc ^ 2)
    Xs3 = (-BB + Sqr((BB ^ 2) - 4 * AA * CC)) / (2 * AA)    'Positive root (negative root taken for H5)
                                                 Ys3 = m3 * Xs3 + b3
                                                 X3 = Dv / Sin(psi) * ((Pr - Xs3) / (Dv - bc - Ys3))

                                             Else    'Compute X3 as normal (no shading by adjacent row)
    
    X3 = Dv / Sin(psi) * ((Pr - Xs) / (Dv - bc - Ys))

End If

Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
                                             OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

                                             Dim MIN As Double
                                             If Abs(Tan(rawpsi)) < 1 Then
                                             MIN = Tan(Pi / 2 - psi)
                                             Else
                                             MIN = 1
                                             End If

                                             Heighte3 = X3 - OC * MIN

                                             End Function

                                             Function Heighte4(Pr As Double, row As Double, theta As Double, rawpsi As Double, _
                                                               hc As Double, wc As Double, Dv As Double, FOV As Double, _
                                                               thetas As Double, psis As Double) As Double

                                             'Heighte4 = Distance from center of footprint of radiometer to chord projected by
'sunlit-shaded soil boundary on near-side of canopy, where crop canopy is modelled as ELLIPSE (m)
                                             'Pr = Perpendicular distance of radiometer from canopy row center (m)
'row = Crop row spacing (m)
                                             'Theta = Zenith angle of radiometer (radians)
'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
                                             '   where zero is looking parallel to crop row and pi/2 is looking perpendicular
'   to crop row (radians)
                                             'hc = Canopy height (m)
'wc = canopy width (m)
                                             'Dv = Vertical height of radiometer relative to soil (m)
'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
                                             'thetas = Solar zenith angle (radians)
'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
                                             'crop row and pi/2 is looking perpendicular to crop row (radians)

'Assign value to pi
                                             Dim Pi As Double
                                             Pi = 3.14159265358979

                                             'Constrain 45 < psi < 90
Dim psi As Double
If Abs(rawpsi) < 45 * Pi / 180 Then
    psi = Pi / 2 - (Abs(rawpsi))
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psi = (Abs(rawpsi)) - Pi / 2
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psi = Pi - Abs(rawpsi)
        Else
            psi = Abs(rawpsi)
        End If
    End If
End If

Dim bc As Double    'Vertical axis of elliptical canopy (m)
                                             Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
Dim Xs As Double    'Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
                                             Dim Ys As Double    'Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
Dim X4 As Double    'Horizontal distance from radiometer to ground-projected
                                             'sunlit-shaded soil boundary on near-side of canopy (m)
bc = hc / 2
thetasp = -thetas * Sin(psis)
Xs = wc / 2 / Sqr(1 + (bc ^ 2) / ((wc / 2) ^ 2) * ((Tan(thetasp)) ^ 2))
Ys = -(bc ^ 2) / ((wc / 2) ^ 2) * Xs * Tan(thetasp)
X4 = (Pr - Xs + (Tan(thetasp)) * (bc + Ys)) / Sin(psi)

Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
                                             OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

                                             Dim MIN As Double
                                             If Abs(Tan(rawpsi)) < 1 Then
                                             MIN = Tan(Pi / 2 - psi)
                                             Else
                                             MIN = 1
                                             End If

                                             Heighte4 = X4 - OC * MIN

                                             End Function
                                             Function Heighte5(Pr As Double, row As Double, theta As Double, rawpsi As Double, _
                                                               hc As Double, wc As Double, Dv As Double, FOV As Double, _
                                                               thetas As Double, psis As Double) As Double

                                             'Heighte5 = Distance from center of footprint of radiometer to chord projected by
'sunlit-shaded boundary on far-side of canopy, where crop canopy is modelled as ELLIPSE (m)
                                             'Pr = Perpendicular distance of radiometer from canopy row center (m)
'row = Crop row spacing (m)
                                             'Theta = Zenith angle of radiometer (radians)
'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
                                             '   where zero is looking parallel to crop row and pi/2 is looking perpendicular
'   to crop row (radians)
                                             'hc = Canopy height (m)
'wc = canopy width (m)
                                             'Dv = Vertical height of radiometer relative to soil (m)
'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
                                             'thetas = Solar zenith angle (radians)
'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
                                             'crop row and pi/2 is looking perpendicular to crop row (radians)

'Assign value to pi
                                             Dim Pi As Double
                                             Pi = 3.14159265358979

                                             'Constrain 45 < psi < 90
Dim psi As Double
If Abs(rawpsi) < 45 * Pi / 180 Then
    psi = Pi / 2 - (Abs(rawpsi))
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psi = (Abs(rawpsi)) - Pi / 2
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psi = Pi - Abs(rawpsi)
        Else
            psi = Abs(rawpsi)
        End If
    End If
End If

Dim bc As Double    'Vertical axis of elliptical canopy (m)
                                             Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
Dim Xs As Double    'Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
                                             Dim Ys As Double    'Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
Dim X5 As Double    'Horizontal distance from radiometer to ground-projected
                                             'sunlit-shaded boundary on canopy (m)
bc = hc / 2
thetasp = -thetas * Sin(psis)
Xs = -wc / 2 / Sqr(1 + (bc ^ 2) / ((wc / 2) ^ 2) * ((Tan(thetasp)) ^ 2)) 'Xs is negative (positive for H3)
                                             Ys = -(bc ^ 2) / ((wc / 2) ^ 2) * Xs * Tan(thetasp) 'Ys is positive or negative

'Determine critical perpendicular solar zenith angle,
                                             'beyond which results in adjacent row shading
Dim Xscr As Double  'Horizontal distance from canopy ellipse origin to tangent of sunray
                                             'along thetasp (m)
Dim Yscr As Double  'Vertical distance from canopy ellipse origin to tangent of sunray
                                             'along thetasp (m)
Dim thetaspcr As Double     'Critical perpendicular solar zenith angle

                                             Xscr = -2 * ((wc / 2) ^ 2) / row     'Xscr is negative (positive for H3)
Yscr = -Sqr(((bc ^ 2) * Xscr * (row + 2 * Xscr)) / -2 / ((wc / 2) ^ 2))  'Yscr is negative; note -2 and row+2*Xscr
                                             thetaspcr = Atn(-((wc / 2) ^ 2) * Yscr / (bc ^ 2) / Xscr)   'thetascr is negative

If thetasp < thetaspcr Then     'Shadows cast by adjacent rows and H5 is raised.
                                             'NOTE: thetascr is negative (thetascr for H3 is positive)
    
    Dim m5 As Double
    Dim b5 As Double
    Dim AA As Double
    Dim BB As Double
    Dim CC As Double
    Dim Xs5 As Double
    Dim Ys5 As Double
    
    m5 = 1 / Tan(thetasp)
    b5 = -Ys + m5 * (row + Xs) 'NOTE: signs are negative for H3
                                                 AA = (bc ^ 2) + ((wc / 2) ^ 2) * (m5 ^ 2)
                                                 BB = 2 * m5 * b5 * ((wc / 2) ^ 2)
                                                 CC = ((wc / 2) ^ 2) * (b5 ^ 2) - ((wc / 2) ^ 2) * (bc ^ 2)
                                                 Xs5 = (-BB - Sqr((BB ^ 2) - 4 * AA * CC)) / (2 * AA)    'NOTE: Negative root (H3 positive)
    Ys5 = m5 * Xs5 + b5
    X5 = Dv / Sin(psi) * ((Pr - Xs5) / (Dv - bc - Ys5))

Else    'Compute X5 as normal (no shading by adjacent row)

                                                 X5 = Dv / Sin(psi) * ((Pr - Xs) / (Dv - bc - Ys))

                                             End If

                                             Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

Dim MIN As Double
If Abs(Tan(rawpsi)) < 1 Then
MIN = Tan(Pi / 2 - psi)
Else
MIN = 1
End If

Heighte5 = X5 - OC * MIN

End Function
Function Heighte6(Pr As Double, row As Double, theta As Double, rawpsi As Double, _
hc As Double, wc As Double, Dv As Double, FOV As Double, _
thetas As Double, psis As Double) As Double

'Heighte46 = Distance from center of footprint of radiometer to chord projected by
                                             'sunlit-shaded soil boundary on far-side of canopy, where crop canopy is modelled as ELLIPSE (m)
'Pr = Perpendicular distance of radiometer from canopy row center (m)
                                             'row = Crop row spacing (m)
'Theta = Zenith angle of radiometer (radians)
                                             'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
'   where zero is looking parallel to crop row and pi/2 is looking perpendicular
                                             '   to crop row (radians)
'hc = Canopy height (m)
                                             'wc = canopy width (m)
'Dv = Vertical height of radiometer relative to soil (m)
                                             'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
'thetas = Solar zenith angle (radians)
                                             'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
'crop row and pi/2 is looking perpendicular to crop row (radians)

                                             'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

'Constrain 45 < psi < 90
                                             Dim psi As Double
                                             If Abs(rawpsi) < 45 * Pi / 180 Then
                                                 psi = Pi / 2 - (Abs(rawpsi))
                                             Else
                                                 If Abs(rawpsi) > 135 * Pi / 180 Then
                                                     psi = (Abs(rawpsi)) - Pi / 2
                                                 Else
                                                     If Abs(rawpsi) > 90 * Pi / 180 Then
                                                         psi = Pi - Abs(rawpsi)
                                                     Else
                                                         psi = Abs(rawpsi)
                                                     End If
                                                 End If
                                             End If

                                             Dim bc As Double    'Vertical axis of elliptical canopy (m)
Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
                                             Dim Xs As Double    'Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
Dim Ys As Double    'Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
                                             Dim X6 As Double    'Horizontal distance from radiometer to ground-projected
'sunlit-shaded soil boundary on near-side of canopy (m)
                                             bc = hc / 2
                                             thetasp = -thetas * Sin(psis)
                                             Xs = -wc / 2 / Sqr(1 + (bc ^ 2) / ((wc / 2) ^ 2) * ((Tan(thetasp)) ^ 2))    'NOTE: negative sign (H4 positive)
Ys = -(bc ^ 2) / ((wc / 2) ^ 2) * Xs * Tan(thetasp)
X6 = (Pr - Xs + (Tan(thetasp)) * (bc + Ys)) / Sin(psi)

Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
                                             OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

                                             Dim MIN As Double
                                             If Abs(Tan(rawpsi)) < 1 Then
                                             MIN = Tan(Pi / 2 - psi)
                                             Else
                                             MIN = 1
                                             End If

                                             Heighte6 = X6 - OC * MIN

                                             End Function

                                             Function quartic1IRT(X1 As Double, Y1 As Double, ac As Double, bc As Double) As Double
                                             'Compute the 1st root of a quartic equation of form
'Ax^4 + Bx^3 + Cx^2 + Dx + E = 0

                                             Dim A As Double
                                             Dim b As Double
                                             Dim c As Double
                                             Dim d As Double
                                             Dim E As Double
                                             Dim alpha As Double
                                             Dim beta As Double
                                             Dim gamma As Double
                                             Dim P As Double
                                             Dim Q As Double
                                             Dim r As Double
                                             Dim u As Double
                                             Dim y As Double
                                             Dim W As Double

                                             'Compute coefficients of quartic
A = (Y1 ^ 2) * (bc ^ 2) - (bc ^ 4)
b = -2 * X1 * Y1 * (bc ^ 2)
c = (Y1 ^ 2) * (ac ^ 2) + (X1 ^ 2) * (bc ^ 2) - 2 * (ac ^ 2) * (bc ^ 2)
d = -2 * X1 * Y1 * (ac ^ 2)
E = (X1 ^ 2) * (ac ^ 2) - (ac ^ 4)

If Abs(X1) < 0.001 Then     'Assume quadratic equation
                                             quartic1IRT = -Sqr((-c + Sqr(c ^ 2 - 4 * A * E)) / (2 * A))
                                             GoTo 20
                                             End If

                                             'Compute quartic
alpha = -((3 * b ^ 2) / (8 * A ^ 2)) + c / A
beta = (b ^ 3) / (8 * A ^ 3) - (b * c) / (2 * A ^ 2) + d / A
gamma = -(3 * b ^ 4) / (256 * A ^ 4) + (c * b ^ 2) / (16 * A ^ 3) - (b * d) / (4 * A ^ 2) + E / A
P = -((alpha ^ 2) / 12) - gamma
Q = -(alpha ^ 3) / (108) + (alpha * gamma / 3) - (beta ^ 2) / 8
r = Q / 2 + Sqr((Q ^ 2) / 4 + (P ^ 3) / 27)
If r < 0 Then
u = -((Abs(r)) ^ (1 / 3))
Else
u = r ^ (1 / 3)
End If

Dim Uy As Double
    If u = 0 Then
        Uy = 0
    Else
        Uy = P / (3 * u)
    End If
y = -5 / 6 * alpha - u + Uy
W = Sqr(Abs(alpha + 2 * y))

Dim ZZ As Double
If W = 0 Then
    ZZ = Abs(3 * alpha + 2 * y)
    quartic1IRT = -(b / (4 * A)) + (W - Sqr(ZZ)) / 2 - y '(y subtracted to smooth curve)
                                                 GoTo 20
                                             Else
                                                 ZZ = -(3 * alpha + 2 * y + 2 * beta / W)
                                                 If ZZ < 0 Then
                                                     ZZ = Abs(3 * alpha + 2 * y - 2 * beta / W)
                                                     quartic1IRT = -(b / (4 * A)) + (-W - Sqr(ZZ)) / 2
                                                 Else
                                                     quartic1IRT = -(b / (4 * A)) + (W - Sqr(ZZ)) / 2
                                                 End If
                                             End If

                                             20 End Function

                                             Function quartic2IRT(X1 As Double, Y1 As Double, ac As Double, bc As Double) As Double
                                             'Compute the 2nd root of a quartic equation of form
'Ax^4 + Bx^3 + Cx^2 + Dx + E = 0

                                             Dim A As Double
                                             Dim b As Double
                                             Dim c As Double
                                             Dim d As Double
                                             Dim E As Double
                                             Dim alpha As Double
                                             Dim beta As Double
                                             Dim gamma As Double
                                             Dim P As Double
                                             Dim Q As Double
                                             Dim r As Double
                                             Dim u As Double
                                             Dim y As Double
                                             Dim W As Double

                                             'Compute coefficients of quartic
A = (Y1 ^ 2) * (bc ^ 2) - (bc ^ 4)
b = -2 * X1 * Y1 * (bc ^ 2)
c = (Y1 ^ 2) * (ac ^ 2) + (X1 ^ 2) * (bc ^ 2) - 2 * (ac ^ 2) * (bc ^ 2)
d = -2 * X1 * Y1 * (ac ^ 2)
E = (X1 ^ 2) * (ac ^ 2) - (ac ^ 4)

If Abs(X1) < 0.001 Then     'Assume quadratic equation
                                             quartic2IRT = Sqr((-c + Sqr(c ^ 2 - 4 * A * E)) / (2 * A))
                                             GoTo 20
                                             End If

                                             'Compute quartic
alpha = -((3 * b ^ 2) / (8 * A ^ 2)) + c / A
beta = (b ^ 3) / (8 * A ^ 3) - (b * c) / (2 * A ^ 2) + d / A
gamma = -(3 * b ^ 4) / (256 * A ^ 4) + (c * b ^ 2) / (16 * A ^ 3) - (b * d) / (4 * A ^ 2) + E / A
P = -((alpha ^ 2) / 12) - gamma
Q = -(alpha ^ 3) / (108) + (alpha * gamma / 3) - (beta ^ 2) / 8
r = Q / 2 + Sqr((Q ^ 2) / 4 + (P ^ 3) / 27)
If r < 0 Then
u = -((Abs(r)) ^ (1 / 3))
Else
u = r ^ (1 / 3)
End If

Dim Uy As Double
    If u = 0 Then
        Uy = 0
    Else
        Uy = P / (3 * u)
    End If
y = -5 / 6 * alpha - u + Uy
W = Sqr(Abs(alpha + 2 * y))

Dim ZZ As Double
If W = 0 Then
    ZZ = Abs(3 * alpha + 2 * y)
    quartic2IRT = -(b / (4 * A)) + (-W + Sqr(ZZ)) / 2 - y '(y subtracted to smooth curve)
                                                 GoTo 20
                                             Else
                                                 ZZ = -(3 * alpha + 2 * y + 2 * beta / W)
                                                 If ZZ < 0 Then
                                                     ZZ = Abs(3 * alpha + 2 * y - 2 * beta / W)
                                                     quartic2IRT = -(b / (4 * A)) + (-W + Sqr(ZZ)) / 2
                                                 Else
                                                     quartic2IRT = -(b / (4 * A)) + (W + Sqr(ZZ)) / 2
                                                 End If
                                             End If

                                             20 End Function
                                             'Function chord to compute the chord area of an ellipse
Function Chord(rawpsi As Double, MAJOR As Double, MINOR As Double, H As Double) As Double

'Assign value to pi
                                             Dim Pi As Double
                                             Pi = 3.14159265358979

                                             Dim X As Double     'Axis used to compute chord area where H crosses (m)
Dim y As Double     'Axis perpendicular to X (m)
                                             Dim psi As Double   'Angle of chord wrt Y-axis, constrained to 45 < psi < 90 (rad)

'Constrain 45 < psi < 90, specify X and Y
                                             If Abs(rawpsi) < 45 * Pi / 180 Then
                                                 psi = Pi / 2 - (Abs(rawpsi))
                                                 X = MINOR
                                                 y = MAJOR
                                             Else
                                                 If Abs(rawpsi) > 135 * Pi / 180 Then
                                                     psi = (Abs(rawpsi)) - Pi / 2
                                                     X = MINOR
                                                     y = MAJOR
                                                 Else
                                                     If Abs(rawpsi) > 90 * Pi / 180 Then
                                                         psi = Pi - Abs(rawpsi)
                                                         X = MAJOR
                                                         y = MINOR
                                                     Else
                                                         psi = Abs(rawpsi)
                                                         X = MAJOR
                                                         y = MINOR
                                                     End If
                                                 End If
                                             End If

                                             'Compute distance from ellipse center to tangent (maximum possible H to form a chord)
'Dim ec As Double 'Ellipse eccentricity (m)
'Dim asin1 As Double 'Parameter used to compute inverse sine
'Dim alpha1 As Double    'Internal angle (rad)
'Dim HH1 As Double    'Radius of internal construction circle (m)
'Dim T1 As Double    'Distance from ellipse center to tangent (m)

'ec = Sqr(1 - (X ^ 2) / (y ^ 2))
                                             'asin1 = ec * Sin(psi + Pi / 2)
'alpha1 = Atn(asin1 / Sqr(-asin1 * asin1 + 1))
                                             'HH1 = y * (Sin(Pi / 2 - psi - alpha1)) / (Sin(psi + Pi / 2))
'T1 = ec * y + HH1 / Sin(psi)

                                             'Compute T1
Dim T1 As Double    'Distance from radiometer ellipse footprint center to line
                                             '   tangent to radiometer ellipse for given azimuth and zenith angles (m).
If (Abs(psi - 90 * Pi / 180) < 0.01) Then   '90 degree radiometer azimuth
                                                 T1 = y
                                                 If H > T1 Then  'Chord outside of Ellipse
        Chord = 0
        GoTo 20
    Else            'Case 2 with equilateral triangle
                                                     Chord = X * y * Atn((Sqr(y ^ 2 - H ^ 2)) / H) - H * X / y * Sqr(y ^ 2 - H ^ 2)
                                                     GoTo 20
                                                 End If
                                             Else
                                                 If Abs(psi) < 0.01 Then     'ZERO degree radiometer azimuth
        T1 = X
        If H > T1 Then  'Chord outside of Ellipse
                                                         Chord = 0
                                                         GoTo 20
                                                     Else            'Case 4 with equilateral triangle
            Chord = X * y * Atn((Sqr(X ^ 2 - H ^ 2)) / H) - H * y / X * Sqr(X ^ 2 - H ^ 2)
            GoTo 20
        End If
    Else                        'Radiometer azimuth between ZERO and 90 degrees
                                                     Dim xtr1 As Double
                                                     Dim ytr1 As Double
                                                     xtr1 = y / Sqr(1 + (X ^ 2) / (y ^ 2) / ((Tan(psi)) ^ 2))
                                                     ytr1 = (X ^ 2) / (y ^ 2) * xtr1 / Tan(psi)
                                                     T1 = xtr1 + ytr1 / Tan(psi)
                                                 End If
                                             End If

                                             'Determine if H cuts a chord through ellipse
If H > T1 Then
Chord = 0
GoTo 20
End If

If Abs(X - y) < 0.001 Then 'Ellipse is a circle
                                             Dim hc As Double
                                             Dim yy As Double
                                             Dim Acosyy As Double
                                             hc = H * Sin(psi)
                                             yy = hc / X
                                             Acosyy = Atn(-yy / Sqr(-yy * yy + 1)) + 2 * Atn(1)
                                             Chord = (X ^ 2) * Acosyy - hc * Sqr(X ^ 2 - hc ^ 2)
                                             GoTo 20
                                             Else
                                             GoTo 10
                                             End If

                                             'Chord = Chord area (m2)
'psi = Azimuth angle (radians)
                                             'Y = Major axis (m)
'X = Minor axis (m)
                                             'H = Distance from ellipse center to chord along major axis Y (m)
'C = Case number, where
                                             'CASE 1: X > H*tan(psi); H < Y
'CASE 2: X < H*tan(psi); H < Y
                                             'CASE 3: X < H*tan(psi); H > Y
'CASE 4: X > H*tan(psi); H > Y

                                             'Determine chord case
10 Dim CC As Integer
If H < y Then
    If X > H * Tan(psi) Then
        CC = 1#
        Else
        CC = 2#
    End If
Else
    If X > H * Tan(psi) Then
        CC = 4#
        Else
        CC = 3#
    End If
End If

'See diagrams for variables used below
                                             Dim aq As Double    'Used in quadratic equation for q
Dim bq As Double    'Used in quadratic equation for q
                                             Dim cq As Double    'Used in quadratic equation for q
Dim Q As Double
Dim s As Double
Dim alpha As Double
Dim A As Double

Dim at As Double    'Used in quadratic equation for t
                                             Dim bt As Double    'Used in quadratic equation for t
Dim ct As Double    'Used in quadratic equation for t
                                             Dim t As Double
                                             Dim u As Double
                                             Dim beta As Double
                                             Dim b As Double

                                             'Declare sector and triangle variables, where Chord = Aes - Aet
Dim Aes As Double   'Area enclosed by sector (m2)
                                             Dim Aet As Double   'Area enclosed by triangle (m2)

If CC = 1 Then

aq = (y ^ 2) / (X ^ 2) + (1 / Tan(psi)) ^ 2
bq = 2 * H / Tan(psi)
cq = (H ^ 2) - (y ^ 2)
Q = (-bq + Sqr((bq ^ 2) - 4 * aq * cq)) / (2 * aq)
s = Q / Tan(psi)
alpha = Atn(Q / (H + s))
A = Sqr((Q ^ 2) + ((H + s) ^ 2))

at = (X ^ 2) / (y ^ 2) + (Tan(psi)) ^ 2
bt = 2 * H * ((Tan(psi)) ^ 2)   'negative for Cases 2 & 3
                                             ct = (H ^ 2) * ((Tan(psi)) ^ 2) - (X ^ 2)
                                             t = (-bt + Sqr((bt ^ 2) - 4 * at * ct)) / (2 * at)
                                             u = (H + t) * Tan(psi)  '-t for Cases 2 & 3
beta = Atn(t / u)       'Inverse for Cases 2 & 3
                                             b = Sqr((t ^ 2) + (u ^ 2))

                                             Aes = X * y / 2 * (Atn(y / X * Tan(alpha)) + Atn(X / y * Tan(beta)) + Pi / 2)
                                             Aet = A * b / 2 * Sin(alpha + beta + Pi / 2)
                                             Chord = Aes - Aet
                                             GoTo 20
                                             End If

                                             If CC = 2 Then

                                             aq = (y ^ 2) / (X ^ 2) + (1 / Tan(psi)) ^ 2
                                             bq = 2 * H / Tan(psi)
                                             cq = (H ^ 2) - (y ^ 2)
                                             Q = (-bq + Sqr((bq ^ 2) - 4 * aq * cq)) / (2 * aq)
                                             s = Q / Tan(psi)
                                             alpha = Atn(Q / (H + s))
                                             A = Sqr((Q ^ 2) + ((H + s) ^ 2))

                                             at = (X ^ 2) / (y ^ 2) + (Tan(psi)) ^ 2
                                             bt = -2 * H * ((Tan(psi)) ^ 2)
                                             ct = (H ^ 2) * ((Tan(psi)) ^ 2) - (X ^ 2)
                                             t = (-bt - Sqr((bt ^ 2) - 4 * at * ct)) / (2 * at)
                                             u = (H - t) * Tan(psi)  '+t for case 1
beta = Atn(u / t)       'Inverse for case 1
                                             b = Sqr((t ^ 2) + (u ^ 2))

                                             Aes = X * y / 2 * (Atn(y / X * Tan(alpha)) + Atn(y / X * Tan(beta)))
                                             Aet = A * b / 2 * Sin(alpha + beta)
                                             Chord = Aes - Aet
                                             GoTo 20
                                             End If

                                             If CC = 3 Then

                                             aq = (y ^ 2) / (X ^ 2) + (1 / Tan(psi)) ^ 2 'tan for Cases 1 & 2
bq = -2 * H / Tan(psi)                  'positive and tan for Cases 1 & 2
                                             cq = (H ^ 2) - (y ^ 2)
                                             Q = (-bq - Sqr((bq ^ 2) - 4 * aq * cq)) / (2 * aq)
                                             s = Q / Tan(psi)            'tan for Cases 1 & 2
alpha = Atn(Q / (H - s))    '+s for Cases 1 & 2
                                             A = Sqr((Q ^ 2) + ((H - s) ^ 2))   '+s for Cases 1 & 2

at = (X ^ 2) / (y ^ 2) + (Tan(psi)) ^ 2
bt = -2 * H * ((Tan(psi)) ^ 2)
ct = (H ^ 2) * ((Tan(psi)) ^ 2) - (X ^ 2)
t = (-bt - Sqr((bt ^ 2) - 4 * at * ct)) / (2 * at)
u = (H - t) * Tan(psi)  '+t for case 1
                                             beta = Atn(u / t)       'Inverse for case 1
b = Sqr((t ^ 2) + (u ^ 2))

Aes = X * y / 2 * (Atn(y / X * Tan(beta)) - Atn(y / X * Tan(alpha)))
Aet = A * b / 2 * Sin(beta - alpha)
Chord = Aes - Aet
GoTo 20
End If

If CC = 4 Then

aq = (y ^ 2) / (X ^ 2) + (1 / Tan(psi)) ^ 2 'tan for Cases 1 & 2
                                             bq = -2 * H / Tan(psi)                  'positive and tan for Cases 1 & 2
cq = (H ^ 2) - (y ^ 2)
Q = (-bq - Sqr((bq ^ 2) - 4 * aq * cq)) / (2 * aq)
s = Q / Tan(psi)            'tan for Cases 1 & 2
                                             alpha = Atn(Q / (H - s))    '+s for Cases 1 & 2
A = Sqr((Q ^ 2) + ((H - s) ^ 2))   '+s for Cases 1 & 2

                                             at = (X ^ 2) / (y ^ 2) + (Tan(psi)) ^ 2
                                             bt = 2 * H * ((Tan(psi)) ^ 2)   'negative for Cases 2 & 3
ct = (H ^ 2) * ((Tan(psi)) ^ 2) - (X ^ 2)
t = (-bt + Sqr((bt ^ 2) - 4 * at * ct)) / (2 * at)
u = (H + t) * Tan(psi)  '-t for Cases 2 & 3
                                             beta = Atn(t / u)       'Inverse for Cases 2 & 3
b = Sqr((t ^ 2) + (u ^ 2))

Aes = X * y / 2 * (Atn(X / y * Tan(beta)) + Pi / 2 - Atn(y / X * Tan(alpha)))
Aet = A * b / 2 * Sin(-alpha + beta + Pi / 2)

Chord = Aes - Aet
GoTo 20
End If

20 End Function


Attribute VB_Name = "Module4"
Option Explicit
Function fveg(XE As Double, thetar As Double, rawpsi As Double, hc As Double, _
wc As Double, row As Double, LAI As Double, Pr As Double, Vr As Double, _
FOV As Double, thetas As Double, psis As Double) As Double
'Function fveg to compute the fraction of vegetation appearing in a radiometer footprint
                                             'where the crop rows are modeled as continuous ellipses.

'LADFOPT = Option for leaf angle distribution function (LADF), where
                                             '           (1 = Ellipsoidal, 2 = Beta)
'For ellipsoidal LADF option:
                                                 'Xe = Ratio of horizontal to vertical projected leaves (spherical Xe = 1)
'For Beta LADF option:
                                                 'tAVG = mean of normalized leaf angle t (e.g., 0.5 for symmetric PDF's)
    'tVAR = variance of normalized leaf angle t

'thetar = Radiometer view zenith angle (radians)
    'rawpsi = Azimuth angel of crop row relative to radiometer view angle,
'   where zero is looking parallel to crop row and pi/2 is looking perpendicular
    '   to crop row (radians)
'hc = Canopy height (m)
    'wc = Canopy width (m)
'row = Crop row spacing (m)
    'LAI = Leaf area index (m2 m-2)

'Pr = Perpendicular distance of radiometer from canopy row center (m)
    'Vr = Vertical height of radiometer relative to soil (m)
'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
    'thetas = Solar zenith angle (rad)
'psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    '       and 90 degrees for perpendicular orientation (rad)

'Variables used in this function
    Dim KBR As Double   'Extinction coefficient of radiometer viewing canopy
Dim PFR As Double    'Path length fraction of radiometer viewing continuous ellipse
    Dim MFR As Double    'Multiple row function of radiometer viewing continuous ellipses
Dim psir As Double  'radiometer azimuth angle relative to crop row, constrained to 45-90 deg
    Dim ER As Double    'Extinction of radiometer view path through canopy
Dim fcs1 As Double  'Fraction of solid sunlit continuous ellispe appearing in radiometer footprint
    Dim fcs2 As Double  'Fraction of solid shaded continuous ellispe appearing in radiometer footprint

'Assign value to pi
    Dim Pi As Double
    Pi = 3.14159265358979

    KBR = (Sqr(XE ^ 2 + (Tan(thetar)) ^ 2)) / _
                    (XE + 1.774 * (XE + 1.182) ^ -0.733)

                    'Constrain 45 < rawpsi < 90
If Abs(rawpsi) < 45 * Pi / 180 Then
    psir = Pi / 2 - (Abs(rawpsi))
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psir = (Abs(rawpsi)) - Pi / 2
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psir = Pi - Abs(rawpsi)
        Else
            psir = Abs(rawpsi)
        End If
    End If
End If

PFR = PLF(thetar, psir, hc, wc, row)
MFR = MRF(thetar, psir, hc, wc, row)

ER = Exp(-KBR * row / wc * LAI * PFR * MFR)

fcs1 = fcs(1, Pr, row, thetar, rawpsi, hc, wc, Vr, FOV, thetas, psis)
fcs2 = fcs(2, Pr, row, thetar, rawpsi, hc, wc, Vr, FOV, thetas, psis)

fveg = (fcs1 + fcs2) * (1 - ER)

End Function
Function fcs(OPT As Integer, Pr As Double, row As Double, theta As Double, rawpsi As Double, _
hc As Double, wc As Double, Dv As Double, FOV As Double, thetas As Double, psis As Double) As Double

'Function fcs to compute the fraction of sunlit or shaded canopy or soil
                    'appearing in the elliptical footprint of a radiometer, where the canopy
'is modelled as ELLIPSE

                    'OPT = 1 for sunlit canopy
'      2 for shaded canopy
                    '      3 for sunlit soil
'      4 for shaded soil

                    'Pr = Perpendicular distance of radiometer from canopy row center (m)
'row = Crop row spacing (m)
                    'Theta = Zenith angle of radiometer (radians)
'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
                    '   where zero is looking parallel to crop row and pi/2 is looking perpendicular
'   to crop row (radians)
                    'hc = Canopy height (m)
'wc = canopy width (m)
                    'Dv = Vertical height of radiometer relative to soil (m)
'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
                    'thetas = Solar zenith angle (radians)
'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
                    'crop row and pi/2 is looking perpendicular to crop row (radians)

'Assign value to pi
                    Dim Pi As Double
                    Pi = 3.14159265358979

                    'Compute major (ar) and minor (br) axes of radiometer footprint

Dim ar As Double
Dim OA As Double
Dim br As Double

ar = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) - Tan(theta - Atn(1 / 2 / FOV)))
OA = Dv * Tan(theta - Atn(1 / 2 / FOV))
br = (Sqr(((ar + OA) ^ 2) + (Dv ^ 2))) / (2 * FOV)

Dim Aell As Double    'Total elliptical area of radiometer footprint (m2)
                    Aell = ar * br * Pi

                    'Constrain 45 < psi < 90
Dim psi As Double
If Abs(rawpsi) < 45 * Pi / 180 Then
    psi = Pi / 2 - (Abs(rawpsi))
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psi = (Abs(rawpsi)) - Pi / 2
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psi = Pi - Abs(rawpsi)
        Else
            psi = Abs(rawpsi)
        End If
    End If
End If

'Compute T1
                    Dim T1 As Double    'Distance from radiometer ellipse footprint center to line
'   tangent to radiometer ellipse for given azimuth and zenith angles (m).
                    If (Abs(psi - 90 * Pi / 180) < 0.01) Then   '90 degree radiometer azimuth
    T1 = ar
    GoTo 10
Else
    If Abs(psi) < 0.01 Then     'ZERO degree radiometer azimuth
                            T1 = br
                                    GoTo 10
                                        Else                        'Radiometer azimuth between ZERO and 90 degrees
        Dim xtr1 As Double
        Dim ytr1 As Double
        xtr1 = ar / Sqr(1 + (br ^ 2) / (ar ^ 2) / ((Tan(psi)) ^ 2))
        ytr1 = (br ^ 2) / (ar ^ 2) * xtr1 / Tan(psi)
        T1 = xtr1 + ytr1 / Tan(psi)
    End If
End If

'Determine number of rows appearing in radiometer footprint, make it an odd number,
                                        'and add TWO extra rows either side to account for adjacent row shading
10 Dim NR As Integer
NR = 2 * (Application.WorksheetFunction.RoundUp((2 * T1 * Sin(psi) / row), 0)) + 1# + 4#

Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
                                        Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
thetasp = -thetas * Sin(psis)

Dim MIN As Double
If Abs(Tan(rawpsi)) < 1 Then
MIN = Tan(Pi / 2 - psi)
Else
MIN = 1
End If

OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

'Declare arrays
                                        ReDim Pnr(NR) As Double
                                        ReDim H(NR, 6) As Double
                                        ReDim tantheta(NR, 6) As Double
                                        ReDim ac(NR, 6) As Double
                                        ReDim fc(NR, 4) As Double
                                        Dim fcr(4) As Double

                                        Dim N1 As Integer
                                        For N1 = 1 To NR 'Row number

    Pnr(N1) = Pr + (row / 2) * (2 * N1 - NR - 1)
    
    H(N1, 1) = Heighte1(Pnr(N1), theta, rawpsi, hc, wc, Dv, FOV)
    H(N1, 2) = Heighte2(Pnr(N1), theta, rawpsi, hc, wc, Dv, FOV)
    H(N1, 3) = Heighte3(Pnr(N1), row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
    H(N1, 4) = Heighte4(Pnr(N1), row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
    H(N1, 5) = Heighte5(Pnr(N1), row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)
    H(N1, 6) = Heighte6(Pnr(N1), row, theta, rawpsi, hc, wc, Dv, FOV, thetas, psis)

'    tantheta(N1, 1) = (H(N1, 1) + OC * MIN) / Dv
                                        '    tantheta(N1, 2) = (H(N1, 2) + OC * MIN) / Dv
'    tantheta(N1, 3) = (H(N1, 3) + OC * MIN) / Dv
                                        '    tantheta(N1, 4) = (H(N1, 4) + OC * MIN) / Dv
'    tantheta(N1, 5) = (H(N1, 5) + OC * MIN) / Dv
                                        '    tantheta(N1, 6) = (H(N1, 6) + OC * MIN) / Dv

    If H(N1, 4) > H(N1, 1) Then
        H(N1, 4) = H(N1, 1) 'H1 obscurs H4 (no shaded soil visible on near-side)
                                                H(N1, 3) = H(N1, 1) 'H1 obscurs H3 (no shaded canopy visible on near-side)
    End If
    
    If H(N1, 6) < H(N1, 2) Then
        H(N1, 6) = H(N1, 2) 'H2 obscurs H6 (no shaded soil visible on far-side)
                                                        H(N1, 5) = H(N1, 2) 'H2 obscurs H5 (no shaded canopy visible on far-side)
    End If
    

'    If H(N1, 4) > H(N1, 1) Then H(N1, 4) = H(N1, 1) 'H1 obscurs H4 (no shaded soil visible on near-side)
'    If H(N1, 3) < H(N1, 1) Then H(N1, 3) = H(N1, 1) 'H1 obscurs H3 (no shaded canopy visible on near-side)
'    If H(N1, 6) < H(N1, 2) Then H(N1, 6) = H(N1, 2) 'H2 obscurs H6 (no shaded soil visible on far-side)
'    If H(N1, 5) > H(N1, 2) Then H(N1, 5) = H(N1, 2) 'H2 obscurs H5 (no shaded canopy visible on far-side)
    
Next

Dim N2 As Integer
For N2 = 2 To NR   'Account for adjacent rows obscuring chord locations

                                                            If H((N2 - 1), 2) > H(N2, 4) Then H(N2, 4) = H((N2 - 1), 2)
                                                                'Far side of canopy boundary in row N2-1 obscurs near side of sunlit-shaded soil boundary in row N2
    If H((N2 - 1), 2) > H(N2, 1) Then H(N2, 1) = H((N2 - 1), 2)
    'Far side of canopy boundary in row N2-1 obscurs near side of canopy in row N2
                                                                    If H((N2 - 1), 2) > H(N2, 3) Then H(N2, 3) = H((N2 - 1), 2)
                                                                        'Far side of canopy boundary in row N2-1 obscurs near side of sunlit-shaded canopy boundary in row N2
    
    If H(N2, 1) < H((N2 - 1), 6) Then H((N2 - 1), 6) = H(N2, 1)
    'Near side of canopy in row N2 obscurs far side of sunlit-shaded soil boundary in row N2-1
                                                                            If H(N2, 1) < H((N2 - 1), 2) Then H((N2 - 1), 2) = H(N2, 1)
                                                                                'Near side of canopy in row N2 obscurs far side of canopy in row N2-1
    If H(N2, 1) < H((N2 - 1), 5) Then H((N2 - 1), 5) = H(N2, 1)
    'Near side of canopy in row N2 obscurs far side of sunlit-shaded canopy boundary in row N2-1

                                                                                Next

                                                                                'Build array of chord areas

Dim N3 As Integer
For N3 = 1 To NR
    Dim y As Integer
    For y = 1 To 6  'N3 Chord numbers 1 to 6
                                                                                        If H(N3, y) > 0 Then
                                                                                                    ac(N3, y) = Chord(psi, ar, br, Abs(H(N3, y)))
                                                                                                            Else
                                                                                                                        ac(N3, y) = Aell - Chord(psi, ar, br, Abs(H(N3, y)))
                                                                                                                                End If
                                                                                                                                    Next
                                                                                                                                    Next

                                                                                                                                    'Compute areas of sunlit and shaded soil and canopy visible to radiometer

Dim N4 As Integer
For N4 = 2 To (NR - 1)

    fc(N4, 1) = (ac(N4, 3) - ac(N4, 2)) 'Sunlit canopy
                                                                                                                                        'If fc(N4, 1) < 0 Then fc(N4, 1) = 0
    
    fc(N4, 2) = (ac(N4, 1) - ac(N4, 3)) + ac(N4, 5) - ac(N4, 2) 'Shaded canopy
                                                                                                                                            'If fc(N4, 2) < 0 Then fc(N4, 2) = 0
    
    fc(N4, 3) = (ac((N4 - 1), 6) - ac(N4, 4) + ac(N4, 6) - ac((N4 + 1), 4)) * 0.5 'Sunlit soil
                                                                                                                                                'If fc(N4, 3) < 0 Then fc(N4, 3) = 0
    
    fc(N4, 4) = ac(N4, 4) - ac(N4, 1) + ac(N4, 2) - ac(N4, 6) 'Shaded soil
                                                                                                                                                    'If fc(N4, 4) < 0 Then fc(N4, 4) = 0

Next

'Initialize fcr values
                                                                                                                                                    fcr(1) = 0
                                                                                                                                                    fcr(2) = 0
                                                                                                                                                    fcr(3) = 0
                                                                                                                                                    fcr(4) = 0

                                                                                                                                                    'Sum areas for each row
Dim N5 As Integer
For N5 = 2 To (NR - 1)

    fcr(1) = fc(N5, 1) + fcr(1)
    fcr(2) = fc(N5, 2) + fcr(2)
    fcr(3) = fc(N5, 3) + fcr(3)
    fcr(4) = fc(N5, 4) + fcr(4)

Next
'20 fcs = H(3, OPT)

                                                                                                                                                    fcs = (fcr(OPT)) / Aell

                                                                                                                                                    End Function

                                                                                                                                                    Function Heighte1(Pr As Double, theta As Double, rawpsi As Double, _
                                                                                                                                                                      hc As Double, wc As Double, Dv As Double, FOV As Double) As Double

                                                                                                                                                    'Heighte1 = Distance from center of footprint of radiometer to chord shadow
'   cast by near-edge of row crop canopy, where crop canopy is modelled as ELLIPSE (m)
                                                                                                                                                    'Pr = Perpendicular distance of radiometer from canopy row center (m)
'Theta = Zenith angle of radiometer (radians)
                                                                                                                                                    'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
'   where zero is looking parallel to crop row and pi/2 is looking perpendicular
                                                                                                                                                    '   to crop row (radians)
'hc = Canopy height (m)
                                                                                                                                                    'wc = canopy width (m)
'Dv = Vertical height of radiometer relative to soil (m)
                                                                                                                                                    'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")

'Assign value to pi
                                                                                                                                                    Dim Pi As Double
                                                                                                                                                    Pi = 3.14159265358979

                                                                                                                                                    'Constrain 45 < psi < 90
Dim psi As Double
If Abs(rawpsi) < 45 * Pi / 180 Then
    psi = Pi / 2 - (Abs(rawpsi))
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psi = (Abs(rawpsi)) - Pi / 2
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psi = Pi - Abs(rawpsi)
        Else
            psi = Abs(rawpsi)
        End If
    End If
End If

Dim ac As Double    'Horizonal axis of elliptical canopy (m)
                                                                                                                                                    Dim bc As Double    'Vertical axis of elliptical canopy (m)
Dim X1 As Double    'Horizonal distance from canopy ellipse origin to radiometer (m)
                                                                                                                                                    Dim Y1 As Double    'Vertical distance from canopy ellipse origin to radiometer (m)
ac = wc / 2 / Sin(psi)
bc = hc / 2
X1 = Pr / Sin(psi)
Y1 = Dv - bc

'Find tantheta1, the inverse slope of tangent line from radiometer to near edge of canopy
                                                                                                                                                    Dim tantheta1 As Double
                                                                                                                                                    tantheta1 = quartic1IRT(X1, Y1, ac, bc)

                                                                                                                                                    Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

Dim MIN As Double
If Abs(Tan(rawpsi)) < 1 Then
MIN = Tan(Pi / 2 - psi)
Else
MIN = 1
End If

Heighte1 = Dv * tantheta1 - OC * MIN

End Function
Function Heighte2(Pr As Double, theta As Double, rawpsi As Double, _
hc As Double, wc As Double, Dv As Double, FOV As Double) As Double

'Heighte2 = Distance from center of footprint of radiometer to chord shadow
                                                                                                                                                    '   cast by far-edge of row crop canopy, where crop canopy is modelled as ELLIPSE (m)
'Pr = Perpendicular distance of radiometer from canopy row center (m)
                                                                                                                                                    'Theta = Zenith angle of radiometer (radians)
'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
                                                                                                                                                    '   where zero is looking parallel to crop row and pi/2 is looking perpendicular
'   to crop row (radians)
                                                                                                                                                    'hc = Canopy height (m)
'wc = canopy width (m)
                                                                                                                                                    'Dv = Vertical height of radiometer relative to soil (m)
'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")

                                                                                                                                                    'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

'Constrain 45 < psi < 90
                                                                                                                                                    Dim psi As Double
                                                                                                                                                    If Abs(rawpsi) < 45 * Pi / 180 Then
                                                                                                                                                        psi = Pi / 2 - (Abs(rawpsi))
                                                                                                                                                        Else
                                                                                                                                                            If Abs(rawpsi) > 135 * Pi / 180 Then
                                                                                                                                                                    psi = (Abs(rawpsi)) - Pi / 2
                                                                                                                                                                        Else
                                                                                                                                                                                If Abs(rawpsi) > 90 * Pi / 180 Then
                                                                                                                                                                                            psi = Pi - Abs(rawpsi)
                                                                                                                                                                                                    Else
                                                                                                                                                                                                                psi = Abs(rawpsi)
                                                                                                                                                                                                                        End If
                                                                                                                                                                                                                            End If
                                                                                                                                                                                                                            End If

                                                                                                                                                                                                                            Dim ac As Double    'Horizonal axis of elliptical canopy (m)
Dim bc As Double    'Vertical axis of elliptical canopy (m)
                                                                                                                                                                                                                            Dim X1 As Double    'Horizonal distance from canopy ellipse origin to radiometer (m)
Dim Y1 As Double    'Vertical distance from canopy ellipse origin to radiometer (m)
                                                                                                                                                                                                                            ac = wc / 2 / Sin(psi)
                                                                                                                                                                                                                            bc = hc / 2
                                                                                                                                                                                                                            X1 = Pr / Sin(psi)
                                                                                                                                                                                                                            Y1 = Dv - bc

                                                                                                                                                                                                                            'Find tantheta2, the inverse slope of tangent line from radiometer to far edge of canopy
Dim tantheta2 As Double
tantheta2 = quartic2IRT(X1, Y1, ac, bc)

Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
                                                                                                                                                                                                                            OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

                                                                                                                                                                                                                            Dim MIN As Double
                                                                                                                                                                                                                            If Abs(Tan(rawpsi)) < 1 Then
                                                                                                                                                                                                                            MIN = Tan(Pi / 2 - psi)
                                                                                                                                                                                                                            Else
                                                                                                                                                                                                                            MIN = 1
                                                                                                                                                                                                                            End If

                                                                                                                                                                                                                            Heighte2 = Dv * tantheta2 - OC * MIN

                                                                                                                                                                                                                            End Function
                                                                                                                                                                                                                            Function Heighte3(Pr As Double, row As Double, theta As Double, rawpsi As Double, _
                                                                                                                                                                                                                                              hc As Double, wc As Double, Dv As Double, FOV As Double, _
                                                                                                                                                                                                                                              thetas As Double, psis As Double) As Double

                                                                                                                                                                                                                            'Heighte3 = Distance from center of footprint of radiometer to chord projected by
'sunlit-shaded boundary on near-side of canopy, where crop canopy is modelled as ELLIPSE (m)
                                                                                                                                                                                                                            'Pr = Perpendicular distance of radiometer from canopy row center (m)
'row = Crop row spacing (m)
                                                                                                                                                                                                                            'Theta = Zenith angle of radiometer (radians)
'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
                                                                                                                                                                                                                            '   where zero is looking parallel to crop row and pi/2 is looking perpendicular
'   to crop row (radians)
                                                                                                                                                                                                                            'hc = Canopy height (m)
'wc = canopy width (m)
                                                                                                                                                                                                                            'Dv = Vertical height of radiometer relative to soil (m)
'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
                                                                                                                                                                                                                            'thetas = Solar zenith angle (radians)
'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
                                                                                                                                                                                                                            'crop row and pi/2 is looking perpendicular to crop row (radians)

'Assign value to pi
                                                                                                                                                                                                                            Dim Pi As Double
                                                                                                                                                                                                                            Pi = 3.14159265358979

                                                                                                                                                                                                                            'Constrain 45 < psi < 90
Dim psi As Double
If Abs(rawpsi) < 45 * Pi / 180 Then
    psi = Pi / 2 - (Abs(rawpsi))
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psi = (Abs(rawpsi)) - Pi / 2
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psi = Pi - Abs(rawpsi)
        Else
            psi = Abs(rawpsi)
        End If
    End If
End If

Dim bc As Double    'Vertical axis of elliptical canopy (m)
                                                                                                                                                                                                                            Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
Dim Xs As Double    'Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
                                                                                                                                                                                                                            Dim Ys As Double    'Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
Dim X3 As Double    'Horizontal distance from radiometer to ground-projected
                                                                                                                                                                                                                            'sunlit-shaded boundary on canopy (m)
bc = hc / 2
thetasp = -thetas * Sin(psis)
Xs = wc / 2 / Sqr(1 + (bc ^ 2) / ((wc / 2) ^ 2) * ((Tan(thetasp)) ^ 2))    'Xs is positive
                                                                                                                                                                                                                            Ys = -(bc ^ 2) / ((wc / 2) ^ 2) * Xs * Tan(thetasp) 'Ys is positive or negative

'Determine critical perpendicular solar zenith angle,
                                                                                                                                                                                                                            'beyond which results in adjacent row shading
Dim Xscr As Double  'Horizontal distance from canopy ellipse origin to tangent of sunray
                                                                                                                                                                                                                            'along thetasp (m)
Dim Yscr As Double  'Vertical distance from canopy ellipse origin to tangent of sunray
                                                                                                                                                                                                                            'along thetasp (m)
Dim thetaspcr As Double     'Critical perpendicular solar zenith angle

                                                                                                                                                                                                                            Xscr = 2 * ((wc / 2) ^ 2) / row     'Xscr is positive
Yscr = -Sqr(((bc ^ 2) * Xscr * (row - 2 * Xscr)) / 2 / ((wc / 2) ^ 2))  'Yscr is negative
                                                                                                                                                                                                                            thetaspcr = Atn(-((wc / 2) ^ 2) * Yscr / (bc ^ 2) / Xscr)   'thetascr is positive

If thetasp > thetaspcr Then     'Shadows cast by adjacent rows and H3 is raised

                                                                                                                                                                                                                                Dim m3 As Double
                                                                                                                                                                                                                                    Dim b3 As Double
                                                                                                                                                                                                                                        Dim AA As Double
                                                                                                                                                                                                                                            Dim BB As Double
                                                                                                                                                                                                                                                Dim CC As Double
                                                                                                                                                                                                                                                    Dim Xs3 As Double
                                                                                                                                                                                                                                                        Dim Ys3 As Double

                                                                                                                                                                                                                                                            m3 = 1 / Tan(thetasp)
                                                                                                                                                                                                                                                                b3 = -Ys - m3 * (row - Xs)
                                                                                                                                                                                                                                                                    AA = (bc ^ 2) + ((wc / 2) ^ 2) * (m3 ^ 2)
                                                                                                                                                                                                                                                                        BB = 2 * m3 * b3 * ((wc / 2) ^ 2)
                                                                                                                                                                                                                                                                            CC = ((wc / 2) ^ 2) * (b3 ^ 2) - ((wc / 2) ^ 2) * (bc ^ 2)
                                                                                                                                                                                                                                                                                Xs3 = (-BB + Sqr((BB ^ 2) - 4 * AA * CC)) / (2 * AA)    'Positive root (negative root taken for H5)
    Ys3 = m3 * Xs3 + b3
    X3 = Dv / Sin(psi) * ((Pr - Xs3) / (Dv - bc - Ys3))

Else    'Compute X3 as normal (no shading by adjacent row)

                                                                                                                                                                                                                                                                                    X3 = Dv / Sin(psi) * ((Pr - Xs) / (Dv - bc - Ys))

                                                                                                                                                                                                                                                                                    End If

                                                                                                                                                                                                                                                                                    Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

Dim MIN As Double
If Abs(Tan(rawpsi)) < 1 Then
MIN = Tan(Pi / 2 - psi)
Else
MIN = 1
End If

Heighte3 = X3 - OC * MIN

End Function

Function Heighte4(Pr As Double, row As Double, theta As Double, rawpsi As Double, _
hc As Double, wc As Double, Dv As Double, FOV As Double, _
thetas As Double, psis As Double) As Double

'Heighte4 = Distance from center of footprint of radiometer to chord projected by
                                                                                                                                                                                                                                                                                    'sunlit-shaded soil boundary on near-side of canopy, where crop canopy is modelled as ELLIPSE (m)
'Pr = Perpendicular distance of radiometer from canopy row center (m)
                                                                                                                                                                                                                                                                                    'row = Crop row spacing (m)
'Theta = Zenith angle of radiometer (radians)
                                                                                                                                                                                                                                                                                    'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
'   where zero is looking parallel to crop row and pi/2 is looking perpendicular
                                                                                                                                                                                                                                                                                    '   to crop row (radians)
'hc = Canopy height (m)
                                                                                                                                                                                                                                                                                    'wc = canopy width (m)
'Dv = Vertical height of radiometer relative to soil (m)
                                                                                                                                                                                                                                                                                    'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
'thetas = Solar zenith angle (radians)
                                                                                                                                                                                                                                                                                    'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
'crop row and pi/2 is looking perpendicular to crop row (radians)

                                                                                                                                                                                                                                                                                    'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

'Constrain 45 < psi < 90
                                                                                                                                                                                                                                                                                    Dim psi As Double
                                                                                                                                                                                                                                                                                    If Abs(rawpsi) < 45 * Pi / 180 Then
                                                                                                                                                                                                                                                                                        psi = Pi / 2 - (Abs(rawpsi))
                                                                                                                                                                                                                                                                                        Else
                                                                                                                                                                                                                                                                                            If Abs(rawpsi) > 135 * Pi / 180 Then
                                                                                                                                                                                                                                                                                                    psi = (Abs(rawpsi)) - Pi / 2
                                                                                                                                                                                                                                                                                                        Else
                                                                                                                                                                                                                                                                                                                If Abs(rawpsi) > 90 * Pi / 180 Then
                                                                                                                                                                                                                                                                                                                            psi = Pi - Abs(rawpsi)
                                                                                                                                                                                                                                                                                                                                    Else
                                                                                                                                                                                                                                                                                                                                                psi = Abs(rawpsi)
                                                                                                                                                                                                                                                                                                                                                        End If
                                                                                                                                                                                                                                                                                                                                                            End If
                                                                                                                                                                                                                                                                                                                                                            End If

                                                                                                                                                                                                                                                                                                                                                            Dim bc As Double    'Vertical axis of elliptical canopy (m)
Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
                                                                                                                                                                                                                                                                                                                                                            Dim Xs As Double    'Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
Dim Ys As Double    'Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
                                                                                                                                                                                                                                                                                                                                                            Dim X4 As Double    'Horizontal distance from radiometer to ground-projected
'sunlit-shaded soil boundary on near-side of canopy (m)
                                                                                                                                                                                                                                                                                                                                                            bc = hc / 2
                                                                                                                                                                                                                                                                                                                                                            thetasp = -thetas * Sin(psis)
                                                                                                                                                                                                                                                                                                                                                            Xs = wc / 2 / Sqr(1 + (bc ^ 2) / ((wc / 2) ^ 2) * ((Tan(thetasp)) ^ 2))
                                                                                                                                                                                                                                                                                                                                                            Ys = -(bc ^ 2) / ((wc / 2) ^ 2) * Xs * Tan(thetasp)
                                                                                                                                                                                                                                                                                                                                                            X4 = (Pr - Xs + (Tan(thetasp)) * (bc + Ys)) / Sin(psi)

                                                                                                                                                                                                                                                                                                                                                            Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

Dim MIN As Double
If Abs(Tan(rawpsi)) < 1 Then
MIN = Tan(Pi / 2 - psi)
Else
MIN = 1
End If

Heighte4 = X4 - OC * MIN

End Function
Function Heighte5(Pr As Double, row As Double, theta As Double, rawpsi As Double, _
hc As Double, wc As Double, Dv As Double, FOV As Double, _
thetas As Double, psis As Double) As Double

'Heighte5 = Distance from center of footprint of radiometer to chord projected by
                                                                                                                                                                                                                                                                                                                                                            'sunlit-shaded boundary on far-side of canopy, where crop canopy is modelled as ELLIPSE (m)
'Pr = Perpendicular distance of radiometer from canopy row center (m)
                                                                                                                                                                                                                                                                                                                                                            'row = Crop row spacing (m)
'Theta = Zenith angle of radiometer (radians)
                                                                                                                                                                                                                                                                                                                                                            'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
'   where zero is looking parallel to crop row and pi/2 is looking perpendicular
                                                                                                                                                                                                                                                                                                                                                            '   to crop row (radians)
'hc = Canopy height (m)
                                                                                                                                                                                                                                                                                                                                                            'wc = canopy width (m)
'Dv = Vertical height of radiometer relative to soil (m)
                                                                                                                                                                                                                                                                                                                                                            'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
'thetas = Solar zenith angle (radians)
                                                                                                                                                                                                                                                                                                                                                            'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
'crop row and pi/2 is looking perpendicular to crop row (radians)

                                                                                                                                                                                                                                                                                                                                                            'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

'Constrain 45 < psi < 90
                                                                                                                                                                                                                                                                                                                                                            Dim psi As Double
                                                                                                                                                                                                                                                                                                                                                            If Abs(rawpsi) < 45 * Pi / 180 Then
                                                                                                                                                                                                                                                                                                                                                                psi = Pi / 2 - (Abs(rawpsi))
                                                                                                                                                                                                                                                                                                                                                                Else
                                                                                                                                                                                                                                                                                                                                                                    If Abs(rawpsi) > 135 * Pi / 180 Then
                                                                                                                                                                                                                                                                                                                                                                            psi = (Abs(rawpsi)) - Pi / 2
                                                                                                                                                                                                                                                                                                                                                                                Else
                                                                                                                                                                                                                                                                                                                                                                                        If Abs(rawpsi) > 90 * Pi / 180 Then
                                                                                                                                                                                                                                                                                                                                                                                                    psi = Pi - Abs(rawpsi)
                                                                                                                                                                                                                                                                                                                                                                                                            Else
                                                                                                                                                                                                                                                                                                                                                                                                                        psi = Abs(rawpsi)
                                                                                                                                                                                                                                                                                                                                                                                                                                End If
                                                                                                                                                                                                                                                                                                                                                                                                                                    End If
                                                                                                                                                                                                                                                                                                                                                                                                                                    End If

                                                                                                                                                                                                                                                                                                                                                                                                                                    Dim bc As Double    'Vertical axis of elliptical canopy (m)
Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
                                                                                                                                                                                                                                                                                                                                                                                                                                    Dim Xs As Double    'Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
Dim Ys As Double    'Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                    Dim X5 As Double    'Horizontal distance from radiometer to ground-projected
'sunlit-shaded boundary on canopy (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                    bc = hc / 2
                                                                                                                                                                                                                                                                                                                                                                                                                                    thetasp = -thetas * Sin(psis)
                                                                                                                                                                                                                                                                                                                                                                                                                                    Xs = -wc / 2 / Sqr(1 + (bc ^ 2) / ((wc / 2) ^ 2) * ((Tan(thetasp)) ^ 2)) 'Xs is negative (positive for H3)
Ys = -(bc ^ 2) / ((wc / 2) ^ 2) * Xs * Tan(thetasp) 'Ys is positive or negative

                                                                                                                                                                                                                                                                                                                                                                                                                                    'Determine critical perpendicular solar zenith angle,
'beyond which results in adjacent row shading
                                                                                                                                                                                                                                                                                                                                                                                                                                    Dim Xscr As Double  'Horizontal distance from canopy ellipse origin to tangent of sunray
'along thetasp (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                    Dim Yscr As Double  'Vertical distance from canopy ellipse origin to tangent of sunray
'along thetasp (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                    Dim thetaspcr As Double     'Critical perpendicular solar zenith angle

Xscr = -2 * ((wc / 2) ^ 2) / row     'Xscr is negative (positive for H3)
                                                                                                                                                                                                                                                                                                                                                                                                                                    Yscr = -Sqr(((bc ^ 2) * Xscr * (row + 2 * Xscr)) / -2 / ((wc / 2) ^ 2))  'Yscr is negative; note -2 and row+2*Xscr
thetaspcr = Atn(-((wc / 2) ^ 2) * Yscr / (bc ^ 2) / Xscr)   'thetascr is negative

                                                                                                                                                                                                                                                                                                                                                                                                                                    If thetasp < thetaspcr Then     'Shadows cast by adjacent rows and H5 is raised.
'NOTE: thetascr is negative (thetascr for H3 is positive)

                                                                                                                                                                                                                                                                                                                                                                                                                                        Dim m5 As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                            Dim b5 As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim AA As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                    Dim BB As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                        Dim CC As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                            Dim Xs5 As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim Ys5 As Double

                                                                                                                                                                                                                                                                                                                                                                                                                                                                    m5 = 1 / Tan(thetasp)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        b5 = -Ys + m5 * (row + Xs) 'NOTE: signs are negative for H3
    AA = (bc ^ 2) + ((wc / 2) ^ 2) * (m5 ^ 2)
    BB = 2 * m5 * b5 * ((wc / 2) ^ 2)
    CC = ((wc / 2) ^ 2) * (b5 ^ 2) - ((wc / 2) ^ 2) * (bc ^ 2)
    Xs5 = (-BB - Sqr((BB ^ 2) - 4 * AA * CC)) / (2 * AA)    'NOTE: Negative root (H3 positive)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Ys5 = m5 * Xs5 + b5
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                X5 = Dv / Sin(psi) * ((Pr - Xs5) / (Dv - bc - Ys5))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Else    'Compute X5 as normal (no shading by adjacent row)
    
    X5 = Dv / Sin(psi) * ((Pr - Xs) / (Dv - bc - Ys))

End If

Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim MIN As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If Abs(Tan(rawpsi)) < 1 Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                MIN = Tan(Pi / 2 - psi)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                MIN = 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Heighte5 = X5 - OC * MIN

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End Function
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Function Heighte6(Pr As Double, row As Double, theta As Double, rawpsi As Double, _
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  hc As Double, wc As Double, Dv As Double, FOV As Double, _
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  thetas As Double, psis As Double) As Double

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'Heighte46 = Distance from center of footprint of radiometer to chord projected by
'sunlit-shaded soil boundary on far-side of canopy, where crop canopy is modelled as ELLIPSE (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'Pr = Perpendicular distance of radiometer from canopy row center (m)
'row = Crop row spacing (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'Theta = Zenith angle of radiometer (radians)
'Rawpsi = Azimuth angel of crop row relative to radiometer view angle,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                '   where zero is looking parallel to crop row and pi/2 is looking perpendicular
'   to crop row (radians)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'hc = Canopy height (m)
'wc = canopy width (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'Dv = Vertical height of radiometer relative to soil (m)
'FOV = Field of view number of radiometer (e.g., 2:1 FOV would be specified as "2")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'thetas = Solar zenith angle (radians)
'psis = Solar azimuth angle relative to crop row, where zero is looking parallel to
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'crop row and pi/2 is looking perpendicular to crop row (radians)

'Assign value to pi
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim Pi As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Pi = 3.14159265358979

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'Constrain 45 < psi < 90
Dim psi As Double
If Abs(rawpsi) < 45 * Pi / 180 Then
    psi = Pi / 2 - (Abs(rawpsi))
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psi = (Abs(rawpsi)) - Pi / 2
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psi = Pi - Abs(rawpsi)
        Else
            psi = Abs(rawpsi)
        End If
    End If
End If

Dim bc As Double    'Vertical axis of elliptical canopy (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim thetasp As Double  'Solar zenith angle perpendicular to canopy (rad)
Dim Xs As Double    'Horizontal distance from canopy ellipse origin to tangent of sunray along thetasp (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim Ys As Double    'Vertical distance from canopy ellipse origin to tangent of sunray along thetasp (m)
Dim X6 As Double    'Horizontal distance from radiometer to ground-projected
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'sunlit-shaded soil boundary on near-side of canopy (m)
bc = hc / 2
thetasp = -thetas * Sin(psis)
Xs = -wc / 2 / Sqr(1 + (bc ^ 2) / ((wc / 2) ^ 2) * ((Tan(thetasp)) ^ 2))    'NOTE: negative sign (H4 positive)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Ys = -(bc ^ 2) / ((wc / 2) ^ 2) * Xs * Tan(thetasp)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                X6 = (Pr - Xs + (Tan(thetasp)) * (bc + Ys)) / Sin(psi)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim OC As Double    'Horizontal distance from radiometer to center of radiometer footprint (m)
OC = Dv / 2 * (Tan(theta + Atn(1 / 2 / FOV)) + Tan(theta - Atn(1 / 2 / FOV)))

Dim MIN As Double
If Abs(Tan(rawpsi)) < 1 Then
MIN = Tan(Pi / 2 - psi)
Else
MIN = 1
End If

Heighte6 = X6 - OC * MIN

End Function

Function quartic1IRT(X1 As Double, Y1 As Double, ac As Double, bc As Double) As Double
'Compute the 1st root of a quartic equation of form
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'Ax^4 + Bx^3 + Cx^2 + Dx + E = 0

Dim A As Double
Dim b As Double
Dim c As Double
Dim d As Double
Dim E As Double
Dim alpha As Double
Dim beta As Double
Dim gamma As Double
Dim P As Double
Dim Q As Double
Dim r As Double
Dim u As Double
Dim y As Double
Dim W As Double

'Compute coefficients of quartic
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                A = (Y1 ^ 2) * (bc ^ 2) - (bc ^ 4)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                b = -2 * X1 * Y1 * (bc ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                c = (Y1 ^ 2) * (ac ^ 2) + (X1 ^ 2) * (bc ^ 2) - 2 * (ac ^ 2) * (bc ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                d = -2 * X1 * Y1 * (ac ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                E = (X1 ^ 2) * (ac ^ 2) - (ac ^ 4)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If Abs(X1) < 0.001 Then     'Assume quadratic equation
quartic1IRT = -Sqr((-c + Sqr(c ^ 2 - 4 * A * E)) / (2 * A))
GoTo 20
End If

'Compute quartic
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                alpha = -((3 * b ^ 2) / (8 * A ^ 2)) + c / A
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                beta = (b ^ 3) / (8 * A ^ 3) - (b * c) / (2 * A ^ 2) + d / A
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                gamma = -(3 * b ^ 4) / (256 * A ^ 4) + (c * b ^ 2) / (16 * A ^ 3) - (b * d) / (4 * A ^ 2) + E / A
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                P = -((alpha ^ 2) / 12) - gamma
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Q = -(alpha ^ 3) / (108) + (alpha * gamma / 3) - (beta ^ 2) / 8
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                r = Q / 2 + Sqr((Q ^ 2) / 4 + (P ^ 3) / 27)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If r < 0 Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                u = -((Abs(r)) ^ (1 / 3))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                u = r ^ (1 / 3)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim Uy As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    If u = 0 Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Uy = 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Uy = P / (3 * u)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            End If
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            y = -5 / 6 * alpha - u + Uy
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            W = Sqr(Abs(alpha + 2 * y))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Dim ZZ As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            If W = 0 Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ZZ = Abs(3 * alpha + 2 * y)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    quartic1IRT = -(b / (4 * A)) + (W - Sqr(ZZ)) / 2 - y '(y subtracted to smooth curve)
    GoTo 20
Else
    ZZ = -(3 * alpha + 2 * y + 2 * beta / W)
    If ZZ < 0 Then
        ZZ = Abs(3 * alpha + 2 * y - 2 * beta / W)
        quartic1IRT = -(b / (4 * A)) + (-W - Sqr(ZZ)) / 2
    Else
        quartic1IRT = -(b / (4 * A)) + (W - Sqr(ZZ)) / 2
    End If
End If

20 End Function

Function quartic2IRT(X1 As Double, Y1 As Double, ac As Double, bc As Double) As Double
'Compute the 2nd root of a quartic equation of form
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    'Ax^4 + Bx^3 + Cx^2 + Dx + E = 0

Dim A As Double
Dim b As Double
Dim c As Double
Dim d As Double
Dim E As Double
Dim alpha As Double
Dim beta As Double
Dim gamma As Double
Dim P As Double
Dim Q As Double
Dim r As Double
Dim u As Double
Dim y As Double
Dim W As Double

'Compute coefficients of quartic
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    A = (Y1 ^ 2) * (bc ^ 2) - (bc ^ 4)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    b = -2 * X1 * Y1 * (bc ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    c = (Y1 ^ 2) * (ac ^ 2) + (X1 ^ 2) * (bc ^ 2) - 2 * (ac ^ 2) * (bc ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    d = -2 * X1 * Y1 * (ac ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    E = (X1 ^ 2) * (ac ^ 2) - (ac ^ 4)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    If Abs(X1) < 0.001 Then     'Assume quadratic equation
quartic2IRT = Sqr((-c + Sqr(c ^ 2 - 4 * A * E)) / (2 * A))
GoTo 20
End If

'Compute quartic
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    alpha = -((3 * b ^ 2) / (8 * A ^ 2)) + c / A
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    beta = (b ^ 3) / (8 * A ^ 3) - (b * c) / (2 * A ^ 2) + d / A
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    gamma = -(3 * b ^ 4) / (256 * A ^ 4) + (c * b ^ 2) / (16 * A ^ 3) - (b * d) / (4 * A ^ 2) + E / A
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    P = -((alpha ^ 2) / 12) - gamma
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Q = -(alpha ^ 3) / (108) + (alpha * gamma / 3) - (beta ^ 2) / 8
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    r = Q / 2 + Sqr((Q ^ 2) / 4 + (P ^ 3) / 27)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    If r < 0 Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    u = -((Abs(r)) ^ (1 / 3))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    u = r ^ (1 / 3)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    End If

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Dim Uy As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        If u = 0 Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Uy = 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Uy = P / (3 * u)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                y = -5 / 6 * alpha - u + Uy
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                W = Sqr(Abs(alpha + 2 * y))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim ZZ As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If W = 0 Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ZZ = Abs(3 * alpha + 2 * y)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        quartic2IRT = -(b / (4 * A)) + (-W + Sqr(ZZ)) / 2 - y '(y subtracted to smooth curve)
    GoTo 20
Else
    ZZ = -(3 * alpha + 2 * y + 2 * beta / W)
    If ZZ < 0 Then
        ZZ = Abs(3 * alpha + 2 * y - 2 * beta / W)
        quartic2IRT = -(b / (4 * A)) + (-W + Sqr(ZZ)) / 2
    Else
        quartic2IRT = -(b / (4 * A)) + (W + Sqr(ZZ)) / 2
    End If
End If

20 End Function
'Function chord to compute the chord area of an ellipse
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Function Chord(rawpsi As Double, MAJOR As Double, MINOR As Double, H As Double) As Double

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'Assign value to pi
Dim Pi As Double
Pi = 3.14159265358979

Dim X As Double     'Axis used to compute chord area where H crosses (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Dim y As Double     'Axis perpendicular to X (m)
Dim psi As Double   'Angle of chord wrt Y-axis, constrained to 45 < psi < 90 (rad)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'Constrain 45 < psi < 90, specify X and Y
If Abs(rawpsi) < 45 * Pi / 180 Then
    psi = Pi / 2 - (Abs(rawpsi))
    X = MINOR
    y = MAJOR
Else
    If Abs(rawpsi) > 135 * Pi / 180 Then
        psi = (Abs(rawpsi)) - Pi / 2
        X = MINOR
        y = MAJOR
    Else
        If Abs(rawpsi) > 90 * Pi / 180 Then
            psi = Pi - Abs(rawpsi)
            X = MAJOR
            y = MINOR
        Else
            psi = Abs(rawpsi)
            X = MAJOR
            y = MINOR
        End If
    End If
End If

'Compute distance from ellipse center to tangent (maximum possible H to form a chord)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'Dim ec As Double 'Ellipse eccentricity (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'Dim asin1 As Double 'Parameter used to compute inverse sine
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'Dim alpha1 As Double    'Internal angle (rad)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'Dim HH1 As Double    'Radius of internal construction circle (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'Dim T1 As Double    'Distance from ellipse center to tangent (m)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'ec = Sqr(1 - (X ^ 2) / (y ^ 2))
'asin1 = ec * Sin(psi + Pi / 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'alpha1 = Atn(asin1 / Sqr(-asin1 * asin1 + 1))
'HH1 = y * (Sin(Pi / 2 - psi - alpha1)) / (Sin(psi + Pi / 2))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'T1 = ec * y + HH1 / Sin(psi)

'Compute T1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Dim T1 As Double    'Distance from radiometer ellipse footprint center to line
'   tangent to radiometer ellipse for given azimuth and zenith angles (m).
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        If (Abs(psi - 90 * Pi / 180) < 0.01) Then   '90 degree radiometer azimuth
    T1 = y
    If H > T1 Then  'Chord outside of Ellipse
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Chord = 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        GoTo 20
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Else            'Case 2 with equilateral triangle
        Chord = X * y * Atn((Sqr(y ^ 2 - H ^ 2)) / H) - H * X / y * Sqr(y ^ 2 - H ^ 2)
        GoTo 20
    End If
Else
    If Abs(psi) < 0.01 Then     'ZERO degree radiometer azimuth
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    T1 = X
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            If H > T1 Then  'Chord outside of Ellipse
            Chord = 0
            GoTo 20
        Else            'Case 4 with equilateral triangle
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Chord = X * y * Atn((Sqr(X ^ 2 - H ^ 2)) / H) - H * y / X * Sqr(X ^ 2 - H ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    GoTo 20
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            End If
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Else                        'Radiometer azimuth between ZERO and 90 degrees
        Dim xtr1 As Double
        Dim ytr1 As Double
        xtr1 = y / Sqr(1 + (X ^ 2) / (y ^ 2) / ((Tan(psi)) ^ 2))
        ytr1 = (X ^ 2) / (y ^ 2) * xtr1 / Tan(psi)
        T1 = xtr1 + ytr1 / Tan(psi)
    End If
End If

'Determine if H cuts a chord through ellipse
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If H > T1 Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Chord = 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                GoTo 20
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If Abs(X - y) < 0.001 Then 'Ellipse is a circle
Dim hc As Double
Dim yy As Double
Dim Acosyy As Double
hc = H * Sin(psi)
yy = hc / X
Acosyy = Atn(-yy / Sqr(-yy * yy + 1)) + 2 * Atn(1)
Chord = (X ^ 2) * Acosyy - hc * Sqr(X ^ 2 - hc ^ 2)
GoTo 20
Else
GoTo 10
End If

'Chord = Chord area (m2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'psi = Azimuth angle (radians)
'Y = Major axis (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'X = Minor axis (m)
'H = Distance from ellipse center to chord along major axis Y (m)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'C = Case number, where
'CASE 1: X > H*tan(psi); H < Y
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'CASE 2: X < H*tan(psi); H < Y
'CASE 3: X < H*tan(psi); H > Y
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'CASE 4: X > H*tan(psi); H > Y

'Determine chord case
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                10 Dim CC As Integer
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If H < y Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    If X > H * Tan(psi) Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            CC = 1#
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            CC = 2#
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    If X > H * Tan(psi) Then
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            CC = 4#
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            CC = 3#
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'See diagrams for variables used below
Dim aq As Double    'Used in quadratic equation for q
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim bq As Double    'Used in quadratic equation for q
Dim cq As Double    'Used in quadratic equation for q
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim Q As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim s As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim alpha As Double
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim A As Double

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim at As Double    'Used in quadratic equation for t
Dim bt As Double    'Used in quadratic equation for t
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim ct As Double    'Used in quadratic equation for t
Dim t As Double
Dim u As Double
Dim beta As Double
Dim b As Double

'Declare sector and triangle variables, where Chord = Aes - Aet
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Dim Aes As Double   'Area enclosed by sector (m2)
Dim Aet As Double   'Area enclosed by triangle (m2)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If CC = 1 Then

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                aq = (y ^ 2) / (X ^ 2) + (1 / Tan(psi)) ^ 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                bq = 2 * H / Tan(psi)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                cq = (H ^ 2) - (y ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Q = (-bq + Sqr((bq ^ 2) - 4 * aq * cq)) / (2 * aq)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                s = Q / Tan(psi)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                alpha = Atn(Q / (H + s))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                A = Sqr((Q ^ 2) + ((H + s) ^ 2))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                at = (X ^ 2) / (y ^ 2) + (Tan(psi)) ^ 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                bt = 2 * H * ((Tan(psi)) ^ 2)   'negative for Cases 2 & 3
ct = (H ^ 2) * ((Tan(psi)) ^ 2) - (X ^ 2)
t = (-bt + Sqr((bt ^ 2) - 4 * at * ct)) / (2 * at)
u = (H + t) * Tan(psi)  '-t for Cases 2 & 3
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                beta = Atn(t / u)       'Inverse for Cases 2 & 3
b = Sqr((t ^ 2) + (u ^ 2))

Aes = X * y / 2 * (Atn(y / X * Tan(alpha)) + Atn(X / y * Tan(beta)) + Pi / 2)
Aet = A * b / 2 * Sin(alpha + beta + Pi / 2)
Chord = Aes - Aet
GoTo 20
End If

If CC = 2 Then

aq = (y ^ 2) / (X ^ 2) + (1 / Tan(psi)) ^ 2
bq = 2 * H / Tan(psi)
cq = (H ^ 2) - (y ^ 2)
Q = (-bq + Sqr((bq ^ 2) - 4 * aq * cq)) / (2 * aq)
s = Q / Tan(psi)
alpha = Atn(Q / (H + s))
A = Sqr((Q ^ 2) + ((H + s) ^ 2))

at = (X ^ 2) / (y ^ 2) + (Tan(psi)) ^ 2
bt = -2 * H * ((Tan(psi)) ^ 2)
ct = (H ^ 2) * ((Tan(psi)) ^ 2) - (X ^ 2)
t = (-bt - Sqr((bt ^ 2) - 4 * at * ct)) / (2 * at)
u = (H - t) * Tan(psi)  '+t for case 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                beta = Atn(u / t)       'Inverse for case 1
b = Sqr((t ^ 2) + (u ^ 2))

Aes = X * y / 2 * (Atn(y / X * Tan(alpha)) + Atn(y / X * Tan(beta)))
Aet = A * b / 2 * Sin(alpha + beta)
Chord = Aes - Aet
GoTo 20
End If

If CC = 3 Then

aq = (y ^ 2) / (X ^ 2) + (1 / Tan(psi)) ^ 2 'tan for Cases 1 & 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                bq = -2 * H / Tan(psi)                  'positive and tan for Cases 1 & 2
cq = (H ^ 2) - (y ^ 2)
Q = (-bq - Sqr((bq ^ 2) - 4 * aq * cq)) / (2 * aq)
s = Q / Tan(psi)            'tan for Cases 1 & 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                alpha = Atn(Q / (H - s))    '+s for Cases 1 & 2
A = Sqr((Q ^ 2) + ((H - s) ^ 2))   '+s for Cases 1 & 2

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                at = (X ^ 2) / (y ^ 2) + (Tan(psi)) ^ 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                bt = -2 * H * ((Tan(psi)) ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ct = (H ^ 2) * ((Tan(psi)) ^ 2) - (X ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                t = (-bt - Sqr((bt ^ 2) - 4 * at * ct)) / (2 * at)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                u = (H - t) * Tan(psi)  '+t for case 1
beta = Atn(u / t)       'Inverse for case 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                b = Sqr((t ^ 2) + (u ^ 2))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Aes = X * y / 2 * (Atn(y / X * Tan(beta)) - Atn(y / X * Tan(alpha)))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Aet = A * b / 2 * Sin(beta - alpha)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Chord = Aes - Aet
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                GoTo 20
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                If CC = 4 Then

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                aq = (y ^ 2) / (X ^ 2) + (1 / Tan(psi)) ^ 2 'tan for Cases 1 & 2
bq = -2 * H / Tan(psi)                  'positive and tan for Cases 1 & 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                cq = (H ^ 2) - (y ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Q = (-bq - Sqr((bq ^ 2) - 4 * aq * cq)) / (2 * aq)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                s = Q / Tan(psi)            'tan for Cases 1 & 2
alpha = Atn(Q / (H - s))    '+s for Cases 1 & 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                A = Sqr((Q ^ 2) + ((H - s) ^ 2))   '+s for Cases 1 & 2

at = (X ^ 2) / (y ^ 2) + (Tan(psi)) ^ 2
bt = 2 * H * ((Tan(psi)) ^ 2)   'negative for Cases 2 & 3
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ct = (H ^ 2) * ((Tan(psi)) ^ 2) - (X ^ 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                t = (-bt + Sqr((bt ^ 2) - 4 * at * ct)) / (2 * at)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                u = (H + t) * Tan(psi)  '-t for Cases 2 & 3
beta = Atn(t / u)       'Inverse for Cases 2 & 3
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                b = Sqr((t ^ 2) + (u ^ 2))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Aes = X * y / 2 * (Atn(X / y * Tan(beta)) + Pi / 2 - Atn(y / X * Tan(alpha)))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Aet = A * b / 2 * Sin(-alpha + beta + Pi / 2)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Chord = Aes - Aet
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                GoTo 20
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                End If

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                20 End Function


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
