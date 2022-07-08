from scipy import constants
import os
from pandas import read_csv, DataFrame
import logging
logging.basicConfig(level=logging.INFO)
home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Inputs')

#physical constants
pi = constants.pi
g = constants.g #m/s2
boltz = constants.Stefan_Boltzmann# W/K4/m2
vonk = 0.4 #Von Karman
Tk = 273.16 #0 C
rho_a = 1.205 #kg/m3
Gsc = 1367 #W/m2
lam = 2450 #kJ/kg
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# heat capacity of mixed air at constant pressure (J kg-1 K-1)
c_p = 1013

# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# gas constant for dry air, J/(kg*degK)
R_d = 287.04

class inputs:
    def __init__(self):
        self.read_params()
        self.read_can()
        self.read_met()
        return None

    def read_params(self):
        for f in os.listdir(p):
            if 'parameters' in f and f[-4:]=='.csv':
                parameters = read_csv(os.path.join(p,f))
                #defaults
                #user adjustable parameters
                self.Cx = 90#Empirical constant  s m-1
                self.b = 0.012#Empirical constant = 0.012 or b = 0.012[1 + cos(psiw)],
                #       where psiw is wind direction, ccw from north
                self.c = 0.0025#Empirical constant = 0.0025        
                #Monin-Obukhov parameters
                self.dhc = .67#displacement height/canopy height C&N98, K95 say 0.65
                self.zomhc = 0.125#momemtun roughness over canopy height K95
                self.zohzomhc = 1#heat roughness over momentum roughness C&N say .2,
                # K95 say exp(-2),
                # K99 set equal for series model
                #bare soil roughness
                self.hs = .4e-3 #m
                self.zombs = 1 #m
                self.zohzombs = np.exp(-4.5) #from Stewart paper
                #Stewart, J. B., W. P. Kustas, K. S. Humes, W. D. Nichols, M. S.
                #Moran, and H. A. R. de Bruin, 1994: Sensible heat fluxradiometric
                #surface temperature relationship for eight semiarid
                #areas. J. Appl. Meteor., 33, 1110–1117.
                #soil heat flux
                self.GRnDay = -0.1
                self.GRnNight = -0.5
                self.aG = -0.31
                #canopy stomatal resistance s/m
                self.rcDay = 50
                self.rcNight = 200
                #soil albedo vis/nir/wet/dry
                self.rhosNir = 0.15
                self.rhosNis = 0.25
                #soil emissivity
                self.emisSoil = 0.98
                #residue albedo
                self.rhorNir = 0.15 
                self.rhorVis = 0.25
                #residue emissivity
                self.emisRes = 0.98
                #leaf absorbance/emittance/reflectance
                self.emisVeg = 0.98
                self.zetaVis = 0.85#Leaf shortwave absorption in the visible (PAR) spectra (no units)
                self.zetaNir = 0.2#Leaf shortwave absorption in the near infrared spectra (no units)
                #leaf angle param
                self.XE = 1.0
                #canopy lw extinction
                self.kappaIR = 0.95
                #solar params
                self.PISI = 0.457#Fraction of visible shortwave irriadiance in global irradiance;
                #~0.457 at Bushland, which agrees with Meek et al. (1984) for other Western US locations
                self.KbVisConst = 1.034#Constant used in empirical equation to calculate the fraction
                # of direct beam irradiance in the visible (PAR) band (no units)
                self.KbVisExp = 2.234#Exponent used in empirical equation to calculate the fraction
                # of direct beam irradiance in the visible (PAR) band (no units)
                self.KbNirConst = 1.086#Constant used in empirical equation to calculate the fraction
                # of direct beam irradiance in the near-infrared (NIR) band (no units)
                self.KbNirExp = 2.384#Exponent used in empirical equation to calculate the fraction
                #of direct beam irradiance in the near-infrared (NIR) band (no units)
                #don't consider residue a separate class
                self.Kt = 
                self.modeltype = '2SEB'
                #do consider residue a separate class
                #modeltype = '3SEB'
                self.lat = np.nan#(dec deg)
                self.lon = np.nan#(dec deg)
                self.ele = np.nan#elevation (m)
                #Measurement heights (speed and temp)
                self.zu = 2
                self.zt = 2
                self.rowc = .6#row width (m)
                self.rowdir = # degrees from N, clockwise
                self.lz = -120#longitude of tz center
                self.tstep =
                self.tol = #tolerance for rah calc 
    def read_met(self):
        for f in os.listdir(p):
            if 'timeseries' in f and f[-4:]=='.csv':
                timeseries = read_csv(os.path.join(p,f))
                self.doy
                self.t
                self.P =
                self.P[np.isnan(self.P)]=aero.P_from_z(self.ele)
                self.Ta
                self.RH
                #compute ea, esa
                self.ea = aero.ea(self.P , self.Ta , self.RH)
                self.esa = aero.esa(self.Ta)
                self.RsOpt
                self.Rs
                self.latent = aero.latent(self.Ta)
                self.rho_a = aero.rho_a(self.Ta,self.ea,self.P)
                self.LAI =
                self.wc =
                self.wc[np.isnan(self.wc)]=(self.f_veg_sun[np.isnan(self.wc)]+self.f_veg_shade[np.isnan(self.wc)])*self.rowc[np.isnan(self.wc)]
                self.hc =
                self.LAIL = self.LAIL*self.rowc/self.wc
                self.precip =
                self.irrig =
    def read_can(self):
        for f in os.listdir(p):
            if 'cvfractions' in f and f[-4:]=='.csv':
                cvfractions = read_csv(os.path.join(p,f))
                self.f_veg_sun = 
                self.f_veg_shade = 
                self.f_soil_sun = 
                self.f_soil_shade = 
                self.f_res_sun = 
                self.f_res_shade = 
                self.T_veg_sun = 
                self.T_veg_shade = 
                self.T_soil_sun = 
                self.T_soil_shade = 
                self.T_res_sun = 
                self.T_res_shade = 
                self.T_rad = self.f_veg_sun*self.T_veg_sun+self.f_veg_shade*self.T_veg_shade+self.f_soil_sun*self.T_soil_sun+self.f_soil_shade*self.T_soil_shadeself.f_res_sun*self.T_res_sun+self.f_res_shade*self.T_res_shade
 
                self.T_s = (self.f_soil_sun*self.T_soil_sun+self.f_soil_shade*self.T_soil_shade+self.f_res_sun*self.T_res_sun+self.f_res_shade*self.T_res_shade)/(self.f_soil_sun+self.f_soil_shade+self.f_res_sun+self.f_res_shade)
 
                self.T_c = (self.f_veg_sun*self.T_veg_sun+self.f_veg_shade*self.T_veg_shade)/(self.f_veg_sun+self.f_veg_shade)
