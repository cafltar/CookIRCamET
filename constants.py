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
gsc = 1367 #W/m2
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
r_d = 287.04

def sind(theta):
    return np.sin(np.deg2rad(theta))

def cosd(theta):
    return np.cos(np.deg2rad(theta))

def tand(theta):
    return np.tan(np.deg2rad(theta))
    
def asind(r):
    return np.rad2deg(np.asin(r))

def acosd(r):
    return np.rad2deg(np.acos(r))

def atand(r):
    return np.rad2deg(np.atan(r))

class inputs:
    def __init__(self):
        self.read_params()
        self.read_can()
        self.read_met()
        self.soil_albedo()
        return None

    def read_params(self):
        for f in os.listdir(p):
            if 'parameters' in f and f[-4:]=='.csv':
                parameters = read_csv(os.path.join(p,f))
                #defaults
                #user adjustable parameters
                self.cx = 90#Empirical constant  s m-1
                self.b = 0.012#Empirical constant = 0.012 or b = 0.012[1 + cos(psiw)],
                #       where psiw is wind direction, ccw from north
                self.c = 0.0025#Empirical constant = 0.0025        
                #Monin-Obukhov parameters
                self.dhc = .67#displacement height/canopy height C&N98, K95 say 0.65
                self.zomhc = 0.125#momemtun roughness over canopy height K95
                self.zohzomhc = 1#heat roughness over momentum roughness C&N say .2,
                # K95 say exp(-2),
                # K99 set equal for series model
                #surface roughness - bare soil
                self.dhbs = 0 #m
                self.zombs = 1 #m
                self.zohzombs = np.exp(-4.5) #from Stewart paper
                #Stewart, J. B., W. P. Kustas, K. S. Humes, W. D. Nichols, M. S.
                #Moran, and H. A. R. de Bruin, 1994: Sensible heat fluxradiometric
                #surface temperature relationship for eight semiarid
                #areas. J. Appl. Meteor., 33, 1110–1117.
                #soil heat flux
                #standing stubble
                self.dhss = 0.53 #m
                self.zomss = 0.058 #m
                self.zohzomss = 0.2 #from Sauer 1996
                #soil heat flux
                self.grn_day = -0.1
                self.grn_night = -0.5
                self.aG = -0.31
                #canopy stomatal resistance s/m
                self.rc_day = 50
                self.rc_night = 200
                #soil albedo vis/nir/wet/dry
                self.rhos_dry_nir = 0.15
                self.rhos_dry_vis = 0.25
                self.rhos_wet_nir = 0.15
                self.rhos_wet_vis = 0.25
                #soil emissivity
                self.emis_soil = 0.98
                #residue albedo
                self.rhor_dry_nir = 0.15 
                self.rhor_dry_vis = 0.25
                self.rhor_wet_nir = 0.15 
                self.rhor_wet_vis = 0.25
                #residue emissivity
                self.emis_res = 0.98
                #leaf absorbance/emittance/reflectance
                self.emis_veg = 0.98
                self.zeta_vis = 0.85#Leaf shortwave absorption in the visible (PAR) spectra (no units)
                self.zeta_nir = 0.2#Leaf shortwave absorption in the near infrared spectra (no units)
                #leaf angle param
                self.xe = 1.0
                #canopy lw extinction
                self.kappa_ir = 0.95
                #solar params
                self.pisi = 0.457#Fraction of visible shortwave irriadiance in global irradiance;
                #~0.457 at Bushland, which agrees with Meek et al. (1984) for other Western US locations
                self.kb_vis_const = 1.034#Constant used in empirical equation to calculate the fraction
                # of direct beam irradiance in the visible (PAR) band (no units)
                self.kb_vis_exp = 2.234#Exponent used in empirical equation to calculate the fraction
                # of direct beam irradiance in the visible (PAR) band (no units)
                self.kb_nir_const = 1.086#Constant used in empirical equation to calculate the fraction
                # of direct beam irradiance in the near-infrared (NIR) band (no units)
                self.kb_nir_exp = 2.384#Exponent used in empirical equation to calculate the fraction
                #of direct beam irradiance in the near-infrared (NIR) band (no units)
                self.kt = 1#turbidity coefficient only modify for severe turbidity 
                #don't consider residue a separate class
                self.model_type = '2SEB'
                #do consider residue a separate class
                #modeltype = '3SEB'
                self.lat = np.nan#(dec deg)
                self.lon = np.nan#(dec deg)
                self.radiometer_zenith = np.nan#(dec deg)
                self.radiometer_azimuth = np.nan#(dec deg)
                self.ele = np.nan#elevation (m)
                #Measurement heights (speed and temp)
                self.zu = 2
                self.zt = 2
                self.row_width = .6#row width (m)
                self.row_dir = # degrees from N, clockwise
                self.lz = -120#longitude of tz center
                self.tstep =
                self.tol = #tolerance for rah calc
                #evap coefficients for soil wetness/albedo
                self.ke_max
                self.kc_max
                #fraction wetted from irrigation % Typcially assume = 1.0 for sprinkler, 0.5 for LEPA; 0.1 for SDI
                self.fwet
                
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
                self.u
                #compute ea, esa
                self.ea = aero.ea(self.P , self.Ta , self.RH)
                self.esa = aero.esa(self.Ta)
                self.Rs = 
                self.latent = aero.latent(self.Ta)
                self.rho_a = aero.rho_a(self.Ta,self.ea,self.P)
                self.precip =
                self.irrig =

    def read_crop(self):
        for f in os.listdir(p):
            if 'crop' in f and f[-4:]=='.csv':
                crop = read_csv(os.path.join(p,f))
                self.LAI =
                self.SAI = #stem area index (stubble)
                self.wc =
                self.wc[np.isnan(self.wc)]=(self.f_veg_sun[np.isnan(self.wc)]+self.f_veg_shade[np.isnan(self.wc)])*self.rowc[np.isnan(self.wc)]
                self.hc = #canopy height
                self.hbs = #soil roughness Campbell & Norman 1998 2-6 mm
                self.hfr = #residue flat 1cm
                self.hss = #residue standing
                self.stubble = #logical - true if stubble, false if flat or none 
                self.LAIL = self.LAIL*self.rowc/self.wc
                
    def read_images(self):
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
                self.f_soil = self.f_soil_shade+self.f_soil_sun
                self.f_res = self.f_res_shade+self.f_res_sun
                self.f_veg = self.f_veg_shade+self.f_veg_sun
                self.hs[(~self.stubble)] = mp.maximum(self.hr,self.hbs)
                #relative soil fraction of ground
                self.f_soil_rel = self.f_soil/(self.f_soil+self.f_res)
                self.T_surf = ((self.f_soil_sun*self.T_soil_sun**4+self.f_soil_shade*self.T_soil_shade**4+self.f_res_sun*self.T_res_sun**4+self.f_res_shade*self.T_res_shade**4)/(self.f_res+self.f_soil))**(1/4)
                self.T_veg = ((self.f_veg_sun*self.T_veg_sun**4+self.f_veg_shade*self.T_veg_shade**4)/(self.f_veg))**(1/4)
                self.Tac = np.na*np.ones(self.T_veg.shape)
    def soil_albedo(self,RWC):                                                   
        self.soil_alb_vis =#Soil albedo in visible band, DAILY time steps (no units)                                               
        self.soil_alb_nir = #Soil albedo in nir band, DAILY time steps (no units)
        self.res_alb_vis =#Soil albedo in visible band, DAILY time steps (no units)                                               
        self.res_alb_nir = #Soil albedo in nir band, DAILY time steps (no units)
        self.alb_vis =#total albedo in visible band, DAILY time steps (no units)                                               
        self.alb_nir = #total albedo in nir band, DAILY time steps (no units)
