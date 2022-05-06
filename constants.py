from scipy import constants

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

Cx = 90#Empirical constant  s m-1

b = 0.012#Empirical constant = 0.012 or b = 0.012[1 + cos(psiw)],
#       where psiw is wind direction, ccw from north
c = 0.0025#Empirical constant = 0.0025
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# gas constant for dry air, J/(kg*degK)
R_d = 287.04

#Monin-Obukhov parameters
dhc = .67#displacement height/canopy height C&N98, K95 say 0.65
zomhc = 0.125#momemtun roughness over canopy height K95
zohzom = 1#heat roughness over momentum roughness C&N say .2, K95 say exp(-2)

#Measurement heights (speed and temp)
zu = 2
zt = 2

#reference height if taken on separate height
hcref = np.nan

#bare soil roughness
#zombs = .4e-3 #m
#zoh_bs =

#soil heat flux
GRnDay = 0.1
GRnNight = 0.5

#canopy stomatal resistance s/m
rcDay = 50
rcNight = 200

#soil albedo vis/nir/wet/dry
rhosNir = 0.15
rhosNis = 0.25

#soil emissivity
emisSoil = 0.98

#residue albedo
#rhoresnir = 
#rhoresvis =

#residue emissivity
emisRes = 0.98

#leaf absorbance/emittance/reflectance
emisVeg = 0.98
zetaVis = 0.85#Leaf shortwave absorption in the visible (PAR) spectra (no units)
zetaNir = 0.2#Leaf shortwave absorption in the near infrared spectra (no units)

#leaf angle param
XE = 1.0

#canopy lw extinction
kappaIR = 0.95

#solar params
PISI = 0.457#Fraction of visible shortwave irriadiance in global irradiance;
    #~0.457 at Bushland, which agrees with Meek et al. (1984) for other Western US locations
KbVisConst = 1.034#Constant used in empirical equation to calculate the fraction
             # of direct beam irradiance in the visible (PAR) band (no units)
KbVisExp = 2.234#Exponent used in empirical equation to calculate the fraction
           # of direct beam irradiance in the visible (PAR) band (no units)
KbNirConst = 1.086#Constant used in empirical equation to calculate the fraction
             # of direct beam irradiance in the near-infrared (NIR) band (no units)
KbNirExp = 2.384#Exponent used in empirical equation to calculate the fraction
           #of direct beam irradiance in the near-infrared (NIR) band (no units)
                
