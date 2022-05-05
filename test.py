import unittest
import aero,canopy,fluxes,solar,tseb

class test_aero(unittest.TestCase):
    def test_ea():
        P = 101.325
        Ta = 20
        HP = 50
        eaOPT = 1
        result = aero.ea(P , Ta , HP , eaOPT)
        self.assertAlmostEqual(result, HP)
        HP = 50
        eaOPT = 2
        result = aero.ea(P , Ta , HP , eaOPT)
        self.assertAlmostEqual(result, 6)
        HP = 
        eaOPT = 3
        result = aero.ea(P , Ta , HP , eaOPT)
        self.assertAlmostEqual(result, 6)

    def test_esa():
    def test_latent():
    def test_slope():
    def test_rho_a():
    def test_Twet():

class test_canopy(unittest.TestCase):
class test_fluxes(unittest.TestCase):
class test_solar(unittest.TestCase):
class test_tseb(unittest.TestCase):
    
if __name__=='__main()__':
    unittest.main()
