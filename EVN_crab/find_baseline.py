import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5

crab = SkyCoord.from_name('M1')

Jb  = EarthLocation(3822625.820*u.m,  -154105.330*u.m,  5086486.215*u.m)
Wb  = EarthLocation(3828750.443*u.m,   442589.538*u.m,  5064921.734*u.m)
Ef  = EarthLocation(4033947.224*u.m,   486990.834*u.m,  4900431.022*u.m)
O8  = EarthLocation(3370965.870*u.m,   711466.239*u.m,  5349664.223*u.m)
Mc  = EarthLocation(4461369.645*u.m,   919597.177*u.m,  4449559.412*u.m)
Tr  = EarthLocation(3638558.209*u.m,  1221970.029*u.m,  5077036.901*u.m)
Hh  = EarthLocation(5085442.761*u.m,  2668263.845*u.m, -2768696.708*u.m)
Ur  = EarthLocation(228310.123*u.m,  4631922.765*u.m,  4367064.080*u.m)
Bd  = EarthLocation(-838201.146*u.m,  3865751.568*u.m,  4987670.881*u.m)
Sv  = EarthLocation(2730173.621*u.m,  1562442.830*u.m,  5529969.163*u.m)
Zc  = EarthLocation(3451207.472*u.m,  3060375.451*u.m,  4391915.056*u.m)
T6  = EarthLocation(-2826708.687*u.m,  4679237.046*u.m,  3274667.519*u.m)
Sr  = EarthLocation(4865182.739*u.m,   791922.723*u.m,  4035137.208*u.m)
Ro  = EarthLocation(4849092.524*u.m,  -360180.167*u.m,  4115109.375*u.m)

EVN = [Jb, Wb, Ef, O8, Mc, Tr, Hh, Ur, Bd, Sv, Zc, T6, Sr, Ro]

t_gp = Time('2015-10-19T00:17:47.415')

tel1 = Ef
tel2 = Jb

uvw_mat = np.zeros((len(EVN), 3))

for i in range(len(EVN)):
    tel = EVN[i]
    
    X = tel.x
    Y = tel.y
    Z = tel.z
    Xvec = np.array([X.value, Y.value, Z.value])
    ot=Time(t_gp, scale='utc', location=tel1)
    ot.delta_ut1_utc = 0.
    obst = ot.sidereal_time('mean')

    # I'm certain there's a better astropy way to get ot_avg in degrees
    h = obst.deg*u.deg - crab.ra   
    dec = crab.dec

    # matrix to transform xyz to uvw
    mat = np.array([(np.sin(h), np.cos(h), 0), (-np.sin(dec)*np.cos(h), np.sin(dec)*np.sin(h), 
                    np.cos(dec)), (np.cos(dec)*np.cos(h), -np.cos(dec)*np.sin(h), np.sin(dec))])

    uvw = np.dot(mat, Xvec)
    uvw_mat[i] = uvw
    
print uvw_mat[0]
