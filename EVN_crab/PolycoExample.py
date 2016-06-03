from pulsar.predictor import Polyco

t0 = fh.tell(unit='time')
t_GP = Time('2015-10-18T23:41:53.316550')

psr_polyco = Polyco('oct_polycob0531_ef.dat')
phase_pol = psr_polyco.phasepol(t0)

phase = np.remainder(phase_pol(t_GP.mjd), 1)

