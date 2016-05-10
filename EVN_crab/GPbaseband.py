from baseband import mark5b
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time

fn = '/cita/h/home-2/xzxu/trails/data/ek036a_ef_no0008.m5a'
#t_gp = Time('2015-10-19T02:02:36.763878')
#t_gp = Time('2015-10-19T02:03:12.329898')
t_gp = Time('2015-10-19T00:17:47.415204')
size = 2 ** 22
sample_rate = 32 * u.MHz
dt1 = 1/sample_rate
#
thread_ids = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
fedge = 1610.49 * u.MHz + ((np.linspace(0,15,16) % 8) // 2) * 32. * u.MHz
fref = fedge.mean() + sample_rate / 4
nchan = 256
# October DM from JB ephemeris (1e-2 is by eye correction)
dm = (56.7957 + 1e-2) * u.pc / u.cm**3


class DispersionMeasure(u.Quantity):

    dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc
    _default_unit = u.pc / u.cm**3

    def __new__(cls, dm, unit=None, dtype=None, copy=True):
        if unit is None:
            unit = getattr(dm, 'unit', cls._default_unit)
        self = super(DispersionMeasure, cls).__new__(cls, dm, unit,
                                                     dtype=dtype, copy=copy)
        if not self.unit.is_equivalent(cls._default_unit):
            raise u.UnitsError("Dispersion measures should have units of "
                               "pc/cm^3")
        return self

    def __quantity_subclass__(self, unit):
        if unit.is_equivalent(self._default_unit):
            return DispersionMeasure, True
        else:
            return super(DispersionMeasure,
                         self).__quantity_subclass__(unit)[0], False

    def __call__(self, f, fref, kind='complex'):
        d = self.dispersion_delay_constant * self
        if kind == 'delay':
            return d * (1./f**2 - 1./fref**2)
        else:
            dang = (d * f * (1./fref-1./f)**2) * u.cycle
            if kind == 'phase':
                return dang
            elif kind == 'complex':
                return np.exp(dang.to(u.rad).value * 1j)

        raise ValueError("kind not one of 'delay', 'phase', or 'complex'")

dm = DispersionMeasure(dm)


if __name__ == '__main__':

	fh = mark5b.open(fn, mode='rs', nchan=16,
			        sample_rate=sample_rate, thread_ids=thread_ids, ref_mjd=57000)
	offset_gp = ((t_gp - fh.tell(unit='time')).to(u.s).value *
			     fh.frames_per_second * fh.samples_per_frame)
	fh.seek(int(offset_gp) - size // 2)
	d_dispersed = fh.read(size)
	ft = np.fft.rfft(d_dispersed, axis=0)
	# Second half of IFs have Fedge at top, need to subtract frequencies, 
	# and not conjugate coherent phases
	f = fedge + np.fft.rfftfreq(d_dispersed.shape[0], dt1)[:, np.newaxis]
	f[:,8:] = fedge[8:] - np.fft.rfftfreq(d_dispersed.shape[0], dt1)[:, np.newaxis]
	ft[:,:8] *= dm(f[:,:8], fref, kind='complex').conj()
	ft[:,8:] *= dm(f[:,8:], fref, kind='complex')
	d_dedispersed = np.fft.irfft(ft, axis=0)

	# Channelize the data
	dchan = np.fft.rfft(d_dedispersed.reshape(-1, 2*nchan, 16), axis=1)
	# Horribly inelegant, but works for now. 
	# Channels are not in order, and polarizations are separate
	dR = np.concatenate((dchan[:,::-1,8], dchan[...,0], dchan[:,::-1,10], dchan[...,2], dchan[:,::-1,12], dchan[...,4], dchan[:,::-1,14], dchan[...,6]), axis=1)
	dL = np.concatenate((dchan[:,::-1,9], dchan[...,1], dchan[:,::-1,11], dchan[...,3], dchan[:,::-1,13], dchan[...,5], dchan[:,::-1,15], dchan[...,7]), axis=1)
	#plt.ion()

	output = (abs(dR)**2 + abs(dL)**2).T
	outputsumfreq = output.sum(0) #a (8192,) matrix of the sum over all frequencies at given times
	peak_time = np.argmax(outputsumfreq) 
	#a matrix containing only the pulse at 3989 time bin and 3990
	output_pulse = output[:,peak_time:peak_time+1]    
	freq_spec = output_pulse.sum(1)

	#computes signal to noise after summing all the frequencies
	noise_std = np.std(outputsumfreq[:1000]) 
	noise_mean = np.mean(outputsumfreq[:1000])
	sigs_noise = (outputsumfreq-noise_mean)/noise_std
	S_N = max(sigs_noise)
	
	#plots the frequency spectrum: 
	plt.close('all')
	f, axarr = plt.subplots(8, 1)
	frequency_interval = 2056/8
	for i in range(8):
		freqs = np.linspace((i)*16,(i+1)*16,frequency_interval)+1610.49 #actual frequencies 
		#freqs = np.linspace((i)*frequency_interval,(i+1)*frequency_interval,frequency_interval)
		axarr[i].plot(freqs,freq_spec[i*frequency_interval:(i+1)*frequency_interval])
		if i%2==0:
			axarr[i].set_ylabel('Intensity of peak with that frequency') 
	axarr[i].set_xlabel('frequency (MHz)')
	plt.show()

	#plots the dynamic spectrum de-dispersed: 
	plt.figure()
	plt.imshow(output, aspect='auto',extent=(-8*8192/1000,8*8192/1000,1610.49+16*8,1610.49))
	plt.xlabel('time (ms)')
	plt.ylabel('frequency (MHz)')
	plt.title('dynamic spectrum of de-dispersed giant pulse')
	plt.colorbar()

	#plots the dynamic specturm dispersed as separate subplots for each chanel:
	dchan_1 = np.fft.rfft(d_dispersed.reshape(-1, 2*nchan, 16), axis=1) # the dispersed spectrum
	f, axarr = plt.subplots(16, 1)
	for i in range(16):
		axarr[i].imshow((abs(dchan_1[...,i])**2).T,aspect = 'auto')
		axarr[i].set_title('chanel {}'.format(i))
		
	#plots the signal vs noise
	plt.figure() 
	plt.plot(sigs_noise)
	plt.xlabel('time ($16 \mu s$)')
	plt.ylabel('S/N')
	plt.show()

    
    
    
    
    
    
    
    
    
    

	




