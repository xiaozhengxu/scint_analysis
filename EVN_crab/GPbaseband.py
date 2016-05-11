from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time

'''
i = 13
j = 15
text_name = 'all20.txt'
with open(text_name, 'r') as f:
	text = f.read()
	text_lines = text.split('\n')
strings1 = text_lines[i-1].split()
strings2 = text_lines[j-1].split()
if text_name[0] == 'a': # get information from a all__.txt file
	scan_no1 = strings1[1]
	scan_no2 = strings2[1]
	t1 = strings1[2]
	t2 = strings2[2]
elif text_name[0] == 's': # get information from a scan__.txt file
	scan_no1 = text_name[4:6]
	scan_no2 = text_name[4:6]
	t1 = strings1[0]
	t2 = strings2[0]
print scan_no1,t1,scan_no2,t2

fn1 = '/cita/h/home-2/xzxu/trails/data/ek036a_ef_no00{}.m5a'.format(scan_no1)
fn2 = '/cita/h/home-2/xzxu/trails/data/ek036a_ef_no00{}.m5a'.format(scan_no2)
t_gp1 = Time(t1)
t_gp2 = Time(t2)
'''

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


def get_correlation_coefficients(gp1,gp2):
	coefficients= []
	for i in range(8):
		X = gp1.freq_specs[i]
		Y = gp2.freq_specs[i]
		#coefficient = np.dot((X-np.mean(X)),(Y-np.mean(Y)))/(np.std(X)*np.std(Y))
		coefficient = np.corrcoef(X, Y)[0,1] # np.corrcoef returns a matrix [1, coeff;coeff,1]
		coefficients.append(coefficient) 
	return np.mean(coefficients),coefficients

class GP_data(object):
	def __init__(self,fn,t_gp):
		fh = mark5b.open(fn, mode='rs', nchan=16,
					    sample_rate=sample_rate, thread_ids=thread_ids, ref_mjd=57000)
		offset_gp = ((t_gp - fh.tell(unit='time')).to(u.s).value *
					 fh.frames_per_second * fh.samples_per_frame)
		fh.seek(int(offset_gp) - size // 2)
		self.d_dispersed = fh.read(size)
		
		self.process_data()
		self.get_output()
		
	def process_data(self):
		ft = np.fft.rfft(self.d_dispersed, axis=0)
		# Second half of IFs have Fedge at top, need to subtract frequencies, 
		# and not conjugate coherent phases
		f = fedge + np.fft.rfftfreq(self.d_dispersed.shape[0], dt1)[:, np.newaxis]
		f[:,8:] = fedge[8:] - np.fft.rfftfreq(self.d_dispersed.shape[0], dt1)[:, np.newaxis]
		ft[:,:8] *= dm(f[:,:8], fref, kind='complex').conj()
		ft[:,8:] *= dm(f[:,8:], fref, kind='complex')
		self.d_dedispersed = np.fft.irfft(ft, axis=0)

		# Channelize the data
		self.dchan = np.fft.rfft(self.d_dedispersed.reshape(-1, 2*nchan, 16), axis=1)
		# Horribly inelegant, but works for now. 
		# Channels are not in order, and polarizations are separate
		self.dR = np.concatenate((self.dchan[:,::-1,8], self.dchan[...,0], self.dchan[:,::-1,10], self.dchan[...,2], self.dchan[:,::-1,12], self.dchan[...,4], self.dchan[:,::-1,14], self.dchan[...,6]), axis=1)
		self.dL = np.concatenate((self.dchan[:,::-1,9], self.dchan[...,1], self.dchan[:,::-1,11], self.dchan[...,3], self.dchan[:,::-1,13], self.dchan[...,5], self.dchan[:,::-1,15], self.dchan[...,7]), axis=1)
		#plt.ion()
	def get_output(self):
		self.output = (abs(self.dR)**2 + abs(self.dL)**2).T
		self.outputsumfreq = self.output.sum(0) #a (8192,) matrix of the sum over all frequencies at given times
		peak_time = np.argmax(self.outputsumfreq) 
		#a matrix containing only the pulse at 3989 time bin and 3990
		self.output_pulse = self.output[:,peak_time:peak_time+1]    
		self.freq_spec = self.output_pulse.sum(1) 
		self.freq_specs = []
		for i in range(8):
			self.freq_specs.append(self.freq_spec[18+i*220+i*36:18+(i+1)*220+i*36]) #the values 220 and 36 cuts the beginning and end of each frequency band
		#computes signal to noise after summing all the frequencies
		self.noise_std = np.std(self.outputsumfreq[:1000]) 
		self.noise_mean = np.mean(self.outputsumfreq[:1000])
		self.sigs_noise = (self.outputsumfreq-self.noise_mean)/self.noise_std
		self.S_N = max(self.sigs_noise)
	
	def plot_figs(self):
		#plots the frequency spectrum: 
		plt.close('all')
		f, axarr = plt.subplots(8, 1)
		frequency_interval = 2056/8
		for i in range(8):
			freqs = np.linspace((i)*16,(i+1)*16,frequency_interval)+1610.49 #actual frequencies 
			freqs = np.linspace((i)*frequency_interval,(i+1)*frequency_interval,frequency_interval)
			axarr[i].plot(freqs,self.freq_spec[i*frequency_interval:(i+1)*frequency_interval])
			if i%2==0:
				axarr[i].set_ylabel('Intensity of peak with that frequency') 
		axarr[i].set_xlabel('frequency (MHz)')
		
		#plots the dynamic spectrum de-dispersed: 
		plt.figure()
		#plt.imshow(self.output, aspect='auto',extent=(-8*8192/1000,8*8192/1000,1610.49+16*8,1610.49))
		plt.imshow(self.output, aspect='auto',extent=(0,8192,1610.49+16*8,1610.49))
		plt.xlabel('time (ms)')
		plt.ylabel('frequency (MHz)')
		plt.title('dynamic spectrum of de-dispersed giant pulse')
		plt.colorbar()

		#plots the dynamic specturm dispersed as separate subplots for each chanel:
		self.dchan_1 = np.fft.rfft(self.d_dispersed.reshape(-1, 2*nchan, 16), axis=1) # the dispersed spectrum
		self.dR1 = np.concatenate((self.dchan_1[:,::-1,8], self.dchan_1[...,0], self.dchan_1[:,::-1,10], self.dchan_1[...,2], self.dchan_1[:,::-1,12], self.dchan_1[...,4], self.dchan_1[:,::-1,14], self.dchan_1[...,6]), axis=1)
		self.dL1 = np.concatenate((self.dchan_1[:,::-1,9], self.dchan_1[...,1], self.dchan_1[:,::-1,11], self.dchan_1[...,3], self.dchan_1[:,::-1,13], self.dchan_1[...,5], self.dchan_1[:,::-1,15], self.dchan_1[...,7]), axis=1)
		self.output1 = (abs(self.dR1)**2 + abs(self.dL1)**2).T
		plt.figure()
		#plt.imshow(self.output, aspect='auto',extent=(-8*8192/1000,8*8192/1000,1610.49+16*8,1610.49)) #actual times and frequencies
		plt.imshow(self.output1, aspect='auto',extent=(0,8192,1610.49+16*8,1610.49))# matrix value times
		plt.xlabel('time (ms)')
		plt.ylabel('frequency (MHz)')
		plt.title('dynamic spectrum of dispersed giant pulse')
		plt.colorbar()
		
		f, axarr = plt.subplots(8, 2)
		for i in range(16):
			axarr[i%8,i/8].imshow((abs(self.dchan_1[...,i])**2).T,aspect = 'auto')
			axarr[i%8,i/8].set_title('chanel {}'.format(i))
		axarr[i%8,i/8].set_title('dispersed pulse in its 16 channels')
		
		#plots the signal vs noise
		plt.figure() 
		plt.plot(self.sigs_noise)
		plt.xlabel('time ($16 \mu s$)')
		plt.ylabel('S/N')
		
		plt.show()
		
    
if __name__ == "__main__":
	gp1 = GP_data(fn1,t_gp1)
	gp2 = GP_data(fn2,t_gp2)
	gp2.plot_figs()    
  	c = get_correlation_coefficients(gp1,gp2)
  	dt = t_gp2-t_gp1
  	dts = dt.sec 
  	print 'The correlation coefficient is {}, the time lag between the two pulses is {} seconds.'.format(c[0],dts)
	
    
    
    
    
    
    

	




