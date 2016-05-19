from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time

'''
'''

size = 2 ** 22
sample_rate = 32 * u.MHz
dt1 = 1/sample_rate
#
thread_ids = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
fedge = 1610.49 * u.MHz + ((np.linspace(0,15,16) % 8) // 2) * 32. * u.MHz
fref = fedge.mean() + sample_rate / 4
nchan = 64
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

def get_SN(outputsumfreq):
	noise_std = np.std(outputsumfreq[:1000]) 
	noise_mean = np.mean(outputsumfreq[:1000])
	sigs_noise = (outputsumfreq-noise_mean)/noise_std
	return sigs_noise

def correlate(X,Y):
	'''My function to find the correlation coefficient, to test np.corrcoef. results are the same.'''
	meanx = np.mean(X)
	meany = np.mean(Y)
	numerator = np.sum((X - meanx)*(Y-meany))
	denominator = np.sqrt(np.sum((X-meanx)**2))*np.sqrt(np.sum((Y-meany)**2))
	return numerator/denominator

def get_correlation_coefficients(fs1,fs2):
	'''function to find the averaged correlation coefficient of the 8 frequency bands.'''
	coefficients= []
	for i in range(8):
		X = fs1[i]
		Y = fs2[i]
		coefficient = np.corrcoef(X, Y)[0,1] # np.corrcoef returns a matrix [1, coeff;coeff,1]
		#meanx = np.mean(X)
		#meany = np.mean(Y)
		#numerator = np.sum((X - meanx)*(Y - meany))
		#numerator = np.sum(np.sqrt(abs(X*Y))*np.sign(X*Y))
		#numerator = np.sum(X*Y)
		#denominator = gp1.sigma*gp2.sigma
		#coefficient = numerator/denominator 
		coefficients.append(coefficient) 
	return np.mean(coefficients),coefficients
	
def overlap_freq_spec(gp1,gp2):
	f, axarr = plt.subplots(8, 1)
	frequency_interval = nchan+1
	for i in range(8):
		freqs = np.linspace((i)*16,(i+1)*16,frequency_interval)+1610.49 #actual frequencies 
		#axarr[i].plot(freqs,gp1.freq_spec[i*frequency_interval:(i+1)*frequency_interval])
		#axarr[i].plot(freqs,gp2.freq_spec[i*frequency_interval:(i+1)*frequency_interval])
		axarr[i].plot(gp2.freq_specs[i])
		axarr[i].plot(gp1.freq_specs[i])
		if i%2==0:
			axarr[i].set_ylabel('Intensity of peak with that frequency') 
	axarr[i].set_xlabel('frequency (MHz)')
	axarr[0].set_title('Frequency spectrum of giant pulses at around '+gp1.t_gp.value+' and '+gp2.t_gp.value)
	plt.show()

class GP_data(object):
	def __init__(self,fn,t_gp):
		self.t_gp = t_gp
		fh = mark5b.open(fn, mode='rs', nchan=16,
					    sample_rate=sample_rate, thread_ids=thread_ids, ref_mjd=57000)
		offset_gp = ((t_gp - fh.tell(unit='time')).to(u.s).value *
					 fh.frames_per_second * fh.samples_per_frame)
		fh.seek(int(offset_gp) - size // 2)
		self.d_dispersed = fh.read(size)
		
		self.process_data()
		self.process_output()
		
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
		self.dchan = np.fft.rfft(self.d_dedispersed.reshape(-1, 2*nchan, 16), axis=1) #axis 1 means across the 257 values in each time bin? 
		# Horribly inelegant, but works for now. 
		# Channels are not in order, and polarizations are separate
		self.dR = np.concatenate((self.dchan[:,::-1,8], self.dchan[...,0], self.dchan[:,::-1,10], self.dchan[...,2], self.dchan[:,::-1,12], self.dchan[...,4], self.dchan[:,::-1,14], self.dchan[...,6]), axis=1)
		self.dL = np.concatenate((self.dchan[:,::-1,9], self.dchan[...,1], self.dchan[:,::-1,11], self.dchan[...,3], self.dchan[:,::-1,13], self.dchan[...,5], self.dchan[:,::-1,15], self.dchan[...,7]), axis=1)
		self.output = (abs(self.dR)**2 + abs(self.dL)**2).T
		
		self.dchan_1 = np.fft.rfft(self.d_dispersed.reshape(-1, 2*nchan, 16), axis=1) # the dispersed spectrum
		self.dR1 = np.concatenate((self.dchan_1[:,::-1,8], self.dchan_1[...,0], self.dchan_1[:,::-1,10], self.dchan_1[...,2], self.dchan_1[:,::-1,12], self.dchan_1[...,4], self.dchan_1[:,::-1,14], self.dchan_1[...,6]), axis=1)
		self.dL1 = np.concatenate((self.dchan_1[:,::-1,9], self.dchan_1[...,1], self.dchan_1[:,::-1,11], self.dchan_1[...,3], self.dchan_1[:,::-1,13], self.dchan_1[...,5], self.dchan_1[:,::-1,15], self.dchan_1[...,7]), axis=1)
		self.output1 = (abs(self.dR1)**2 + abs(self.dL1)**2).T
		
	def process_output(self):
		self.outputsumfreq = self.output.sum(0) #a (8192,) matrix of the sum over all frequencies at given times
		self.outputsumfreq1 = self.output1.sum(0)
		self.sigs_noise = get_SN(self.outputsumfreq)
		self.sigs_noise1 = get_SN(self.outputsumfreq1)
		self.S_N = max(self.sigs_noise)
		print self.S_N
		self.background_freq = self.output[:,2500:3500]
		self.background_freq = self.background_freq.mean(1)
		
		self.peak_times = np.where(self.sigs_noise>self.S_N/15.)
		self.peak_time = np.argmax(self.sigs_noise) 
		self.b = np.where(self.peak_times[0] == self.peak_time)[0][0]
		self.bin1 = self.output[:,self.peak_times[0][:self.b+1]].mean(1)
		self.bin2 = self.output[:,self.peak_times[0][self.b+1:]].mean(1)
		#self.sigma = np.std((self.bin1-np.mean(self.bin1)-self.background_freq)*(self.bin2-np.mean(self.bin2)-self.background_freq))
		#self.sigma = np.sqrt(self.sigma)  
		#self.output_pulse = np.vstack((self.bin1,self.bin2))
		self.output_pulse = self.output[:,self.peak_time:self.peak_time+2]    
		self.freq_spec1 = self.output_pulse.mean(1)-self.background_freq
		
		self.freq_spec = (self.bin1-np.mean(self.bin1)-self.background_freq)*(self.bin2-np.mean(self.bin2)-self.background_freq)
		self.freq_specs = []
		for i in range(8):
			self.freq_specs.append(self.freq_spec[nchan/16+i*nchan*7/8+i*nchan/8:nchan/16+(i+1)*nchan*7/8+i*nchan/8]) #the values 220 and 36 cuts the beginning and end of each frequency band
		#computes signal to noise after summing all the frequencies
		
	def plot_figs(self):
		#plots the frequency spectrum: 
		plt.close('all')
		f, axarr = plt.subplots(8, 1)
		frequency_interval = nchan+1
		for i in range(8):
			freqs = np.linspace((i)*16,(i+1)*16,frequency_interval)+1610.49 #actual frequencies 
			#freqs = np.linspace((i)*frequency_interval,(i+1)*frequency_interval,frequency_interval)
			axarr[i].plot(freqs,self.freq_spec[i*frequency_interval:(i+1)*frequency_interval])
			if i%2==0:
				axarr[i].set_ylabel('Intensity of peak with that frequency') 
		axarr[i].set_xlabel('frequency (MHz)')
		axarr[0].set_title('Frequency spectrum of giant pulse at around '+self.t_gp.value)
		
		#plots the dynamic spectrum de-dispersed: 
		plt.figure()
		#plt.imshow(self.output, aspect='auto',extent=(-8*8192/1000,8*8192/1000,1610.49+16*8,1610.49))
		plt.imshow(self.output, aspect='auto')
		plt.xlabel('time (ms)')
		plt.ylabel('frequency (MHz)')
		plt.title('dynamic spectrum of de-dispersed giant pulse around '+self.t_gp.value)
		plt.colorbar()

		#plots the dynamic specturm dispersed as a single stiched plot:
		plt.figure()
		#plt.imshow(self.output1, aspect='auto',extent=(-8*8192/1000,8*8192/1000,1610.49+16*8,1610.49))#actual times and frequencies
		plt.imshow(self.output1, aspect='auto')
		plt.title('dynamic spectrum of dispersed giant pulse around '+self.t_gp.value+'separated') 
		plt.xlabel('time (ms)')
		plt.ylabel('frequency (MHz)')
		plt.title('dynamic spectrum of dispersed giant pulse around '+self.t_gp.value)
		plt.colorbar()
		
		#plots the dynamic specturm dispersed as separate subplots for each chanel:
		f, axarr = plt.subplots(8, 2)
		for i in range(16):
			axarr[i%8,i/8].imshow((abs(self.dchan_1[...,i])**2).T,aspect = 'auto')
			axarr[i%8,i/8].set_title('chanel {}'.format(i))
		axarr[i%8,i/8].set_title('dispersed pulse in its 16 channels')
		
		#plots the signal vs noise of de-dispersed
		plt.figure() 
		plt.plot(self.sigs_noise)
		plt.xlabel('time ($16 \mu s$)')
		plt.ylabel('S/N')
		plt.title('Signal to noise of de-dispersed pulse around '+self.t_gp.value)
		
		#plots the signal vs noise of dispersed - looks like noise, so won't plot
		'''
		plt.figure() 
		plt.plot(self.sigs_noise1)
		plt.xlabel('time ($16 \mu s$)')
		plt.ylabel('S/N')
		plt.title('Signal to noise of dispersed pulse around '+self.t_gp.value)
		plt.show()
   		'''
   	def plot_fs(self):
   		'''plots the frequency spectrum as a single plot'''
		f, axarr = plt.subplots(2, 1)
		axarr[0].plot(self.freq_spec) 
		axarr[0].set_xlabel('frequency')
		axarr[0].set_ylabel('Auto correlated frequency spectrum')
		axarr[1].plot(self.freq_spec1)
		axarr[1].set_ylabel('Peak frequency spectrum')
		axarr[1].set_xlabel('frequency') 
		plt.show()
if __name__ == "__main__":
	i = 3
	j = 4
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

	fn1 = '/cita/h/home-2/xzxu/trails/data/ef/ek036a_ef_no00{}.m5a'.format(scan_no1)
	fn2 = '/cita/h/home-2/xzxu/trails/data/ef/ek036a_ef_no00{}.m5a'.format(scan_no2)
	#fn2 = '/cita/h/home-2/xzxu/trails/data/jb/ek036a_jb_no00{}.m5a'.format(scan_no1)
	t_gp1 = Time(t1)
	t_gp2 = Time(t2)
	
	gp1 = GP_data(fn1,t_gp1)
	gp2 = GP_data(fn2,t_gp2)
	gp2.plot_fs() 
	#gp2.plot_figs()
	overlap_freq_spec(gp1,gp2)   
  	
  	c = get_correlation_coefficients(gp1.freq_specs,gp2.freq_specs) # This average one is better, as the one below takes into account non-data information, so returns a higher correlation coefficient
  	#c1 = get_correlation_coefficient(gp1,gp2)
  	dt = t_gp2-t_gp1
  	dts = dt.sec 
  	print 'The correlation coefficient is {}, the time lag between the two pulses is {} seconds.'.format(c[0],dts)
	
    
    
    
    
    
    

	




