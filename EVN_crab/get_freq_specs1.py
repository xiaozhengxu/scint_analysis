
from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time

from pickle import dump, load
from os.path import exists 
from GPbaseband1 import *

text_name = 'all7_sorted1.txt'
with open(text_name, 'r') as f:
	text = f.read()
	text_lines = text.split('\n')

print len(text_lines)

nchan = 512
if exists('./figures/correlation_coeff/giant_pulses{}chan.npz'.format(nchan)):
	with np.load('./figures/correlation_coeff/giant_pulses{}chan.npz'.format(nchan)) as npzfile:
		freq_values = npzfile['freq_values']
		time_values = npzfile['time_values']
		noise_sigma_values = npzfile['noise_sigma_values']
else:
	texts = []
	freq_values = np.zeros(shape = (0,2**22/(nchan*2)+8))
	noise_sigma_values = np.zeros(0)
	time_values = np.zeros(0)
	for i,text in enumerate(text_lines[:-1]):
		strings1 = text.split()
		if text_name[0] == 'a': # get information from a all__.txt file
			scan_no = strings1[0]
			t = strings1[1]
		fn = '/cita/d/homes/home-2/xzxu/trails/data/ef/ek036a_ef_no00{}.m5a'.format(scan_no)
		t_gp = Time(t)
		try:	
			gp = GP_data(fn,t_gp)
			if gp.S_N>20:
				freq_values = np.append(freq_values,np.array([gp.freq_spec]),0)
				time_values = np.append(time_values,t)
				noise_sigma_values = np.append(noise_sigma_values,gp.sigma_noise)
				texts.append(text_lines[i]+'\n')
				print 'Appended frequency spectrums for pulse {}'.format(i+1)
		except AssertionError:
			print 'Assertion error:',scan_no,t
			continue

	np.savez('./figures/correlation_coeff/giant_pulses{}chan.npz'.format(nchan),freq_values = freq_values,time_values = time_values,noise_sigma_values = noise_sigma_values)

gpab=np.load('./figures/correlation_coeff/gpa_smoothed.npy')
	
'''
with open('good_pulses.txt','w') as f:
	for t in texts:
		f.write(t)
'''
#gp_average = np.mean(freq_values,0)
#np.save('./figures/correlation_coeff/giant_pulses_average{}chan.npz'.format(nchan),gp_average)
def process_freq_spec(gp1,index):
	'''function takes a continuous frequency spectrum and cuts it into 8 bands, divide by the giant pulse mean, and normalize it by itself gp1=freq_spec[i]. function also plots the frequency spectrum'''
	gp1_8=np.zeros(shape=(0,425))
	for j in range(8):gp1_8=np.append(gp1_8,np.array([gp1[j*513+40:j*513+465]]),0)
	gp1_8/=gpab #Dividing by giant pulse average
	for i in range(8):gp1_8[i]=gp1_8[i]/gp1_8[i].mean()-1.
	
	f,axarr=plt.subplots(8,1)
	for i in range(8):axarr[i].plot(gp1_8[i])
	axarr[0].set_title('Frequency spectrum of giant pulse at{}'.format(time_values[index]))
	
	return gp1_8

def transform_to_timelag(fgp,index):
	tgp=np.zeros(shape=(0,213))
	for i in range(8):tgp=np.append(tgp,np.array([np.fft.rfft(fgp[i])]),0)
	
	f,axarr=plt.subplots(8,1)
	for i in range(8):axarr[i].plot(np.linspace(0,16,213),abs(tgp[i]))
	axarr[0].set_title('Time lag spectrum of giant pulse at {}'.format(time_values[index]))
	axarr[7].set_xlabel('Time lag(\mu s)')
	return tgp

#range: gp1[j*513+40:j*513+465]
#calculate_values()


