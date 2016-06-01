
from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time

from pickle import dump, load
from os.path import exists 
from GPbaseband import *

text_name = 'allGPs.txt'
with open(text_name, 'r') as f:
	text = f.read()
	text_lines = text.split('\n')

print len(text_lines)

#nchan = 128

if exists('./figures/correlation_coeff/{}_{}m_may31.npz'.format(text_name,nchan)):
	#f=open('./figures/correlation_coeff/{}m.npz'.format(text_name),'r') 
	npzfile=np.load('./figures/correlation_coeff/{}_{}m_may31.npz'.format(text_name,nchan))
	freq_values3 = npzfile['freq_values3']
	freq_values2 = npzfile['freq_values2']
	freq_values1 = npzfile['freq_values1']
	time_values = npzfile['time_values']
	print freq_values3.shape
	#f.close()
else:
	freq_values1 = np.zeros(shape = (0,8,nchan*7/8))
	freq_values2 = np.zeros(shape = (0,8,nchan*7/8))
	freq_values3 = np.zeros(shape = (0,8,nchan*7/8))
	time_values = np.zeros(0)
	for i,text in enumerate(text_lines[:-2]):
		strings1 = text.split()
		if text_name[0] == 'a': # get information from a all__.txt file
			scan_no = strings1[1]
			t = strings1[2]
		elif text_name[0] == 's': # get information from a scan__.txt file
			scan_no = text_name[4:6]
			t = strings1[0]
		fn = '/cita/d/homes/home-2/xzxu/trails/data/ef/ek036a_ef_no00{}.m5a'.format(scan_no)
		t_gp = Time(t)
		try:	
			gp = GP_data(fn,t_gp)
			freq_values1 = np.append(freq_values1,np.array([gp.freq_specs1]),0)
			freq_values2 = np.append(freq_values2,np.array([gp.freq_specs2]),0)
			freq_values3 = np.append(freq_values3,np.array([gp.freq_specs3]),0)
			time_values = np.append(time_values,t)
			print 'Appended frequency spectrums for pulse {}'.format(i+1)
		except AssertionError:
			print 'Assertion error:',scan_no,t
			continue
	np.savez('./figures/correlation_coeff/{}_{}m_may31.npz'.format(text_name,nchan),freq_values1 = freq_values1, time_values = time_values, freq_values2=freq_values2, freq_values3 = freq_values3)
	
	
	
def calculate_values():
	ctvalues = np.zeros(shape = (freq_values1.shape[0],freq_values1.shape[0],4)) #matrix
	if exists('./figures/correlation_coeff/{}_{}ctvalues_may31.npy'.format(text_name,nchan)):
		ctvalues = np.load('./figures/correlation_coeff/{}_{}ctvalues_may31.npy'.format(text_name,nchan))
	else:
		ctvalues = np.zeros(shape = (0,4)) 
		for i in range(freq_values1.shape[0]):
			for j in range(freq_values1.shape[0] - i):
				t_gp1 = Time(time_values[i])
				t_gp2 = Time(time_values[i+j])
				dt = t_gp2-t_gp1
				if abs(dt.sec)<300 and abs(dt.sec)>1e-5:
				#if abs(dt.sec)<10 and abs(dt.sec)>0.01:
					c1 = get_correlation_coefficients(freq_values1[i,...],freq_values1[i+j,...])[0]
					c2 = get_correlation_coefficients(freq_values2[i,...],freq_values2[i+j,...])[0]
					c3 = get_correlation_coefficients(freq_values3[i,...],freq_values3[i+j,...])[0]
					ctvalues = np.append(ctvalues,np.array([[c1,c2,c3,abs(dt.sec)]]),axis = 0)
					#print 'i= ',i,'j= ',j
					print ctvalues.shape
		np.save('./figures/correlation_coeff/{}_{}ctvalues_may31.npy'.format(text_name,nchan),ctvalues)
	
	plt.figure()
	plt.semilogx(ctvalues[:,3],ctvalues[:,0],'bo')
	plt.xlabel('time lag (seconds)')
	plt.ylabel('Correlation Coefficient')
	plt.title('{} frequency channels, correlating autocorrelation within single pulses, \n without subtracting mean of bin 1 and bin2 before multiplying'.format(nchan)) 
	
	plt.figure()
	plt.semilogx(ctvalues[:,3],ctvalues[:,1],'bo')
	plt.xlabel('time lag (seconds)')
	plt.ylabel('Correlation Coefficient')
	plt.title('{} frequency channels, correlating autocorrelation within single pulses,\n subtracting mean of bin 1 and bin2 before multiplying'.format(nchan)) 
	
	plt.figure()
	plt.semilogx(ctvalues[:,3],ctvalues[:,2],'bo')
	plt.xlabel('time lag (seconds)')
	plt.ylabel('Correlation Coefficient')
	plt.title('{} frequency channels, correlating original peaks \n without autocorrelation within single pulse'.format(nchan)) 
	plt.show()
	
calculate_values()

