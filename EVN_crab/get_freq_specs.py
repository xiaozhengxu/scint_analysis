
from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time

from pickle import dump, load
from os.path import exists 
from GPbaseband import *

text_name = 'all10.txt'
with open(text_name, 'r') as f:
	text = f.read()
	text_lines = text.split('\n')

print len(text_lines)


if exists('./figures/correlation_coeff/{}_{}m.npz'.format(text_name,nchan)):
	#f=open('./figures/correlation_coeff/{}m.npz'.format(text_name),'r') 
	npzfile=np.load('./figures/correlation_coeff/{}_{}m.npz'.format(text_name,nchan))
	freq_values = npzfile['freq_values']
	time_values = npzfile['time_values']
	print freq_values.shape
	#f.close()
else:
	freq_values = np.zeros(shape = (0,8,nchan*7/8))
	time_values = np.zeros(0)
	for i,text in enumerate(text_lines[:-1]):
		strings1 = text.split()
		if text_name[0] == 'a': # get information from a all__.txt file
			scan_no = strings1[1]
			t = strings1[2]
		elif text_name[0] == 's': # get information from a scan__.txt file
			scan_no = text_name[4:6]
			t = strings1[0]
		fn = '/cita/h/home-2/xzxu/trails/data/ef/ek036a_ef_no00{}.m5a'.format(scan_no)
		t_gp = Time(t)
		try:	
			gp = GP_data(fn,t_gp)
			freq_values = np.append(freq_values,np.array([gp.freq_specs]),0)
			time_values = np.append(time_values,t)
			print 'Appended frequency spectrums for pulse {}'.format(i+1)
		except AssertionError:
			print 'Assertion error:',scan_no,t
			continue
	np.savez('./figures/correlation_coeff/{}_{}m.npz'.format(text_name,nchan),freq_values = freq_values, time_values = time_values)
	
def calculate_values():
	#ctvalues = np.zeros(shape = (freq_values.shape[0],freq_values.shape[0],2)) #matrix
	if exists('./figures/correlation_coeff/{}_{}ctvalues.npy'.format(text_name,nchan)):
		ctvalues = np.load('./figures/correlation_coeff/{}_{}ctvalues.npy'.format(text_name,nchan))
	else:
		ctvalues = np.zeros(shape = (0,2)) 
		for i in range(freq_values.shape[0]):
			for j in range(freq_values.shape[0] - i):
				t_gp1 = Time(time_values[i])
				t_gp2 = Time(time_values[i+j])
				dt = t_gp2-t_gp1
				if abs(dt.sec)<300 and abs(dt.sec)>1e-5:
					c = get_correlation_coefficients(freq_values[i,...],freq_values[i+j,...])[0]
					ctvalues = np.append(ctvalues,np.array([[c,abs(dt.sec)]]),axis = 0)
					print ctvalues.shape
		np.save('./figures/correlation_coeff/{}_{}ctvalues.npy'.format(text_name,nchan),ctvalues)
	
	plt.figure()
	plt.semilogx(ctvalues[:,1],ctvalues[:,0],'bo')
	plt.xlabel('time lag (seconds)')
	plt.ylabel('Correlation Coefficient')
	plt.show()
	
calculate_values()


