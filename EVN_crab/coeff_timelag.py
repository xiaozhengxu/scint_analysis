
from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time

from pickle import dump, load
from os.path import exists 
from GPbaseband import *

text_name = 'all20.txt'
with open(text_name, 'r') as f:
	text = f.read()
	text_lines = text.split('\n')

print len(text_lines)


if exists('./figures/correlation_coeff/tvalues.txt'):
	f=open('./figures/correlation_coeff/tvalues1.txt','r') 
	tvalues=load(f)
	print len(tvalues)
	f.close()
else:
	tvalues = []
if exists('./figures/correlation_coeff/cvalues1.txt'):
	f=open('./figures/correlation_coeff/cvalues1.txt','r') 
	cvalues=load(f)
	print len(cvalues)
	f.close()
else: 
	cvalues= []

def calculate_values(order):
	with open('./figures/correlation_coeff/datalog{}.txt'.format(order), 'w') as f1:
		f1.seek(0,2) #2 means end of file, 0 is the off_set
		for delta in [1,2,3]:	
			for i in range(0,len(text_lines)-delta-1):
			#for i in [25]:
				j = i+delta	

				strings1 = text_lines[i].split()
				strings2 = text_lines[j].split()
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
			
				fn1 = '/cita/h/home-2/xzxu/trails/data/ek036a_ef_no00{}.m5a'.format(scan_no1)
				fn2 = '/cita/h/home-2/xzxu/trails/data/ek036a_ef_no00{}.m5a'.format(scan_no2)
				t_gp1 = Time(t1)
				t_gp2 = Time(t2)
				dt = t_gp2-t_gp1
				if abs(dt.sec)<500 and abs(dt.sec)>5e-5:
					try:	
						gp1 = GP_data(fn1,t_gp1)
						gp2 = GP_data(fn2,t_gp2)
						c = get_correlation_coefficients(gp1,gp2)
						tvalues.append(abs(dt.sec))
						cvalues.append(c[0])
						pulse_times ='scan{}:{},scan{}:{}'.format(scan_no1,t1,scan_no2,t2)
						statement = 'The correlation coefficient is {}, the time lag between the two pulses is {} seconds.'.format(c[0],dt.sec)
						print pulse_times
						print statement
						f1.write(pulse_times+'\n'+statement+'\n') 
				
					except AssertionError:
						print 'Assertion error:',scan_no1,t1,scan_no2,t2
						continue
				
	f=open('./figures/correlation_coeff/tvalues1.txt','w')
	dump(tvalues,f)
	f.close()
	f=open('./figures/correlation_coeff/cvalues1.txt','w')
	dump(cvalues,f)
	f.close()

#calculate_values(int(text_name[5])) #for reading scan_.txt files
calculate_values(4)

plt.figure()
plt.semilogx(tvalues,cvalues,'bo')
plt.xlabel('time lag (seconds)')
plt.ylabel('Correlation Coefficient')
plt.show()
