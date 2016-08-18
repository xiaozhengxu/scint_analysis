
# coding: utf-8

# In[68]:

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5 

import os


# In[102]:

filelist = []
for file_name in os.listdir('/home/ramain/GPs/GPo8'):
    output,increased_ratio, sig_noise = add_pulses(file_name)
    np.save('./GPs/{}'.format(file_name),output)


# In[108]:

teles = ['ef','jb','wb','hh','tr','o8']

def add_pulses(time_string):
    '''Takes in a list output powers from load_gp (outputs)
    Returns an output of summed pulses as numpy arrays to be saved in my directory, 
    as well as how much the signal to noise is improved'''
    output1,peak_time1,sn1 = load_gp(time_string,teles[0],nchan)
    outputsum = output1 
    for tele in teles[1:]:
        output2,peak_time2,sn2 = load_gp(time_string,tele,nchan)
        output2 = roll(output2,peak_time1-peak_time2,axis = 1)
        outputsum = outputsum+output2
    
    outputsumfreq = outputsum.sum(0)
    sig_noise = get_SN(outputsumfreq)
    peak_time = argmax(outputsumfreq)
    snef = max(sn1)
    snsum = max(sig_noise)
    return outputsum, snsum/snef,sig_noise

gp,isnr,sn = add_pulses('p'+time_string9+'.npy')

print isnr

figure()
imshow(gp)

figure()
plot(sn)


# In[ ]:




# In[12]:

get_ipython().magic(u'pylab inline')
time_string1 = '2015-10-19T00:17:47.415' # S/N = 89!, main pulse 
time_string2 = '2015-10-19T02:35:26.143' # S/N = 43, main pulse 
time_string3 = '2015-10-19T02:06:52.280' #S/N = 7.9, inter pulse
time_string4 = '2015-10-19T02:06:58.031' #S/N = 9.6, main pulse, also a double pulse
time_string5 = '2015-10-19T00:55:15.557' #S/N = 8.33, main pulse
time_string6 = '2015-10-19T00:55:49.673' #S/N = 8.55, main pulse
time_string7 = '2015-10-19T01:29:52.509' #S/N = 8.44, main pulse
time_string8 = '2015-10-19T01:55:06.340' #S/N = 8.66, main pulse
time_string9 = '2015-10-19T02:13:46.551' #S/N = 9.56, main pulse
time_string10 = '2015-10-19T02:36:11.418'#S/N = 8.41, main pulse


# In[81]:

def get_SN(outputsumfreq):
    noise_std = np.std(outputsumfreq[:50]) 
    noise_mean = np.mean(outputsumfreq[:50])
    sigs_noise = (outputsumfreq-noise_mean)/noise_std
    return sigs_noise

def load_gp(time_string,telescope,nchan,draw = 0):
    '''Takes in a time string, a string indicating whcih telescope to use, and an interger of nchan'''
    gp1 = np.load('/home/ramain/GPs/GP{}/{}'.format(telescope,time_string))
    gp1t = np.fft.irfft(gp1,axis=1)
    gp1s = gp1t.reshape(-1,2*nchan,16)
    gp1s = np.fft.rfft(gp1s,axis=1)
    dchan = gp1s
    dR = np.concatenate((dchan[:,::-1,8], dchan[...,0], dchan[:,::-1,10], dchan[...,2], dchan[:,::-1,12], dchan[...,4], dchan[:,::-1,14], dchan[...,6]), axis=1)
    dL = np.concatenate((dchan[:,::-1,9], dchan[...,1], dchan[:,::-1,11], dchan[...,3], dchan[:,::-1,13], dchan[...,5], dchan[:,::-1,15], dchan[...,7]), axis=1)
    output = (abs(dR)**2 + abs(dL)**2).T

    outputsumfreq = output.sum(0) #dedispersed
    #computes signal to noise after summing all the frequencies
    sigs_noise = get_SN(outputsumfreq)
    sn_sorted = np.argsort(sigs_noise)

    peak_time = np.argmax(sigs_noise)
    return output, peak_time, sigs_noise


# In[ ]:




# In[76]:

nchan = 16
tele1 = 'ef'
tele2 = 'o8'
output1,peak_time1,sn1 = load_gp(time_string4,tele1,nchan)
output2,peak_time2,sn2 = load_gp(time_string4,tele2,nchan)

plt.figure()
plt.imshow(output1)
plt.figure()
plt.imshow(output2)

plt.figure(figsize = (10,7))
plt.plot(sn1)
plt.plot(sn2)



# In[77]:

print max(sn1)
print max(sn2)


# In[78]:

print peak_time1,peak_time2


# In[79]:

output2n = roll(output2,peak_time1-peak_time2,axis = 1)
outputsumfreq = output2n.sum(0)
sig_noise = get_SN(outputsumfreq)
peak_time2n = argmax(outputsumfreq)
print peak_time2n

# plt.figure(figsize = (15,10))
# plt.plot(sn1)
# plt.plot(sig_noise)

print output2n.shape


# In[80]:

outputsum = output2n+output1
outputsumfreqs = outputsum.sum(0)
sig_noises = get_SN(outputsumfreqs)
plt.figure(figsize = (15,10))
plt.plot(sig_noises)
print max(sig_noises)

