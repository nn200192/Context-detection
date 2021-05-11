import matplotlib.pyplot as plt
import numpy as np



fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()

mu, sigma = 56, 2 # mean and standard deviation
s = np.random.default_rng(seed=1).normal(mu, sigma, 1000)
np.random.shuffle(s)
count, bins, ignored = ax1.hist(s, 30, density=True)
ax1.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')
ax1.set_title('Speed')
ax1.set_xlabel('m/s')
ax1.set_ylabel('Distribution')


mu, sigma = 85, 4 # mean and standard deviation
s = np.random.default_rng(seed=1).normal(mu, sigma, 1000)
np.random.shuffle(s)
count, bins, ignored = ax2.hist(s, 30, density=True)
ax2.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')
ax2.set_title('Heart rate')
ax2.set_xlabel('BPM')
ax2.set_ylabel('Distribution')



mu, sigma = 0, 1 # mean and standard deviation
s = np.random.default_rng(seed=1).normal(mu, sigma, 1000)
np.random.shuffle(s)
count, bins, ignored = ax3.hist(s, 30, density=True)
ax3.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')
ax3.set_title('Hand Acceleration')
ax3.set_xlabel('Magnitude (m/s^2)')
ax3.set_ylabel('Distribution')



mu, sigma = 0, 0.05 # mean and standard deviation
s = np.random.default_rng(seed=1).normal(mu, sigma, 1000)
np.random.shuffle(s)
count, bins, ignored = ax4.hist(s, 30, density=True)
ax4.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')
ax4.set_title('Hand Amplitude')
ax4.set_xlabel('m')
ax4.set_ylabel('Distribution')



mu, sigma = 8, 0.2 # mean and standard deviation
s = np.random.default_rng(seed=1).normal(mu, sigma, 500)
np.random.shuffle(s)
count, bins, ignored = ax5.hist(s, 30, density=True)
ax5.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')
ax5.set_title('Time of Day (Morning)')
ax5.set_xlabel('Time')
ax5.set_ylabel('Distribution')


mu, sigma = 17.5, 0.2 # mean and standard deviation
s = np.random.default_rng(seed=1).normal(mu, sigma, 500)
np.random.shuffle(s)
count, bins, ignored = ax6.hist(s, 30, density=True)
ax6.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')
ax6.set_title('Time of Day (Afternoon)')
ax6.set_xlabel('Time')
ax6.set_ylabel('Distribution')

plt.show()
