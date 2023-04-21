import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from astropy_healpix import HEALPix as Hpix
from pathlib import Path
from animation.plot_job import filepath

flux_data   =   filepath/'weekly_flux_maps.pkl'
exposure_data = filepath/'weekly_exposure_maps.pkl'

class FluxLightCurve:
    
    def __init__(self, skycoord, radius=Angle('1d')):
        
        self.skycoord = skycoord if isinstance(skycoord, SkyCoord) else SkyCoord.from_name(skycoord)
        self.radius = radius
        
        from wtlike.config import first_data
        self.flux_series = self._load_cone(flux_data   )
        self.exposure_series = self._load_cone(exposure_data  )
        self.mjd = first_data + np.arange(len(self.flux_series))*7+3.5 
        
    def _load_cone(self,  filename):
        """Extract a time series from the pickled dict at filename,
        the mean for the pixels within radius of skycoord
        Assume keys are week numbers, starting at 1.
        """
        assert Path(filename).is_file(), f'Filename "{filename}" is not a file"'
        with open(filename, 'rb') as inp:
            edict = pickle.load(inp)
        hpix = Hpix(nside=int(np.sqrt(len(edict[0])/12)), frame='galactic')
        pixels = hpix.cone_search_skycoord(self.skycoord, self.radius)
        return np.array(
            [ np.sum(edict[idx+1][pixels]) for idx in range(len(edict)-1)]
         )
        
    def series_plots(self, marker='.', color='maroon'):
        """Series plots of flux and exposure
        """
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(12,6), sharex=True)
        ax1.scatter(self.mjd, self.flux_series, marker=marker, color=color)
        ax1.set(ylim=(0, None), ylabel='Flux')
        ax2.scatter(self.mjd, self.exposure_series, marker=marker, color=color)
        ax2.set(ylim=(0,None), ylabel='Exposure', xlabel='MJD')
        return fig
    
    def plot_position(self):
        from wtlike.skymaps import AitoffFigure

        afig = AitoffFigure(figsize=(6,3))
        afig.fig.set_facecolor((0,0,0))
        afig.ax.set_facecolor('lavender')
        afig.scatter(self.skycoord, marker='*', s=200, c='red')
        return afig.fig
    
    def flux_vs_exposure(self, ax=None, scat_kw={}, **kwargs):
        fig, ax = plt.subplots(figsize=(5,5)) if ax is None else (ax.figure, ax)
        x,y = self.exposure_series, self.flux_series
        ymean = np.nanmean(y)
        good = y>0.2*ymean
        x,y = x[good], y[good] # remove small flux
        ax.scatter(x, y, **scat_kw)
        ax.axhline(np.nanmean(y), color='grey')
        kw = dict( xlabel='exposure', ylabel='flux')
        kw.update(kwargs)
        ax.set(**kw)
        return fig
        