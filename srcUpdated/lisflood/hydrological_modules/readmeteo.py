"""

Copyright 2019 European Union

Licensed under the EUPL, Version 1.2 or as soon they will be approved by the European Commission  subsequent versions of the EUPL (the "Licence");

You may not use this work except in compliance with the Licence.
You may obtain a copy of the Licence at:

https://joinup.ec.europa.eu/sites/default/files/inline-files/EUPL%20v1_2%20EN(1).txt

Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an "AS IS" basis,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the Licence for the specific language governing permissions and limitations under the Licence.

"""
from __future__ import print_function, absolute_import

from ..global_modules.settings import LisSettings,inttodate,calendar,datetoint
from ..global_modules.add1 import readmapsparse
from ..global_modules.netcdf import xarray_reader
import numpy as np
import random
import os
from scipy.ndimage import gaussian_filter


class readmeteo(object):

    """
     # ************************************************************
     # ***** READ METEOROLOGICAL DATA              ****************
     # ************************************************************
    """

    def __init__(self, readmeteo_variable):
        self.var = readmeteo_variable
        settings = LisSettings.instance()
        option = settings.options
        binding  = settings.binding

        
        # initialise xarray readers
        self.var.trace  = []   # will store tuples (model_step, forcing_idx)
        if option['readNetcdfStack']:
            self.forcings = {}
            for data in ['PrecipitationMaps', 'TavgMaps', 'ET0Maps', 'E0Maps']:
                self.forcings[data] = xarray_reader(data)

            self.var.global_run_start = datetoint(binding['StepStart'], binding=binding)[0]
            

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
    def dynamic(self):
        """ dynamic part of the readmeteo module
            read meteo input maps
        """
        settings = LisSettings.instance()
        option = settings.options
        binding = settings.binding
        
        # ************************************************************
        # ***** READ METEOROLOGICAL DATA *****************************
        # ************************************************************
        if option['readNetcdfStack']:
            

            current_sample = self.var.currentSampleNumber()
            
            absolute_run_start = self.var.global_run_start
            
            step = self.var.currentTimeStep() - absolute_run_start


            # Read from NetCDF stack files
            self.var.Precipitation = self.forcings['PrecipitationMaps'][step] * self.var.DtDay * self.var.PrScaling
            self.var.Tavg = self.forcings['TavgMaps'][step]
            self.var.ETRef = self.forcings['ET0Maps'][step] * self.var.DtDay * self.var.CalEvaporation
            self.var.EWRef =self.forcings['E0Maps'][step] * self.var.DtDay * self.var.CalEvaporation
            
            
            
            # ************************************************************************************************
            
        else:
            # Read from stack of maps in Pcraster format
            self.var.Precipitation = readmapsparse(binding['PrecipitationMaps'], self.var.currentTimeStep(), self.var.Precipitation) * self.var.DtDay * self.var.PrScaling
            # precipitation (conversion to [mm] per time step)
            self.var.Tavg = readmapsparse(binding['TavgMaps'], self.var.currentTimeStep(), self.var.Tavg)
            # average DAILY temperature (even if you are running the model on say an hourly time step) [degrees C]
            self.var.ETRef = readmapsparse(binding['ET0Maps'], self.var.currentTimeStep(), self.var.ETRef) * self.var.DtDay * self.var.CalEvaporation
            # daily reference evapotranspiration (conversion to [mm] per time step)
            # potential evaporation rate from a bare soil surface (conversion to [mm] per time step)
            self.var.EWRef = readmapsparse(binding['E0Maps'], self.var.currentTimeStep(), self.var.EWRef) * self.var.DtDay * self.var.CalEvaporation
            # potential evaporation rate from water surface (conversion to [mm] per time step)
    
        # -----------------------------------------Precipitation perturbation----------------------------------------------------------------------------------------------------------------------------
        
        
        
        # 4. Perturbation parameters (spatial + temporal correlation, lognormal)
        # ---------------------------------------------
        spatial_corr_length_m = 15000  # 1 km correlation length
        grid_resolution_m = 5000      # 11 km per grid cell
        sigma_spatial = spatial_corr_length_m / grid_resolution_m  # ≈0.0909

        rho_temporal = 0.6             # AR(1) coefficient
        c_v = 0.60       # CV = 20%

        # Derive lognormal parameters
        sigma_logn = np.sqrt(np.log(c_v**2 + 1.0))
        mu_logn = -0.5 * sigma_logn**2
        
        # ---------------------------------------------
        # 5. Generate 24 ensembles, record perturbed TS for all stations
        #    pert_ts shape: (n_ensembles, n_time, n_stations)
        # ---------------------------------------------
        #pert_ts = np.zeros((n_ensembles, n_time, n_stations))
        
        
        base_map = self.var.Precipitation
                
        # (a) draw IID Gaussian noise, apply spatial smoothing
        rand_norm = np.random.randn(*self.var.Precipitation.shape)
        spatial_corr = gaussian_filter(rand_norm, sigma=sigma_spatial, mode="reflect")
        spatial_corr /= np.std(spatial_corr)
                
        # (b) AR(1) in time
        if self.var.prev_gauss is None:
            curr_gauss = spatial_corr.copy()
        else:
            curr_gauss = rho_temporal * self.var.prev_gauss + np.sqrt(1 - rho_temporal**2) * spatial_corr
                
        # (c) lognormal multiplier
        multiplier = np.exp(mu_logn + sigma_logn * curr_gauss)
                
        # (d) apply perturbation and record for each station
        self.var.Precipitation = np.maximum(base_map * multiplier,0)
        self.var.prev_gauss = curr_gauss
        #print ('curr_gauss: ', curr_gauss )
        
        
        # Generate perturbed temperature data with gaussian noise
        
        def generate_gaussian_temperature(tmp_data, mean, noise_std):
            """
       
            
            :param precip_data: Original precipitation data (numpy array)
            :param scale_factors: List of factors to scale the precipitation data (e.g., [0.8, 1.2] for ±20%)
            :param noise_std: Standard deviation of the normal noise to add to the precipitation data
            :return: List of perturbed precipitation datasets
            """
            
            #scale_factor = round(np.random.uniform(scale_factors[0], scale_factors[1]),2)
            noise = np.random.normal(loc=mean, scale=noise_std, size=tmp_data.shape)
            perturbed_tmp = np.maximum(tmp_data + noise,0)
            return perturbed_tmp

        
        
        #scale_factors = [0,3]  # ±25% scaling
        mean = 0
        noise_std = 1
        self.var.Tavg = generate_gaussian_temperature(self.var.Tavg, mean, noise_std)
    
        
        # ---------------------------

        self.var.ESRef = (self.var.EWRef + self.var.ETRef)/2
        

        if option['TemperatureInKelvin']:
            self.var.Tavg -= 273.15
        
