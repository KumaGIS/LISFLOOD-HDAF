
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
from __future__ import absolute_import

from pcraster.framework import MonteCarloModel
import numpy as np
from pcraster import *
from pcraster.framework import *

from .global_modules.zusatz import DynamicModel, EnKfModel
from .global_modules.settings import MaskInfo
from .global_modules.add1 import *


class LisfloodModel_EnKF(DynamicModel, MonteCarloModel, EnKfModel):

    """ LISFLOOD initial part
        same as the PCRaster script -initial-
        this part is to initialize the variables
        it will call the initial part of the hydrological modules
    """

    def __init__(self):
        """ init part of the initial part
            defines the mask map and the outlet points
            initialization of the hydrological modules
        """
        DynamicModel.__init__(self)
        MonteCarloModel.__init__(self)
        EnKfModel.__init__(self)

    def setState(self):
        
        ## ******************************************* NEW CODE ************************************************************
        
        """Return the current discharge state vector (ChanQAvg) for this ensemble member."""
        
        maskinfo = MaskInfo.instance()
        sample = str(self.currentSampleNumber())
        
        names = ["ChanQAvg","W1aR","W1bR","W2R","UZR"]
        state_vars = []
        
        # 1) Collect all state variables + Sanity check
        for nm in names:
            v = loadObject(nm, sample)
            assert v.ndim == 1, f"{nm} not 1‑D"
            assert v.shape[0] == maskinfo.info.mapC[0],  f"{nm} length {v.shape[0]} ≠ {maskinfo.info.mapC[0]}"
            assert np.isfinite(v).all(), f"{nm} has non‑finite entries"
            state_vars.append(v)
        
        # 2) Concatenate all state vectors
        state_vector = np.concatenate(state_vars)
        
        #print(f"[EnKF][setState] Loaded {len(names)} vars, each size={maskinfo.info.mapC[0]}, total={state_vector.size}")
        
        return state_vector
    
   
    def setObservations(self):
        
        # 1) extract the filter time step
        timestep = self.currentTimeStep()
        # set the scenario
        scenario = 'A'
        # 2) read the observation data 
        obs_map = readmap(generateNameT("A_Q", timestep))
        
        obs_locs = {
            'A': [(7,11), (20,25)],
            'B': [(7,11), (14,14), (14,11)],
            'C': [(7,11), (17,15), (20,25), (22,25)],
            'D': [(18,23), (20,25), (23,23), (22,25)],
            'E': [(17,13), (17,15), (18,23), (23,23), (22,25)],
            'F': [(7,11), (14,11), (18,23), (20,25), (23,23)],
            'G': [(14,14), (17,13), (17,15), (18,23), (20,25), (23,23), (22,25)],
            'H': [(7,11), (14,14), (14,11), (17,13), (17,15), (18,23), (20,25), (23,23), (22,25)]}[scenario]
        
        
        m = len(obs_locs)            # 7 observations
        N = self.nrSamples()  # Number of ensemble members
        
        # 3) Extract raw obs values (1-based)
        grid   = pcr2numpy(obs_map, np.nan)
        #values = np.array([cellvalue(obs_map, r, c)[0] for (r, c) in obs_locs])              # 1-based
        values = np.array([ grid[r-1, c-1] for (r, c) in obs_locs ])                            # 0-based
        
        
        # 4) Build perturbed observation matrix D (m×N)
        obs_error_frac = 0.10                  # 8% error, for example
        sigma          = obs_error_frac * values
        noise          = np.random.randn(m, N) * sigma[:, None]
        observations   = values[:, None] + noise
        observations   = np.nan_to_num(observations, nan=0.0)
        
        # 5) Build diagonal covariance R (m×m)
        covariance = np.diag(sigma**2)
        
        
        # 6) after building 'observations' and 'covariance'
        observations = np.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
        covariance   = np.nan_to_num(covariance, nan=1e-6, posinf=1e6, neginf=1e-6)

        assert np.all(np.isfinite(observations)), "NaN in observation matrix"
        assert np.all(np.isfinite(covariance)),   "NaN in R matrix"
        
        ## ------------------------------ Define Measurement Operator Matrix H -------------------------------------------------
        # 6) Build H (m×n_state) via MaskInfo
        maskinfo      = MaskInfo.instance()
        maskflat      = maskinfo.info.maskflat    # bool array, length = full_grid_size :contentReference[oaicite:0]{index=0}
        
        M      = maskinfo.info.mapC[0]       # number of active pixels (n_state)
        K        = 5                         # e.g. [ChanQAvg, W1aR, W1bR, W2R, UZR]
        n_state  = M * K
        
        comp_idx    = np.where(maskflat == False)[0]   # compressed→full-grid index map, Here 'False' gives the active cells where 'True' extract the outside region
        rows, cols = maskinfo.info.shape
        
        
        # 7) flatten full-grid indices for each obs location
        obs_flat = [ (r-1)*cols + (c-1) for (r, c) in obs_locs ]
        idx_map     = {f:i for i, f in enumerate(comp_idx)}
        
        # 8) Check for invalid entries
        invalid = []
        for i, flat in enumerate(obs_flat):
            if flat not in idx_map:
                r, c = obs_locs[i]
                invalid.append((i, (r, c), flat))
                
        if invalid:
            msg = "The following observation locations lie outside the domain mask:\n"
            for idx, (r, c), flat in invalid:
                msg += f" • obs #{idx} at 1-based (row={r},col={c}) → flat_index={flat}\n"
            raise RuntimeError(msg)
        
        
        obs_state_idx = [ idx_map[f] for f in obs_flat ]
        
        # only discharge lives in entries [0:M], so H[i, comp_idx] = 1
        
        # 9) assemble H
        H = np.zeros((m, n_state), dtype=float)
        for i, ci in enumerate(obs_state_idx):
            H[i, ci] = 1.0
            
        # DEBUG
        # 1) Shape check
        assert H.shape == (m, n_state), \
            f"H must be {m}×{n_state}, got {H.shape}"
        
        # 2) Row-sum and entry-values check
        row_sums = H.sum(axis=1)
        assert np.all(row_sums == 1), \
            "Each H row should sum to exactly 1"
        assert set(np.unique(H)) <= {0.0, 1.0}, \
            "H must contain only 0s and 1s"
        
        # 3) (Optional) Spot-check that the '1's are where you expect:
        for i, (r, c) in enumerate(obs_locs):
            # compute full-grid index and then compressed index
            full_idx = (r-1)*cols + (c-1)
            cmp_idx = comp_idx.tolist().index(full_idx)
            assert H[i, cmp_idx] == 1.0, \
                f"H[{i}, {cmp_idx}] should be 1.0 for obs at ({r},{c})"
            
        # ─── hand off to EnKF ───────────────────────────────────────────────
        # 10) Returning the observations, covariance and the measurement operator to the model
        self.setObservedMatrices(observations, covariance)       # store obs & R :contentReference[oaicite:1]{index=1}
        self.setMeasurementOperator(H)       # store H

        
    def resume(self):
        
        """
        After EnKF analysis, split the big state-vector into each variable,
        clip negatives, dump them, then do the usual routing update on discharge.
        """
        # ------------------------------------------------------------------------------------------------------------------------------
        # 1) load the forecast we dumped pre‐analysis
        sample = str(self.currentSampleNumber())
        old_ChanQAvg = loadObject("ChanQAvg", sample)
        old_W1a = loadObject("W1aR", sample)
        old_W1b = loadObject("W1bR", sample)
        old_W2 = loadObject("W2R", sample)
        old_UZ = loadObject("UZR", sample)

        # -------------------------------------------------------------------------------------------------------------------------------
        # 2) Get the updated state vectors
        # retrieve the updated state vector
        maskinfo = MaskInfo.instance()
        
        # how many pixels per layer
        M = maskinfo.info.mapC[0]
        
        # list of your state‐var names in the same order used in setState()
        names = ["ChanQAvg","W1aR","W1bR","W2R","UZR"]
        K     = len(names)
        n_state = M * K
        
        # pull the combined analysis vector
        full_vec = self.getStateVector(sample)
        
        #sanity check
        if full_vec.shape != (n_state,):
            raise ValueError(f"Expected analysis vector of length {n_state}, got {full_vec.shape}")  # :contentReference[oaicite:0]{index=0}
        
        
        # 3) slice into individual arrays, clip negatives, dump each
        for i, nm in enumerate(names):
            start = i * M
            end   = (i+1) * M
            arr   = full_vec[start:end]
            assert arr.shape[0] == M,  f"{nm} length {arr.shape[0]} ≠ {M}"
            # clip
            negs = (arr < 0).sum()
            if negs:
                print(f"[EnKF][RESUME] {nm}: clipped {negs} negatives → 0")
                arr = np.clip(arr, 0.0, None)
            dumpObject(f"{nm}_EnKF", arr, sample)
            
        # --------------------------------------------------------------------------------------------------------------------------------------
        # 4) Update all soil states using updated state vector
        
        W1a_new = loadObject("W1aR_EnKF", sample)     # Updated soil layer 1a
        W1b_new = loadObject("W1bR_EnKF", sample)     # Updated soil layer 1b
        W2_new  = loadObject("W2R_EnKF", sample)     # Updated soil layer 2
        
        # pull out the residual & sat. limits for rach soil layer
        res1a = self.soil_module.var.WRes1a[0]
        sat1a = self.soil_module.var.WS1a[0]
        
        #print('res1a shape: ', self.soil_module.var.WRes1a.shape)
        
        res1b = self.soil_module.var.WRes1b[0]
        sat1b = self.soil_module.var.WS1b[0]
        
        res2  = self.soil_module.var.WRes2[0]
        sat2  = self.soil_module.var.WS2[0]
        #print('res2 shape: ', self.soil_module.var.WRes2.shape)
        
        # clamp each vector between [res, sat]
        W1a_new = np.clip(W1a_new, res1a, sat1a)
        W1b_new = np.clip(W1b_new, res1b, sat1b)
        W1_new = W1a_new + W1b_new
        W2_new  = np.clip(W2_new,  res2,  sat2)

        
        # recombine top soil and dump
        dumpObject("W1a_EnKF", W1a_new, sample)
        dumpObject("W1b_EnKF", W1b_new, sample)
        dumpObject("W1_EnKF",  W1_new,  sample)
        dumpObject("W2_EnKF",  W2_new,  sample)
        
        # --------------------------------------------------------------------------------------------------------------------------------------
        # 5) Update all upper GWZone states using updated state vector
        UZ_new = loadObject("UZR_EnKF", sample)
        UZ_new = np.maximum(UZ_new, 0.0)
        dumpObject("UZ_EnKF", UZ_new, sample)
        
        # --------------------------------------------------------------------------------------------------------------------------------------
        # 6) Update all routing states using assimilated discharge for entire domain
        # dump the cleaned, corrected vector
        ChanQAvg_ENKF = loadObject("ChanQAvg_EnKF", sample)
        
        
        # Split routing - update both main and floodplain channels [DOUBLE‐CHANNEL (split kinematic + floodplain)]
        # 6.1. preserve old split‐ratio
        Q1_old = loadObject("ChanQKin", sample)
        Q2_old = loadObject("Chan2QKin",sample)
        ChanQ_old = loadObject("ChanQ", sample)
        Q_limit = self.routing_module.var.QLimit
        
        SumQ   = (ChanQ_old + Q_limit)
        target_sum = (ChanQAvg_ENKF + Q_limit)
        
        # 6.2. Maintain current flow distribution ratios
        ratioQ1 = np.where(SumQ > 0, Q1_old / SumQ, 0.5)
        ratioQ2 = np.where(SumQ > 0, Q2_old / SumQ, 0.5)
        
        
        # 6.3. Update discharge components
        ChanQKin_EnKF = np.maximum(ratioQ1 * target_sum,0)
        Chan2QKin_EnKF = np.maximum(ratioQ2 * target_sum,0)
        
        # sanity‐check and logging
        if ChanQKin_EnKF.shape != Q1_old.shape:
            raise ValueError(f"Shape mismatch: forecast {Q1_old.shape} vs analysis {ChanQKin_EnKF.shape}")
            
        if Chan2QKin_EnKF.shape != Q2_old.shape:
            raise ValueError(f"Shape mismatch: forecast {Q2_old.shape} vs analysis {Chan2QKin_EnKF.shape}")
        
        # Starting to dump after filter step  
        dumpObject("ChanQKin_EnKF", ChanQKin_EnKF, sample)
        dumpObject("Chan2QKin_EnKF", Chan2QKin_EnKF, sample)
        
        
        # 6.4. invert to volumes
        
        V1_old = loadObject("ChanM3Kin", sample)
        V2_old = loadObject("Chan2M3Kin", sample)
        
        ChanM3Kin_EnKF = self.routing_module.var.ChanLength * self.routing_module.var.ChannelAlpha * (ChanQKin_EnKF ** self.routing_module.var.Beta)
        ChanM3Kin_EnKF = np.maximum(ChanM3Kin_EnKF, 0.0)
        
        Chan2M3Kin_EnKF = self.routing_module.var.ChanLength * self.routing_module.var.ChannelAlpha2 * (Chan2QKin_EnKF ** self.routing_module.var.Beta)
        Chan2M3Kin_EnKF = np.maximum(Chan2M3Kin_EnKF, 0.0)
        
        # sanity‐check and logging
        if ChanM3Kin_EnKF.shape != V1_old.shape:
            raise ValueError(f"Shape mismatch: forecast {V1_old.shape} vs analysis {ChanM3Kin_EnKF.shape}")
            
        if Chan2M3Kin_EnKF.shape != V2_old.shape:
            raise ValueError(f"Shape mismatch: forecast {V2_old.shape} vs analysis {Chan2M3Kin_EnKF.shape}")
            
    
        # Starting to dump after filter step  
        dumpObject("ChanM3Kin_EnKF", ChanM3Kin_EnKF, sample)
        dumpObject("Chan2M3Kin_EnKF", Chan2M3Kin_EnKF, sample)
    
        # 6.5. Update auxiliary variables
        
        ChanQ_EnKF = np.maximum((ChanQKin_EnKF + Chan2QKin_EnKF - self.routing_module.var.QLimit), 0.0)
        
        if ChanQ_EnKF.shape != ChanQ_old.shape:
            raise ValueError(f"Shape mismatch: forecast {ChanQ_old.shape} vs analysis {ChanQ_EnKF.shape}")
            
        dumpObject("ChanQ_EnKF", ChanQ_EnKF, sample)
    
        # 7) let stateVar.resume() read & apply it wholesale
        self.stateVar_module.resume()
        
         
    
    
        