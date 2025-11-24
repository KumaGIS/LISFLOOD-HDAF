
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
        # 1) Load current ChanQAvg state vector, Which is already a 1D numpy array of length = number of active pixels
        maskinfo = MaskInfo.instance()
        sample = str(self.currentSampleNumber())
        
        state_vector = loadObject("ChanQAvg", sample)
        
        # 2) Any positives?
        has_positive = (state_vector > 0).any()
        
        
        # 3) Sanity check: must match the compressed mask size
        assert state_vector.shape[0] == maskinfo.info.mapC[0], (f"Expected state size {maskinfo.info.mapC[0]}, got {state_vector.shape[0]}")

        # sanity check (optional)
        #assert max_val >= 0, "Unexpected negative maximum in state vector"
        # 4) Return the state vector
        return state_vector
        
        

    def setObservations(self):
        
        # 1) Load the observation data according to the current scenario
        # extract the filter time step
        timestep = self.currentTimeStep()
        # set the scenario
        scenario = 'A'
        # read the observation data 
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
        
        
        # 2) Extract raw obs values (1-based)
        grid   = pcr2numpy(obs_map, np.nan)
        #values = np.array([cellvalue(obs_map, r, c)[0] for (r, c) in obs_locs])              # 1-based
        values = np.array([ grid[r-1, c-1] for (r, c) in obs_locs ])                            # 0-based
        
        
        # 3) Build perturbed observation matrix D (m×N)
        obs_error_frac = 0.12                   # 8% error, for example
        sigma          = obs_error_frac * values
        noise          = np.random.randn(m, N) * sigma[:, None]
        observations   = values[:, None] + noise
        observations   = np.nan_to_num(observations, nan=0.0)
        
        # 4) Build diagonal covariance R (m×m)
        covariance = np.diag(sigma**2)
        
        
        # 5) after building 'observations' and 'covariance'
        observations = np.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
        covariance   = np.nan_to_num(covariance, nan=1e-6, posinf=1e6, neginf=1e-6)

        assert np.all(np.isfinite(observations)), "NaN in observation matrix"
        assert np.all(np.isfinite(covariance)),   "NaN in R matrix"
        
        ## ------------------------------ Define Measurement Operator Matrix H -------------------------------------------------
        # 6) Build H (m×n_state) via MaskInfo
        maskinfo      = MaskInfo.instance()
        maskflat      = maskinfo.info.maskflat    # bool array, length = full_grid_size :contentReference[oaicite:0]{index=0}
        comp_idx    = np.where(maskflat == False)[0]   # compressed→full-grid index map
        full_rows, full_cols = maskinfo.info.shape
        
        n_state       = maskinfo.info.mapC[0]     # number of active pixels
        
        # flatten full-grid indices for each obs location
        obs_flat = [ (r-1)*full_cols + (c-1) for (r, c) in obs_locs ]
        idx_map     = {f:i for i, f in enumerate(comp_idx)}
        
        # 7) Check for invalid entries
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
        
        # map each to compressed index
        #obs_comp_idx = [ int(np.where(comp_indices == idx)[0][0]) for idx in obs_full_idx ]
        
        # 8) assemble H
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
            full_idx = (r-1)*full_cols + (c-1)
            cmp_idx = comp_idx.tolist().index(full_idx)
            assert H[i, cmp_idx] == 1.0, \
                f"H[{i}, {cmp_idx}] should be 1.0 for obs at ({r},{c})"
            
        # ─── hand off to EnKF ───────────────────────────────────────────────
        # 9) Return the observation matrix, obs_covariance matrix and Measurement operator matrix
        self.setObservedMatrices(observations, covariance)       # store obs & R :contentReference[oaicite:1]{index=1}
        self.setMeasurementOperator(H)       # store H

    def resume(self):
        
        """
       Called after the filter writes its analysis.  We:
         1) load the original forecast dump (for debug),
         2) pull in the analysis update, and
         3) sanity‐check shapes & print stats,
         4)  check for any negative values, if there are any convert them to 0.
         5) dump the corrected vector under a new key,
         6) invoke stateVar.resume() to apply it.
       """
        
        # retrieve the updated state vector
        sample = str(self.currentSampleNumber())
        # 2) get the filter’s analysis vector
        updateVec = self.getStateVector(sample)
        
        # 1) load the forecast we dumped pre‐analysis
        old_vec = loadObject("ChanQAvg", sample)
        
        print(f"[EnKF][RESUME] sample={sample}  old ChanQAvg stats --> "
              f"min={updateVec.min():.3f}, max={updateVec.max():.3f}, mean={updateVec.mean():.3f}")
        
        
        
        # 3) sanity‐check and logging
        if updateVec.shape != old_vec.shape:
            raise ValueError(f"Shape mismatch: forecast {old_vec.shape} vs analysis {updateVec.shape}")
        
        # 4) clip any unphysical negatives to zero
        num_neg = (updateVec < 0).sum()
        if num_neg:
            print(f"[EnKF][RESUME] sample={sample}  clipped {num_neg} negative values → 0")
            updateVec = np.clip(updateVec, 0.0, None)
            
        print(f"[EnKF][RESUME] sample={sample}  post ChanQAvg stats --> "
              f"min={updateVec.min():.3f}, max={updateVec.max():.3f}, mean={updateVec.mean():.3f}")
        
        # 5) dump the cleaned, corrected vector
        ChanQAvg_ENKF = updateVec
        
        # Starting to dump after filter step   
        dumpObject("ChanQAvg_EnKF", ChanQAvg_ENKF, sample)
        
        # ****************************** Update all routing states using assimilated discharge for entire domain *******************************
        
        # Split routing - update both main and floodplain channels [DOUBLE‐CHANNEL (split kinematic + floodplain)]
        # 1. preserve old split‐ratio
        Q1_old = loadObject("ChanQKin", sample)
        Q2_old = loadObject("Chan2QKin",sample)
        ChanQ_old = loadObject("ChanQ", sample)
        Q_limit = self.routing_module.var.QLimit
        
        SumQ   = (ChanQ_old + Q_limit)
        target_sum = (ChanQAvg_ENKF + Q_limit)
        
        # 2. Maintain current flow distribution ratios
        ratioQ1 = np.where(SumQ > 0, Q1_old / SumQ, 0.5)
        ratioQ2 = np.where(SumQ > 0, Q2_old / SumQ, 0.5)
        
        
        # 3. Update discharge components
        ChanQKin_EnKF = np.maximum(ratioQ1 * target_sum,0)
        Chan2QKin_EnKF = np.maximum(ratioQ2 * target_sum,0)
        
        # sanity‐check and logging
        if ChanQKin_EnKF.shape != Q1_old.shape:
            raise ValueError(f"Shape mismatch: forecast {Q1_old.shape} vs analysis {ChanQKin_EnKF.shape}")
            
        if Chan2QKin_EnKF.shape != Q2_old.shape:
            raise ValueError(f"Shape mismatch: forecast {Q2_old.shape} vs analysis {Chan2QKin_EnKF.shape}")
        
              
        # Starting to dump sub step instantaneous discharge components after filter step  
        dumpObject("ChanQKin_EnKF", ChanQKin_EnKF, sample)
        dumpObject("Chan2QKin_EnKF", Chan2QKin_EnKF, sample)
        
        
        # 4. invert to volumes
        
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
            
    
        # Starting to dump sub step volume components after filter step  
        dumpObject("ChanM3Kin_EnKF", ChanM3Kin_EnKF, sample)
        dumpObject("Chan2M3Kin_EnKF", Chan2M3Kin_EnKF, sample)
    
        # 5. Update auxiliary variables
        
        ChanQ_EnKF = np.maximum((ChanQKin_EnKF + Chan2QKin_EnKF - self.routing_module.var.QLimit), 0.0)
        
        if ChanQ_EnKF.shape != ChanQ_old.shape:
            raise ValueError(f"Shape mismatch: forecast {ChanQ_old.shape} vs analysis {ChanQ_EnKF.shape}")
            
        dumpObject("ChanQ_EnKF", ChanQ_EnKF, sample)
    
    
        # 6) let stateVar.resume() read & apply it wholesale
        self.stateVar_module.resume()