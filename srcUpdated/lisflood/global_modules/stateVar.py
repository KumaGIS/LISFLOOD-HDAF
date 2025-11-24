"""
Copyright 2019 European Union

Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
the European Commission – subsequent versions of the EUPL (the "Licence");

You may not use this work except in compliance with the Licence.
You may obtain a copy of the Licence at:

https://joinup.ec.europa.eu/sites/default/files/inline-files/EUPL%20v1_2%20EN(1).txt

Unless required by applicable law or agreed to in writing, software distributed
under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
See the Licence for the specific language governing permissions and limitations
under the Licence.
"""

from __future__ import absolute_import

from lisflood.global_modules.settings import CDFFlags
from .add1 import *

# ============================================================================
#  CONFIGURATION FLAG
# ============================================================================
# Set this flag according to the EnKF configuration you use:
#
#  - STATE_AUGMENTATION = True
#       Use with Lisflood_EnKF_StateAugment.py
#       State vector includes:
#           ChanQAvg, UZR, W1aR, W1bR, W2R  (root–zone pixels)
#
#  - STATE_AUGMENTATION = False
#       Use with Lisflood_EnKF_NoStateAugment.py
#       State vector includes:
#           ChanQAvg only (discharge–only EnKF)
#
STATE_AUGMENTATION = True
# STATE_AUGMENTATION = False


class stateVar(object):
    """
    Reporting and restoring LISFLOOD state variables at EnKF filter steps.
    This class is called from the routing / state-variable modules.
    """

    def __init__(self, stateVar_variable):
        self.var = stateVar_variable

    # ------------------------------------------------------------------ #
    def dynamic(self):
        """
        Dump model states at EnKF filter steps.

        The set of dumped variables depends on the STATE_AUGMENTATION flag:
          - Common part: snow, basic soil moisture diagnostics, inflow/
            precipitation scaling, groundwater LZ, routing vars, lakes,
            overland, reservoirs, TSS, CDF flags, etc.
          - With state augmentation:
              additionally dump root-zone soil water and UZ for the
              "root" land-use class only (index 0) plus the other classes
              so they can be restored.
          - Without state augmentation:
              dump full W1a / W1b / W1 / W2 / UZ arrays as in the
              original LISFLOOD state-saving routine.
        """
        settings = LisSettings.instance()
        option = settings.options
        filter_steps = settings.filter_steps

        try:
            EnKFset = option['EnKF']
        except Exception:
            EnKFset = 0

        if not (EnKFset and self.var.currentTimeStep() in filter_steps):
            return

        sample = str(self.var.currentSampleNumber())
        dumpObject("StartDate", self.var.CalendarDayStart, sample)

        print(f"Starting to dump at filter step {self.var.currentTimeStep()}")

        # ------------------------------------------------------------------
        # Snow
        # ------------------------------------------------------------------
        dumpObject("SnowCover", self.var.SnowCoverS, sample)

        # ------------------------------------------------------------------
        # Soil Moisture / Land-surface stores
        # ------------------------------------------------------------------
        dumpObject("Intercept", self.var.Interception, sample)
        dumpObject("CumIntercept", self.var.CumInterception, sample)
        dumpObject("Frost", self.var.FrostIndex, sample)
        dumpObject("DSLR", self.var.DSLR, sample)

        if STATE_AUGMENTATION:
            # --- State-augmentation version (root-zone variables only) ----
            # Root land-use class (index 0) – variables used in the EnKF
            dumpObject("W1aR", self.var.W1a[0], sample)
            dumpObject("W1bR", self.var.W1b[0], sample)
            dumpObject("W1R",  self.var.W1[0],  sample)
            dumpObject("W2R",  self.var.W2[0],  sample)

            # Other land-use classes – stored so that they can be restored
            dumpObject("W1aForest", self.var.W1a[1], sample)
            dumpObject("W1aIrr",    self.var.W1a[2], sample)
            dumpObject("W1bForest", self.var.W1b[1], sample)
            dumpObject("W1bIrr",    self.var.W1b[2], sample)
            dumpObject("W1Forest",  self.var.W1[1],  sample)
            dumpObject("W1Irr",     self.var.W1[2],  sample)
        else:
            # --- Discharge-only EnKF – store full arrays -------------------
            dumpObject("W1a", self.var.W1a, sample)
            dumpObject("W1b", self.var.W1b, sample)
            dumpObject("W1",  self.var.W1,  sample)
            dumpObject("W2",  self.var.W2,  sample)

        # ------------------------------------------------------------------
        # Inflow & precipitation scaling factors
        # ------------------------------------------------------------------
        dumpObject("InflowScaling", self.var.InfScaling, sample)
        dumpObject("PrScaling",     self.var.PrScaling, sample)

        # ------------------------------------------------------------------
        # Groundwater
        # ------------------------------------------------------------------
        dumpObject("LZ", self.var.LZ, sample)

        if STATE_AUGMENTATION:
            # Store UZ for each land-use class separately
            dumpObject("UZR",        self.var.UZ[0], sample)
            dumpObject("UZForest",   self.var.UZ[1], sample)
            dumpObject("UZIrrigation", self.var.UZ[2], sample)
        else:
            # Store the full UZ array
            dumpObject("UZ", self.var.UZ, sample)

        # ------------------------------------------------------------------
        # Lakes
        # ------------------------------------------------------------------
        if option.get("simulateLakes", False):
            dumpObject("LakeStorageM3", self.var.LakeStorageM3CC, sample)
            dumpObject("LakeOutflow",   self.var.LakeOutflow, sample)

        # ------------------------------------------------------------------
        # Routing
        #   NOTE: these are *forecast* routing states; EnKF-updated routing
        #   fields are written with the *_EnKF suffix by the EnKF module
        #   and read during `resume()`.
        # ------------------------------------------------------------------
        dumpObject("ChanM3Kin",    self.var.ChanM3Kin, sample)
        dumpObject("ChanQKin",     self.var.ChanQKin, sample)
        dumpObject("ChanQ",        self.var.ChanQ, sample)
        dumpObject("ToChan",       self.var.ToChanM3RunoffDt, sample)
        dumpObject("ChanQAvg",     self.var.ChanQAvg, sample)
        dumpObject("CrossSection2", self.var.CrossSection2Area, sample)

        try:
            dumpObject("Chan2M3Kin",   self.var.Chan2M3Kin, sample)
            dumpObject("Chan2QKin",    self.var.Chan2QKin, sample)
            dumpObject("Sideflow1Chan", self.var.Sideflow1Chan, sample)
            dumpObject("QInM3Old",     self.var.QInM3Old, sample)
            dumpObject("QLimit",       self.var.QLimit, sample)
            dumpObject("Chan2M3Start", self.var.Chan2M3Start, sample)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Overland flow
        # ------------------------------------------------------------------
        dumpObject("OFM3Direct", self.var.OFM3Direct, sample)
        dumpObject("OFM3Other",  self.var.OFM3Other, sample)
        dumpObject("OFM3Forest", self.var.OFM3Forest, sample)

        # ------------------------------------------------------------------
        # Reservoirs
        # ------------------------------------------------------------------
        if option.get("simulateReservoirs", False):
            dumpObject("ReservoirStorageM3CC", self.var.ReservoirStorageM3CC, sample)
            dumpObject("ReservoirFill",        self.var.ReservoirFill, sample)
            dumpObject("QResOutM3Dt",         self.var.QResOutM3Dt, sample)

        # ------------------------------------------------------------------
        # Suspended sediment (if available)
        # ------------------------------------------------------------------
        try:
            dumpObject("Tss", self.var.Tss, sample)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # NetCDF flag object
        # ------------------------------------------------------------------
        cdfflags = CDFFlags.instance()
        dumpObject("cdfFlag", cdfflags, sample)

    # ------------------------------------------------------------------ #
    def resume(self):
        """
        Restore model states at EnKF filter steps.

        Reads:
          - Common variables from dumpObject(...)
          - If STATE_AUGMENTATION: updated soil-moisture and UZ states
            from *_EnKF files written by the EnKF module, and restores
            non-augmented classes from their dedicated files.
          - If not STATE_AUGMENTATION: updated routing states only;
            soil-moisture and UZ are restored from their original dumps.
        """
        sample = str(self.var.currentSampleNumber())
        updateVec = self.var.getStateVector(sample)  # not used, but keeps API
        self.var.CalendarDayStart = loadObject("StartDate", sample)

        # ------------------------------------------------------------------
        # Snow
        # ------------------------------------------------------------------
        self.var.SnowCoverS = loadObject("SnowCover", sample)

        # ------------------------------------------------------------------
        # Soil Moisture / land-surface stores
        # ------------------------------------------------------------------
        self.var.Interception    = loadObject("Intercept", sample)
        self.var.CumInterception = loadObject("CumIntercept", sample)
        self.var.FrostIndex      = loadObject("Frost", sample)
        self.var.DSLR            = loadObject("DSLR", sample)

        if STATE_AUGMENTATION:
            # Updated root-zone layers from EnKF
            self.var.W1a[0] = loadObject("W1a_EnKF", sample)
            self.var.W1b[0] = loadObject("W1b_EnKF", sample)
            self.var.W1[0]  = loadObject("W1_EnKF",  sample)
            self.var.W2[0]  = loadObject("W2_EnKF",  sample)

            # Restore other land-use classes
            self.var.W1a[1] = loadObject("W1aForest", sample)
            self.var.W1a[2] = loadObject("W1aIrr",    sample)
            self.var.W1b[1] = loadObject("W1bForest", sample)
            self.var.W1b[2] = loadObject("W1bIrr",    sample)
            self.var.W1[1]  = loadObject("W1Forest",  sample)
            self.var.W1[2]  = loadObject("W1Irr",     sample)
        else:
            # Restore full arrays (no soil-moisture updating)
            self.var.W1a = loadObject("W1a", sample)
            self.var.W1b = loadObject("W1b", sample)
            self.var.W1  = loadObject("W1",  sample)
            self.var.W2  = loadObject("W2",  sample)

        # ------------------------------------------------------------------
        # Inflow & precipitation scaling
        # ------------------------------------------------------------------
        self.var.InfScaling = loadObject("InflowScaling", sample)
        self.var.PrScaling  = loadObject("PrScaling", sample)

        # ------------------------------------------------------------------
        # Groundwater
        # ------------------------------------------------------------------
        self.var.LZ = loadObject("LZ", sample)

        if STATE_AUGMENTATION:
            self.var.UZ[0] = loadObject("UZ_EnKF",     sample)
            self.var.UZ[1] = loadObject("UZForest",    sample)
            self.var.UZ[2] = loadObject("UZIrrigation", sample)
        else:
            self.var.UZ = loadObject("UZ", sample)

        # ------------------------------------------------------------------
        # Lakes
        # ------------------------------------------------------------------
        settings = LisSettings.instance()
        option = settings.options
        if option.get("simulateLakes", False):
            self.var.LakeStorageM3CC = loadObject("LakeStorageM3", sample)
            self.var.LakeOutflow     = loadObject("LakeOutflow", sample)

        # ------------------------------------------------------------------
        # Routing – always updated from EnKF-modified fields
        # ------------------------------------------------------------------
        self.var.ChanM3Kin         = loadObject("ChanM3Kin_EnKF", sample)
        self.var.ChanQKin          = loadObject("ChanQKin_EnKF",  sample)
        self.var.ChanQ             = loadObject("ChanQ_EnKF",     sample)
        self.var.ToChanM3RunoffDt  = loadObject("ToChan",         sample)
        self.var.ChanQAvg          = loadObject("ChanQAvg_EnKF",  sample)
        # self.var.CrossSection2Area = loadObject("CrossSection2", sample)

        try:
            self.var.Chan2M3Kin   = loadObject("Chan2M3Kin_EnKF", sample)
            self.var.Chan2QKin    = loadObject("Chan2QKin_EnKF",  sample)
            self.var.Sideflow1Chan = loadObject("Sideflow1Chan",  sample)
            self.var.QInM3Old     = loadObject("QInM3Old",        sample)
            # self.var.QLimit      = loadObject("QLimit",          sample)
            self.var.Chan2M3Start = loadObject("Chan2M3Start",    sample)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Overland & reservoirs
        # ------------------------------------------------------------------
        self.var.OFM3Direct = loadObject("OFM3Direct", sample)
        self.var.OFM3Other  = loadObject("OFM3Other",  sample)
        self.var.OFM3Forest = loadObject("OFM3Forest", sample)

        if option.get("simulateReservoirs", False):
            self.var.ReservoirStorageM3CC = loadObject("ReservoirStorageM3CC", sample)
            self.var.ReservoirFill        = loadObject("ReservoirFill",        sample)
            self.var.QResOutM3Dt         = loadObject("QResOutM3Dt",         sample)

        # Suspended sediment
        try:
            self.output_module.var.Tss = loadObject("Tss", sample)
        except Exception:
            pass

        # NetCDF flags
        cdfflags = CDFFlags.instance()
        cdfflags.set(loadObject("cdfFlag", sample))

        print("Resuming from saved state after EnKF...")
