#!/usr/bin/env python

import argparse
import math
import lalsimulation as lalsim
import lal
import numpy as np
import pycbc
import random
import h5py
from pycbc.waveform import FrequencySeries
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import match, sigma, overlap_cplx, overlap
from pycbc.conversions import chirp_distance, mchirp_from_mass1_mass2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#parser.add_argument('--version', action=pycbc.version.Version)
parser.add_argument("-V", "--verbose", action="store_true",
                  help="print extra debugging information", default=False )
parser.add_argument('--bank-file')
parser.add_argument('--job-number', type=int)
parser.add_argument('--total-jobs', type=int)
parser.add_argument('--match-flow', default= 30.0, type=float)
parser.add_argument('--match-fhigh', default=2048.0, type=float)
opt = parser.parse_args()

def compute_sigmasq(htilde, deltaF):
    """
    Find norm of whitened h(f) array.
    """
    # vdot is dot with complex conjugation
    return float(np.vdot(htilde, htilde).real * 4 * deltaF)


def compute_correlation(htilde1, htilde2, deltaF):
    """
    Find the real component of correlation between htilde1 and htilde2.
    """
    # vdot is dot with complex conjugation
    return float(np.vdot(htilde1, htilde2).real * 4 * deltaF)

def compute_complex_correlation(htilde1, htilde2, deltaF):
    """
    Find the complex correlation between htilde1 and htilde2.
    """
    # vdot is dot with complex conjugation
    return np.vdot(htilde1, htilde2) * 4 * deltaF

def generate_random_signal(df, flow=opt.match_flow, ffinal=opt.match_fhigh):
    mass1 = random.random() * (10.05 - 9.95) + 9.95
    mass2 = random.random() * (1.41 - 1.39) + 1.39
    while 1:
        spin1x = random.random() * 2. - 1.
        spin1y = random.random() * 2. - 1.
        spin1z = random.random() * 2. - 1.
        if (spin1x**2 + spin1y**2 + spin1z**2) < 0.99:
            break

    while 1:
        spin2x = random.random() * 2. - 1.
        spin2y = random.random() * 2. - 1.
        spin2z = random.random() * 2. - 1.
        if (spin2x**2 + spin2y**2 + spin2z**2) < 0.99:
            break

    theta = random.random() * np.pi
    phi = random.random() * 2 * np.pi
    iota = random.random() * np.pi
    psi = random.random() * 2 * np.pi
    orb_phase = random.random() * 2 * np.pi

    LALparams = lal.CreateDict()
    # Enforce the Euler angles to be Spin-Taylor
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(LALparams, 320)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(LALparams, 2)


    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        mass1*lal.MSUN_SI,
        mass2*lal.MSUN_SI,
        spin1x,
        spin1y,
        spin1z,
        spin2x,
        spin2y,
        spin2z,
        1.e6*lal.PC_SI,
        iota,
        phi0,
        0,
        0,
        0,
        df,
        flow,
        ffinal,
        flow,
        LALparams,
        lalsim.IMRPhenomXP
    )
    
    flen = hp.data.length
    psd = aLIGOZeroDetHighPower(flen, df, self.flow)
    ASD = psd.data**0.5

    arr_view_h1 = hp.data.data
    arr_view_h2 = hc.data.data

    # Whiten
    arr_view_h1[:] /= ASD[:hp.data.length]
    arr_view_h1[:int(flow / df)] = 0.
    arr_view_h1[int(ffinal/df):hp.data.length] = 0.

    arr_view_h2[:] /= ASD[:hp.data.length]
    arr_view_h2[:int(flow / df)] = 0.
    arr_view_h2[int(ffinal/df):hp.data.length] = 0.
        
    hpP = FrequencySeries(hp.data.data[:], delta_f=hp.deltaF, epoch=hp.epoch)
    hcP = FrequencySeries(hc.data.data[:], delta_f=hc.deltaF, epoch=hc.epoch)

    return hpP * np.sin(2*psi) + hcP * np.cos(2*psi)

def compute_beta(tmplt):
    """ Calculate beta (thetaJL) using code from
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_8c_source.html#l06105
    """
    m1 = tmplt.mass1
    m2 = tmplt.mass2
    s1x = tmplt.spin1x
    s1y = tmplt.spin1y
    s1z = tmplt.spin1z
    s2x = tmplt.spin2x
    s2y = tmplt.spin2y
    s2z = tmplt.spin2z
    flow = tmplt.flow

    eta = m1 * m2 / (m1 + m2) / (m1 + m2);
    v0 = ((m1 + m2) * lal.MTSUN_SI * lal.PI * flow) ** (1. / 3.)

    lmag = (m1 + m2) * (m1 + m2) * eta / v0
    lmag *= (1.0 + v0 * v0 * (1.5 + eta / 6.))

    s1x = m1 * m1 * s1x
    s1y = m1 * m1 * s1y
    s1z = m1 * m1 * s1z
    s2x = m2 * m2 * s2x
    s2y = m2 * m2 * s2y
    s2z = m2 * m2 * s2z
    jx = s1x + s2x
    jy = s1y + s2y
    jz = lmag + s1z + s2z

    jnorm = (jx * jx + jy * jy + jz * jz) ** (1. / 2.)
    jhatz = jz / jnorm

    return np.arccos(jhatz)

def _dphi(theta_jn, phi_jl, beta):
    """Calculate the difference in the phase angle between J-aligned
    and L-aligned frames using code from
    https://git.ligo.org/lscsoft/pesummary/-/blob/master/pesummary/gw/conversions/angles.py#L36

    Parameters
    ----------
    theta_jn: np.ndarray
        the angle between J and line of sight
    phi_jl: np.ndarray
        the precession phase
    beta: np.ndarray
        the opening angle (angle between J and L)
    """
    theta_jn = np.array([theta_jn])
    phi_jl = np.array([phi_jl])
    beta = np.array([beta])
    n = np.column_stack(
        [np.repeat([0], len(theta_jn)), np.sin(theta_jn), np.cos(theta_jn)]
    )
    l = np.column_stack(
        [
            np.sin(beta) * np.cos(phi_jl), np.sin(beta) * np.sin(phi_jl),
            np.cos(beta)
        ]
    )
    cosi = [np.inner(nn, ll) for nn, ll in zip(n, l)]
    inc = np.arccos(cosi)
    sign = np.sign(np.cos(theta_jn) - (np.cos(beta) * np.cos(inc)))
    cos_d = np.cos(phi_jl) * np.sin(theta_jn) / np.sin(inc)
    inds = np.logical_or(cos_d < -1, cos_d > 1)
    cos_d[inds] = np.sign(cos_d[inds]) * 1.
    dphi = -1. * sign * np.arccos(cos_d)
    return dphi[0]

def _dpsi(theta_jn, phi_jl, beta):
    """Calculate the difference between the polarization with respect to the
    total angular momentum and the polarization with respect to the orbital
    angular momentum using code from
    https://git.ligo.org/lscsoft/pesummary/-/blob/master/pesummary/gw/conversions/angles.py#L13
    """
    if theta_jn == 0:
        return -1. * phi_jl
    n = np.array([np.sin(theta_jn), 0, np.cos(theta_jn)])
    j = np.array([0, 0, 1])
    l = np.array([
        np.sin(beta) * np.sin(phi_jl), np.sin(beta) * np.cos(phi_jl), np.cos(beta)
    ])
    p_j = np.cross(n, j)
    p_j /= np.linalg.norm(p_j)
    p_l = np.cross(n, l)
    p_l /= np.linalg.norm(p_l)
    cosine = np.inner(p_j, p_l)
    sine = np.inner(n, np.cross(p_j, p_l))
    dpsi = np.pi / 2 + np.sign(sine) * np.arccos(cosine)
    return dpsi

class _FDTemplate():
    
    def __init__(self, m1, m2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                 theta, phi, iota, psi, orb_phase, flow, ffinal):
        self.flow = flow
        self.f_final = ffinal

        self.mass1 = float(m1)
        self.mass2 = float(m2)
        self.spin1x = float(spin1x)
        self.spin1y = float(spin1y)
        self.spin1z = float(spin1z)
        self.spin2x = float(spin2x)
        self.spin2y = float(spin2y)
        self.spin2z = float(spin2z)

        self.theta = float(theta)
        self.phi = float(phi)
        self.iota = float(iota)
        self.psi = float(psi)
        self.orb_phase = float(orb_phase)
        self.fref = self.flow
        
        outs = self._model_parameters_from_source_frame(
            self.mass1*lal.MSUN_SI,
            self.mass2*lal.MSUN_SI,
            self.flow,
            self.orb_phase,
            self.iota,
            self.spin1x,
            self.spin1y,
            self.spin1z,
            self.spin2x,
            self.spin2y,
            self.spin2z,
        )
        chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz = outs

        self.chi1_l = float(chi1_l)
        self.chi2_l = float(chi2_l)
        self.chip = float(chip)
        self.thetaJN = float(thetaJN)
        self.alpha0 = float(alpha0)
        self.phi0 = float(phi_aligned)
        self.theta = float(theta)
        self.phi = float(phi)
        self.psi = float(psi)
        # This is a correction on psi, currently unused
        self.psi_corr = zeta_polariz
        self.beta = compute_beta(self)
        
        self.comps = {}

    def gen_harmonics_comp(self, thetaJN, alpha0, phi0, psi, df, f_final):
        # calculate cartesian spins for waveform generator
        a1 = np.sqrt(
            np.sum(np.square([self.spin1x, self.spin1y, self.spin1z]))
        )
        a2 = np.sqrt(
            np.sum(np.square([self.spin2x, self.spin2y, self.spin2z]))
        )
        phi1 = np.fmod(
            2 * np.pi + np.arctan2(self.spin1y, self.spin1x),
            2 * np.pi
        )
        phi2 = np.fmod(
            2 * np.pi + np.arctan2(self.spin2y, self.spin2x),
            2 * np.pi
        )
        phi12 = phi2 - phi1
        if phi12 < 0:
            phi12 += 2 * np.pi
        tilt1 = np.arccos(self.spin1z / a1)
        tilt2 = np.arccos(self.spin2z / a2)
        iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                thetaJN, alpha0, tilt1, tilt2, phi12, a1, a2,
                self.mass1*lal.MSUN_SI, self.mass2*lal.MSUN_SI, self.fref, phi0
            )
        # generate hp, hc for given orientation with lalsimulation
        LALparams = lal.CreateDict()
        if self.flag == True:
            lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(LALparams, 320)
            lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(LALparams, 2)
            
        hp, hc = lalsim.SimInspiralFD(
            self.mass1*lal.MSUN_SI, self.mass2*lal.MSUN_SI, spin1x, spin1y,
            spin1z, spin2x, spin2y, spin2z, 1.e6*lal.PC_SI, iota, phi0,
            0, 0, 0, df, self.flow, f_final, self.fref, LALparams,
            lalsim.GetApproximantFromString(self.approximant)
        )
        # 1908.05707 defines psi in J-aligned frame. Need to rotate to
        # L-aligned frame and multiply by w+, wx
        dpsi = _dpsi(thetaJN, alpha0, self.beta)
        fp = np.cos(2 * (psi - dpsi))
        fc = -1. * np.sin(2 * (psi - dpsi))
        h = (fp * hp.data.data[:] + fc * hc.data.data[:])
        # 1908.05707 defines phi in J-aligned frame. Need to rotate to
        # L-aligned frame
        h *= np.exp(2j * _dphi(thetaJN, alpha0, self.beta))
        # create LAL frequency array and return precessing harmonic
        new = lal.CreateCOMPLEX16FrequencySeries(
            "", lal.LIGOTimeGPS(hp.epoch), 0, df, lal.SecondUnit, len(h)
        )
        new.data.data[:] = h[:]
        return new
        
    def compute_waveform_five_comps(self, df, f_final):
        # calculate 5 harmonic decomposition as defined in 1908.05707
        hgen1a = self.gen_harmonics_comp(
            0., 0., 0., 0., df, f_final
            )
        hgen1b = self.gen_harmonics_comp(
            0., 0., np.pi/4., np.pi/4, df, f_final
        )
        # Edit these arrays in place to avoid defining new LAL arrays
        tmp = hgen1a.data.data[:] - hgen1b.data.data[:]
        hgen1b.data.data[:] = (hgen1a.data.data[:] + hgen1b.data.data[:]) / 2.
        hgen1a.data.data[:] = tmp / 2.
        h1 = hgen1a
        h5 = hgen1b
        hgen2a = self.gen_harmonics_comp(
            np.pi/2., 0., np.pi/4., np.pi/4, df, f_final
        )
        hgen2b = self.gen_harmonics_comp(
            np.pi/2., np.pi/2., 0., np.pi/4, df, f_final
        )
        tmp = hgen2a.data.data[:] + hgen2b.data.data[:]
        hgen2b.data.data[:] = -0.25 * (
            hgen2a.data.data[:] - hgen2b.data.data[:]
        )
        hgen2a.data.data[:] = -0.25 * tmp
        h2 = hgen2a
        h4 = hgen2b
        hgen3a = self.gen_harmonics_comp(
            np.pi/2., 0., 0., 0., df, f_final
        )
        hgen3b = self.gen_harmonics_comp(
            np.pi/2., np.pi/2., 0., 0., df, f_final
        )
        hgen3a.data.data[:] = \
            1./6. * (hgen3a.data.data[:] + hgen3b.data.data[:])
        h3 = hgen3a
        hs = (h1, h2, h3, h4, h5)
        return hs
 
    def whiten_and_normalize(self, arr, ASD, flow, f_final, df):
        arr[:] /= ASD[:len(arr)]  # Whiten
        arr[:int(flow / df)] = 0.  # Zeroing out low frequencies
        arr[int(f_final / df):len(arr)] = 0.  # Zeroing out high frequencies
        sigmasq = compute_sigmasq(arr, df)  # Calculate normalization factor
        arr[:] /= sigmasq ** 0.5  # Normalize
        return arr

    def orthogonalize_and_normalize(self, components, df):
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                corr = compute_complex_correlation(components[i], components[j], df)
                components[j][:] -= corr * components[i][:]
            for j in range(i + 1, len(components)):
                norm_temp = compute_sigmasq(components[j], df)
                components[j][:] /= norm_temp**0.5


    def new_get_whitened_normalized_comps(self, df, reverse_flag=False):
        h1, h2, h3, h4, h5 = self.compute_waveform_five_comps(df, self.f_final)
        flen = h5.data.length if reverse_flag else h1.data.length
        psd = aLIGOZeroDetHighPower(flen, df, self.flow)
        ASD = psd.data ** 0.5

        waveforms = [h5, h4, h3, h2, h1] if reverse_flag else [h1, h2, h3, h4, h5]

        # Error check
        if waveforms[0].data.length > len(ASD):
            raise ValueError("waveform has length greater than ASD; cannot whiten")

        # Process waveforms: whiten, normalize, and orthogonalize
        arr_views = [w.data.data for w in waveforms]
        for i, arr in enumerate(arr_views):
            arr_views[i] = self.whiten_and_normalize(arr, ASD, self.flow, self.f_final, df)

        self.orthogonalize_and_normalize(arr_views, df)

        # Create FrequencySeries objects
        freqs = [FrequencySeries(w.data.data[:], delta_f=w.deltaF, epoch=w.epoch, dtype=np.complex128) for w in waveforms]
        self.comps[df] = freqs
        return freqs


    def gen_fd_comp(self, thetaJN, alpha0, phi0, df):
        LALparams = lal.CreateDict()
        if self.flag == True:
            lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(LALparams, 320)
            lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(LALparams, 2)
            
        return lalsim.SimInspiralFD(
            self.mass1*lal.MSUN_SI,
            self.mass2*lal.MSUN_SI,
            self.spin1x,
            self.spin1y,
            self.spin1z,
            self.spin2x,
            self.spin2y,
            self.spin2z,
            1.e6*lal.PC_SI,
            thetaJN,
            phi0,
            0,
            0,
            0,
            df,
            self.flow,
            self.f_final,
            self.fref,
            LALparams,
            lalsim.GetApproximantFromString(self.approximant)
    )
    
    def generate_random_signal(self, df):
        thetaJN = random.random() * np.pi
        alpha0 = random.random() * 2 * np.pi
        phi0 = random.random() * 2 * np.pi
        psi = random.random() * 2 * np.pi
        
        hp, hc = self.gen_fd_comp(thetaJN, alpha0, phi0, df)
        flen = hp.data.length
        psd = self.psd
        ASD = (psd.data**0.5)

        arr_view_h1 = hp.data.data
        arr_view_h2 = hc.data.data

        # Whiten
        arr_view_h1[:] /= ASD[:hp.data.length]
        arr_view_h1[:int(self.flow / df)] = 0.
        arr_view_h1[int(self.f_final/df):hp.data.length] = 0.

        arr_view_h2[:] /= ASD[:hp.data.length]
        arr_view_h2[:int(self.flow / df)] = 0.
        arr_view_h2[int(self.f_final/df):hp.data.length] = 0.
        
        hpP = FrequencySeries(hp.data.data[:], delta_f=hp.deltaF, epoch=hp.epoch)
        hcP = FrequencySeries(hc.data.data[:], delta_f=hc.deltaF, epoch=hc.epoch)

        return hpP * np.sin(2*psi) + hcP * np.cos(2*psi)
    
    def test_five_comps_orthogonality(self, df=1./64.):

        hs = self.get_whitened_normalized_comps(df)

        for i in range(5):
            for j in range(i+1,5):
                absval = abs(overlap_cplx(hs[i], hs[j], low_frequency_cutoff=self.flow))
                if absval > 1E-10:
                    print (i, j, abs(overlap_cplx(hs[i], hs[j], low_frequency_cutoff=self.flow)))
    
    def test_random_signals_overlap(self, N, df=1./8.):
        output_arr = np.zeros([N,11])

        hs = self.new_get_whitened_normalized_comps(df)
        hs_re = self.new_get_whitened_normalized_comps(df, reverse_flag=True)
        
        self.psd = aLIGOZeroDetHighPower(len(hs[0]), df, self.flow)

        for i in range(N):
            hcurr = self.generate_random_signal(df)
            match1    = overlap_cplx(hs[0], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match2    = overlap_cplx(hs[1], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match3    = overlap_cplx(hs[2], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match4    = overlap_cplx(hs[3], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match5    = overlap_cplx(hs[4], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match1_re = overlap_cplx(hs_re[0], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match2_re = overlap_cplx(hs_re[1], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match3_re = overlap_cplx(hs_re[2], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match4_re = overlap_cplx(hs_re[3], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            match5_re = overlap_cplx(hs_re[4], hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
            #print('i:', abs(match1_re), abs(match2_re), abs(match3_re), abs(match4_re), abs(match5_re))

            output_arr[i,0] = abs(match1)
            output_arr[i,1] = (abs(match1)**2 + abs(match2)**2)**0.5
            output_arr[i,2] = (output_arr[i,1]**2 + abs(match3)**2)**0.5
            output_arr[i,3] = (output_arr[i,2]**2 + abs(match4)**2)**0.5
            output_arr[i,4] = (output_arr[i,3]**2 + abs(match5)**2)**0.5
            output_arr[i,5] = abs(match1_re)
            output_arr[i,6] = (abs(match1_re)**2 + abs(match2_re)**2)**0.5
            output_arr[i,7] = (output_arr[i,6]**2 + abs(match3_re)**2)**0.5
            output_arr[i,8] = (output_arr[i,7]**2 + abs(match4_re)**2)**0.5
            output_arr[i,9] = (output_arr[i,8]**2 + abs(match5_re)**2)**0.5
            output_arr[i,10] = sigma(hcurr, low_frequency_cutoff=opt.match_flow, high_frequency_cutoff=opt.match_fhigh)
        return output_arr
    
def get_FFs(tmp):
    oarr = tmp.test_random_signals_overlap(200)
    output2 = np.zeros([2, 11])
    output2[1,1] = (np.mean(oarr[:,0]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[1,2] = (np.mean(oarr[:,1]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[1,3] = (np.mean(oarr[:,2]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[1,4] = (np.mean(oarr[:,3]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[1,5] = (np.mean(oarr[:,4]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)

    #Reversed harmonics output 
    output2[1,6] = (np.mean(oarr[:,5]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[1,7] = (np.mean(oarr[:,6]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[1,8] = (np.mean(oarr[:,7]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[1,9] = (np.mean(oarr[:,8]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[1,10] = (np.mean(oarr[:,9]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    return output2

class PhenomTPTemplate(_FDTemplate):
    approximant = "IMRPhenomTP"
    flag = False

    def _model_parameters_from_source_frame(self, *args):
        return lalsim.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
            *args, None
        )
    
    
class PhenomXPTemplate(_FDTemplate):
    approximant = "IMRPhenomXP"
    flag = False

    def _model_parameters_from_source_frame(self, *args):
        return lalsim.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
            *args, None
        )
    
class PhenomXPTemplate_flag(_FDTemplate):
    approximant = "IMRPhenomXP"
    flag = True

    def _model_parameters_from_source_frame(self, *args):
        return lalsim.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
            *args, None
        ) 

with h5py.File(opt.bank_file, 'r') as tbank:
    num_temps = len(tbank['mass1'][:])

idx_diff = math.ceil(num_temps / opt.total_jobs)
print(idx_diff, num_temps, opt.total_jobs)

start_idx = min(opt.job_number*idx_diff, num_temps)
end_idx = min((opt.job_number+1)*idx_diff, num_temps)
templates = []

with h5py.File(opt.bank_file, 'r') as tbank:
    for i in range(start_idx, end_idx):
        m1 = tbank['mass1'][i]
        m2 = tbank['mass2'][i]
        spin1x = tbank['spin1x'][i]
        spin1y = tbank['spin1y'][i]
        spin1z = tbank['spin1z'][i]
        spin2x = tbank['spin2x'][i]
        spin2y = tbank['spin2y'][i]
        spin2z = tbank['spin2z'][i]
        theta = tbank['latitude'][i]
        phi = tbank['longitude'][i]
        psi = tbank['polarization'][i]
        iota = tbank['inclination'][i]
        orb_phase = tbank['orbital_phase'][i]

        curr_tmp = PhenomXPTemplate_flag(m1, m2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                     theta, phi, iota, psi, orb_phase, opt.match_flow, opt.match_fhigh)
        templates.append(curr_tmp)


output2 = np.zeros([end_idx-start_idx, 11])

print(start_idx, end_idx)

for jj, curr_tmp in enumerate(templates):
    i = jj + start_idx
    if i < start_idx or i >= end_idx:
        continue
    print (i, start_idx, end_idx)
    oarr = curr_tmp.test_random_signals_overlap(200)
    #print(oarr[1])
    mchirp = mchirp_from_mass1_mass2(curr_tmp.mass1, curr_tmp.mass2)

    output2[i-start_idx,0] = i
        
    output2[i-start_idx,1] = (np.mean(oarr[:,0]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[i-start_idx,2] = (np.mean(oarr[:,1]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[i-start_idx,3] = (np.mean(oarr[:,2]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[i-start_idx,4] = (np.mean(oarr[:,3]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[i-start_idx,5] = (np.mean(oarr[:,4]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
  
    #Reversed harmonics output 
    output2[i-start_idx,6] = (np.mean(oarr[:,5]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[i-start_idx,7] = (np.mean(oarr[:,6]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[i-start_idx,8] = (np.mean(oarr[:,7]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[i-start_idx,9] = (np.mean(oarr[:,8]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    output2[i-start_idx,10] = (np.mean(oarr[:,9]**3 * oarr[:,10]**3) / np.mean(oarr[:,10]**3))**(1./3.)
    #av_dist = (np.mean(oarr[:,5]**3))**(1./3.)
    #chirp_dist = chirp_distance(av_dist, mchirp)

np.savetxt('Bank_output_{}.dat'.format(opt.job_number), output2)


