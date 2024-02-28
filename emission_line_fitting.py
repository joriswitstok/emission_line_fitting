#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for fitting emission line spectra.

Joris Witstok, 2023
"""

import numpy as np
sigma2fwhm = (2.0 * np.sqrt(2.0 * np.log(2)))

from pymultinest.solve import Solver
from scipy.stats import norm, skewnorm, gamma
from scipy.ndimage import gaussian_filter1d
from spectres import spectres

def import_matplotlib():
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_style("ticks")

    return (plt, sns)

def import_corner():
    import corner
    return corner

class MN_emission_line_solver(Solver):
    def __init__(self, line_list, wl_obs_list, flux_list, flux_err_list,
                    cont_flux=None, min_amplitude=None, max_amplitude=None, sigma_default=100.0,
                    spectral_resolution_list=None, full_convolution=False, def_n_res=5.0,
                    redshift_prior={"type": "uniform", "params": [0, 20]}, sigma_prior={"type": "uniform", "params": [0, 500]},
                    res_scale_function=None, res_scale_priors=None,
                    mpi_run=False, mpi_comm=None, mpi_rank=0, mpi_ncores=1, mpi_synchronise=None,
                    fit_name=None, print_setup=True, plot_setup=False, mpl_style=None, verbose=True, **solv_kwargs):
        self.mpi_run = mpi_run
        self.mpi_comm = mpi_comm
        self.mpi_rank = mpi_rank
        self.mpi_ncores = mpi_ncores
        self.mpi_synchronise = lambda _: None if mpi_synchronise is None else mpi_synchronise

        self.fit_name = fit_name
        self.mpl_style = mpl_style
        self.verbose = verbose
        if self.verbose and self.mpi_rank == 0:
            print("Initialising MultiNest Solver object{}{}...".format(" for {}".format(self.fit_name) if self.fit_name else '',
                                                                        " with {} cores".format(self.mpi_ncores) if self.mpi_run else ''))

        self.line_list = line_list
        
        self.min_amplitude = 1e-4 if min_amplitude is None else min_amplitude
        self.max_amplitude = 1e4 if max_amplitude is None else max_amplitude
        self.sigma_default = sigma_default
        
        for l in self.line_list:
            if not hasattr(l, "upper_limit"):
                l.upper_limit = False
            
        for l in self.line_list:
            if hasattr(l, "fixed_line_ratio"):
                assert not l.fixed_line_ratio["rline"].upper_limit
            if hasattr(l, "var_line_ratio"):
                assert not l.var_line_ratio["rline"].upper_limit
            if hasattr(l, "coupled_delta_v_line"):
                assert not l.coupled_delta_v_line.upper_limit
            if hasattr(l, "coupled_sigma_v_line"):
                assert not l.coupled_sigma_v_line.upper_limit
        
        self.full_convolution = full_convolution
        self.def_n_res = def_n_res
        
        self.redshift_prior = redshift_prior
        self.fixed_redshift = self.redshift_prior["params"][0] if self.redshift_prior["type"].lower() == "fixed" else None
        
        self.sigma_prior = sigma_prior
        self.res_scale_function = res_scale_function
        if res_scale_priors is None:
            assert self.res_scale_function is None
            self.res_scale_priors = []
        else:
            self.res_scale_priors = res_scale_priors

        self.set_cont_flux(cont_flux)
        self.set_prior()
        self.set_observations(wl_obs_list, flux_list, flux_err_list, spectral_resolution_list)

        if print_setup or plot_setup:
            mean_params = [np.mean(self.get_prior_extrema(prior)) for prior in self.priors]
            min_params = [self.get_prior_extrema(prior)[0] for prior in self.priors]
            max_params = [self.get_prior_extrema(prior)[1] for prior in self.priors]
            if print_setup and self.mpi_rank == 0:
                print("\n{:<5}\t{:<30}\t{:<10}\t{:<30}\t{:<20}\t{:<20}\t{:<20}".format('', "Parameter", "Prior type", "Prior parameters", "Minimum prior value", "Mean prior value", "Maximum prior value"))
                print(*["{:<5}\t{:<30}\t{:<10}\t{:<30}\t{:<20}\t{:<20}\t{:<20}".format(*x) for x in zip(range(1, self.n_dims+1), self.params, [prior["type"] for prior in self.priors],
                                                                                        [str(prior["params"]) for prior in self.priors], min_params, mean_params, max_params)], sep='\n')
                print("\n{:<5}\t{:<30}\t{:<10}\t{:<30}\t{:<20}\t{:<20}\t{:<20}".format('', '', '', "Log-likelihood", self.LogLikelihood(min_params), self.LogLikelihood(mean_params), self.LogLikelihood(max_params)))
        
        if plot_setup:
            self.plot_models(params_list=[min_params, mean_params, max_params],
                                labels_list=["Model (min.)", "Model (mean)", "Model (max.)"], showfig=True)
        
        self.mpi_synchronise(self.mpi_comm)
        super().__init__(n_dims=self.n_dims, use_MPI=self.mpi_run, **solv_kwargs)
        if self.mpi_run:
            self.samples = self.mpi_comm.bcast(self.samples, root=0)
        self.mpi_synchronise(self.mpi_comm)
        self.fitting_complete = True

    def plot_models(self, params_list, labels_list, figname=None, showfig=False):
        if self.mpi_rank == 0:
            plt, sns = import_matplotlib()
            if self.mpl_style:
                plt.style.use(self.mpl_style)
            colors = sns.color_palette()

            n_rows = 1 + self.n_obs
            n_cols = len(params_list)

            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey="row", squeeze=False,
                                        gridspec_kw=dict(hspace=0, wspace=0), figsize=(8.27*n_cols/2, 11.69*n_rows/4))
            gs = axes[0, 0].get_gridspec()
            
            ax_lab = fig.add_subplot(gs[:, :])
            ax_lab.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            for spine in ax_lab.spines.values():
                spine.set_visible(False)
            ax_lab.set_zorder(-1)
            
            ax_lab.set_xlabel(r"$\lambda_\mathrm{obs} \, (\mathrm{\AA})$", labelpad=20)
            ax_lab.set_ylabel(r"$F_\lambda$", labelpad=30)

            axes[0, 0].annotate(text=r"Intrinsic ($R = 100000$)", xy=(0, 1), xytext=(4, -4),
                                    xycoords="axes fraction", textcoords="offset points",
                                    va="top", ha="left", size="xx-small",
                                    bbox=dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))

            wl_emit = np.arange(np.min([0.95*l.wl for l in self.line_list]), np.max([1.05*l.wl for l in self.line_list]), np.min([l.wl for l in self.line_list])/5000.0)
            
            for coli, params, label in zip(range(n_cols), params_list, labels_list):
                z = self.fixed_redshift if self.fixed_redshift else params[self.params.index("redshift")]
                axes[0, coli].plot(wl_emit*(1.0 + z), self.create_model(params, wl_emit=wl_emit, R=1e5), drawstyle="steps-mid",
                                    color=colors[0], alpha=0.8, label=label)

            for ax in axes[0]:
                ax.axhline(y=0, color='k', alpha=0.8)
                ax.legend(loc="upper right", fontsize="xx-small")
            
            models_list = [self.create_model(params)[0] for params in params_list]
            
            for oi, wl_obs, flux, flux_err, spectral_resolution, axes_row in zip(range(self.n_obs), self.wl_obs_list,
                                                                                    self.flux_list, self.flux_err_list,
                                                                                    self.spectral_resolution_list, axes[1:]):
                axes_row[0].annotate(text=r"Resolution {:d} ($R \approx {:.0f}$)".format(oi, np.median(spectral_resolution)),
                                        xy=(0, 1), xytext=(4, -4), xycoords="axes fraction", textcoords="offset points",
                                        va="top", ha="left", size="xx-small",
                                        bbox=dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))

                for coli, ax, params in zip(range(n_cols), axes_row, params_list):
                    ax.axhline(y=0, color='k', alpha=0.8)

                    ax.plot(wl_obs, flux, drawstyle="steps-mid", color=colors[1+oi], alpha=0.8, label="Data")
                    ax.fill_between(wl_obs, (flux-flux_err), (flux+flux_err),
                                    step="mid", alpha=0.1, edgecolor="None", facecolor=colors[1+oi], zorder=4)
                    
                    ax.plot(wl_obs, models_list[coli][oi], color=colors[0], drawstyle="steps-mid", alpha=0.8)
                    
                    dl_obs = np.interp(wl_obs, 0.5*(wl_obs[1:]+wl_obs[:-1]), np.diff(wl_obs))
                    ax.fill_between(np.vstack((wl_obs-0.5*dl_obs, wl_obs+0.5*dl_obs)).reshape((-1,), order='F'), y1=0, y2=1,
                                    where=np.vstack((self.fit_select_list[oi], self.fit_select_list[oi])).reshape((-1,), order='F'),
                                    transform=ax.get_xaxis_transform(), edgecolor="None", facecolor="grey", alpha=0.4)
                    
                    if coli == 1:
                        ax.legend(loc="upper right", fontsize="xx-small")
                
                ax.set_ylim(-2*np.nanmean(flux_err), np.nanmax(flux+flux_err))
            
            if figname:
                fig.savefig(figname, bbox_inches="tight")
            if showfig:
                plt.show()
            
            plt.close(fig)

    def plot_corner(self, params=None, figname=None, showfig=False):
        if self.mpi_rank != 0:
            return None
        else:
            assert self.fitting_complete and hasattr(self, "samples")
            if params is None:
                params = self.params
            else:
                assert np.all([param in self.params for param in params])
            n_params = len(params)
            n_samples = self.samples.shape[0]

            plt, sns = import_matplotlib()
            if self.mpl_style:
                plt.style.use(self.mpl_style)
            corner = import_corner()

            data = self.samples.transpose()
            
            n_bins = max(50, n_samples//500)
            bins = [n_bins] * n_params
            
            # Deselect non-finite data for histograms
            select_data = np.product([np.isfinite(d) for d in data], axis=0).astype(bool)
            data = [d[select_data] for d, param in zip(data, self.params) if param in params]

            ranges = [self.get_prior_extrema(prior) for prior, param in zip(self.priors, self.params) if param in params]
            
            fig = corner.corner(np.transpose(data), labels=params, bins=bins, range=ranges,
                                quantiles=[0.5*(1-0.682689), 0.5, 0.5*(1+0.682689)], smooth=0.5, smooth1d=0.5,
                                color=sns.color_palette()[0], show_titles=True, title_kwargs=dict(size="small"))

            # Extract the axes
            axes_c = np.array(fig.axes).reshape((n_params, n_params))

            # Loop over the histograms
            for ri in range(n_params):
                for ci in range(ri):
                    axes_c[ri, ci].vlines(np.percentile(data[ci], [0.5*(100-68.2689), 50, 0.5*(100+68.2689)]), ymin=0, ymax=1,
                                            transform=axes_c[ri, ci].get_xaxis_transform(), linestyles=['--', '-', '--'], color="grey")
                    axes_c[ri, ci].hlines(np.percentile(data[ri], [0.5*(100-68.2689), 50, 0.5*(100+68.2689)]), xmin=0, xmax=1,
                                            transform=axes_c[ri, ci].get_yaxis_transform(), linestyles=['--', '-', '--'], color="grey")
                    axes_c[ri, ci].plot(np.percentile(data[ci], 50), np.percentile(data[ri], 50), color="grey", marker='s', mfc="None", mec="grey")
            
            if figname:
                fig.savefig(figname, bbox_inches="tight")
            if showfig:
                plt.show()

            plt.close(fig)

    def Gaussian(self, A, x0, y0, sigma, x):
        return A / (np.sqrt(2*np.pi)*sigma) * np.exp( -(x-x0)**2 / (2.0*sigma**2) ) + y0

    def asym_Gaussian(self, A, x0, y0, sigma, a_asym, x):
        """

        Skewed Gaussian profile

        """
        
        if a_asym == 0:
            return self.Gaussian(A, x0, y0, sigma, x)
        else:
            # Scale sigma so variance of the profile will be equal to sigma (https://en.wikipedia.org/wiki/Skew_normal_distribution)
            delta = a_asym / np.sqrt(1.0+a_asym**2)
            sigma_scaled = sigma / np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
            # Compute the mode (maximum) of the function (https://en.wikipedia.org/wiki/Skew_normal_distribution)
            m = sigma_scaled * (np.sqrt(2.0/np.pi)*delta - \
                (1.0 - np.pi/4.0)*(np.sqrt(2.0/np.pi)*delta)**3/(1.0-2.0/np.pi*delta**2) - np.sign(a_asym)/2.0 * np.exp(-2.0*np.pi/np.abs(a_asym)))
            return A * skewnorm.pdf(x, a=a_asym, loc=x0-m, scale=sigma_scaled) + y0

    def line_profile(self, wl, l):
        assert hasattr(l, "line_params")
        
        if hasattr(l, "asymmetric_Gaussian"):
            line_f_l_rest = self.asym_Gaussian(A=l.line_params["amplitude"], x0=l.line_params["wl0"], y0=0, sigma=l.line_params["sigma_l"],
                                                a_asym=l.line_params["a_asym"], x=wl)
        else:
            line_f_l_rest = self.Gaussian(A=l.line_params["amplitude"], x0=l.line_params["wl0"], y0=0, sigma=l.line_params["sigma_l"], x=wl)
        
        return line_f_l_rest

    def set_observations(self, wl_obs_list, flux_list, flux_err_list, spectral_resolution_list):
        self.wl_obs_list = wl_obs_list
        self.flux_list = flux_list
        self.flux_err_list = flux_err_list
        
        self.n_obs = len(self.wl_obs_list)
        assert len(self.flux_list) == self.n_obs and len(self.flux_err_list) == self.n_obs
        
        if spectral_resolution_list is None:
            # Default to a spectral resolution of R = 100000
            spectral_resolution_list = [np.tile(1e5, wl_obs.size) for wl_obs in self.wl_obs_list]
        self.spectral_resolution_list = spectral_resolution_list
        self.med_spectral_resolution_list = [np.median(spectral_resolution) for spectral_resolution in self.spectral_resolution_list]

        if self.fixed_redshift:
            self.wl_emit_list = []
            self.fit_select_list = []
            self.n_res_list = []
            self.wl_emit_model_list = []
            self.model_profile_list = []

            for oi in range(self.n_obs):
                wl_emit = self.wl_obs_list[oi] / (1.0 + self.fixed_redshift)
                self.wl_emit_list.append(wl_emit)
                
                n_res_mult = np.tile(1.0, len(self.line_list))
                fit_select = np.tile(False, wl_emit.size)
                
                minmax_params = [self.get_prior_extrema(prior) for prior in self.priors]
                min_spectral_resolution = self.get_spectral_resolution(theta=None, oi=oi, resolution_extreme="min")[1]
                max_spectral_resolution = self.get_spectral_resolution(theta=None, oi=oi, resolution_extreme="max")[1]
                
                for li, l in enumerate(self.line_list):
                    if l.upper_limit:
                        continue
                    R_min = np.interp(l.wl, wl_emit, min_spectral_resolution)
                    min_line_params = self.get_line_params([minmax_param[0] for minmax_param in minmax_params], l=l, R=R_min)
                    max_line_params = self.get_line_params([minmax_param[1] for minmax_param in minmax_params], l=l, R=R_min)
                    
                    # If the line is (much) narrower than a spectral resolution element, need to increase the number of wavelength bins
                    if self.full_convolution:
                        n_res_mult[li] = l.wl / R_min / min_line_params["sigma_l_convolved"]
                    else:
                        fit_select += (wl_emit >= min_line_params["wl0"] - 7.0 * max_line_params["sigma_l_convolved"]) * \
                                        (wl_emit <= max_line_params["wl0"] + 7.0 * max_line_params["sigma_l_convolved"])
                
                self.fit_select_list.append(fit_select)
                
                n_res = self.def_n_res * np.max(n_res_mult)
                self.n_res_list.append(n_res)

                wl_emit_model = self.get_highres_wl_array([wl_emit], [max_spectral_resolution], n_res=n_res)[1]
                self.wl_emit_model_list.append(wl_emit_model)

                if self.no_cont_flux:
                    self.model_profile_list.append(np.zeros_like(wl_emit_model))
                else:
                    # Add underlying continuum
                    self.model_profile_list.append(self.get_cont_flux(wl_emit=wl_emit_model))

    def get_highres_wl_array(self, wl_emit_list, spectral_resolution_list, specific_lines=[], n_res=None):
        n_R = len(wl_emit_list)
        assert n_R == len(spectral_resolution_list)
        for wl_emit in wl_emit_list:
            # Check that wavelength array is ascending
            assert wl_emit.size > 1
            assert np.all(np.diff(wl_emit) > 0)
        
        if n_res is None:
            n_res = self.def_n_res
        
        min_resolution = np.min(spectral_resolution_list)

        if specific_lines:
            wl = np.min([l.wl for l in specific_lines])
            wl = wl - 3 * n_res * wl / min_resolution
            wl_max = np.max([l.wl for l in specific_lines])
            wl_max = wl_max + 3 * n_res * wl_max / min_resolution
        else:
            wl = np.min(wl_emit_list) * (1 - 3 * n_res / min_resolution)
            wl_max = np.max(wl_emit_list) * (1 + 3 * n_res / min_resolution)
        
        def get_wl_incr(wl):
            resolutions = [np.interp(wl, wl_R, R, left=np.nan, right=np.nan) for wl_R, R in zip(wl_emit_list, spectral_resolution_list)]
            if np.all(np.isnan(resolutions)):
                resolutions = []
                for li in range(n_R):
                    if wl > wl_emit_list[li][0] * (1 - 3 * n_res / spectral_resolution_list[li][0]):
                        resolutions.append(spectral_resolution_list[li][0])
                    elif wl < wl_emit_list[li][-1] * (1 + 3 * n_res / spectral_resolution_list[li][-1]):
                        resolutions.append(spectral_resolution_list[li][-1])
            
            resolution = np.nanmax(resolutions) if resolutions else min_resolution
            return wl / (n_res * resolution)

        wl_emit_bin_edges = []
        wl_incr = 0
        while wl < wl_max:
            wl_emit_bin_edges.append(wl)
            wl_incr = get_wl_incr(wl + 0.5*wl_incr)
            wl += wl_incr
        
        wl_emit_bin_edges = np.array(wl_emit_bin_edges)
        assert wl_emit_bin_edges.size > 1
        
        wl_emit_model = 0.5 * (wl_emit_bin_edges[:-1] + wl_emit_bin_edges[1:])
        
        return (wl_emit_bin_edges, wl_emit_model)

    def set_cont_flux(self, cont_flux):
        self.cont_flux = cont_flux
        self.no_cont_flux = self.cont_flux is None

    def get_cont_flux(self, wl_emit):
        if self.no_cont_flux:
            cont_flux = np.tile(np.nan, wl_emit.size)
        elif isinstance(self.cont_flux, dict):
            if self.cont_flux["type"] == "power-law":
                # Model continuum as a power-law slope (in the rest frame)
                cont_flux = np.where(wl_emit < 1215.6701, 0.0, self.cont_flux['C'] * (wl_emit/self.cont_flux["wl0"])**self.cont_flux["beta"])
            elif self.cont_flux["type"] == "SED_model":
                # Use intrinsic spectrum from a given SED model
                cont_flux = spectres(wl_emit, self.cont_flux["wl_emit_intrinsic"], self.cont_flux["spectrum_intrinsic"], fill=0.0, verbose=False)
            elif self.cont_flux["type"] == "function":
                # Use given function
                cont_flux = self.cont_flux["function"](wl_emit)
        else:
            raise ValueError("continuum type not recognised")
        
        return cont_flux
    
    def get_prior_extrema(self, prior):
        if prior["type"].lower() == "uniform":
            minimum = prior["params"][0]
            maximum = prior["params"][1]
        elif prior["type"].lower() == "loguniform":
            # Set the minimum/maximum value as the edge of the uniform distribution or as a 3σ outlier
            minimum = 10**(prior["params"][0])
            maximum = 10**(prior["params"][1])
        elif prior["type"].lower() == "normal":
            # Set the minimum/maximum value as the edge of the uniform distribution or as a 3σ outlier
            minimum = prior["params"][0] - 3 * prior["params"][1]
            maximum = prior["params"][0] + 3 * prior["params"][1]
        elif prior["type"].lower() == "lognormal":
            # Set the minimum/maximum value as the edge of the uniform distribution or as a 3σ outlier
            minimum = 10**(prior["params"][0] - 3 * prior["params"][1])
            maximum = 10**(prior["params"][0] + 3 * prior["params"][1])
        elif prior["type"].lower() == "gamma":
            minimum = 0
            maximum = prior["params"][0] * prior["params"][1] + 5 * prior["params"][1]
        elif prior["type"].lower() == "fixed":
            minimum = prior["params"][0]
            maximum = prior["params"][0]
        else:
            raise TypeError("prior type '{}' not recognised".format(prior["type"]))
        
        return (minimum, maximum)
    
    def get_spectral_resolution(self, theta, oi, resolution_extreme=''):
        if self.res_scale_function is None:
            if len(self.res_scale_priors) == self.n_obs:
                if self.res_scale_priors[oi]["type"].lower() == "fixed":
                    res_scale = self.res_scale_priors[oi]["params"][0]
                else:
                    if not resolution_extreme:
                        res_scale = theta[self.params.index("res_scale_{:d}".format(oi))]
                    elif resolution_extreme == "min":
                        res_scale = self.get_prior_extrema(self.priors[self.params.index("res_scale_{:d}".format(oi))])[0]
                    elif resolution_extreme == "max":
                        res_scale = self.get_prior_extrema(self.priors[self.params.index("res_scale_{:d}".format(oi))])[1]
            else:
                assert not self.res_scale_priors
                res_scale = 1.0
        else:
            if resolution_extreme:
                res_scale = self.res_scale_function(oi, self.wl_obs_list[oi], resolution_extreme=resolution_extreme)
            else:
                res_scale = self.res_scale_function(oi, self.wl_obs_list[oi], **{param: theta[pi] for pi, param in enumerate(self.params) if "res_scale" in param})
        
        return (self.wl_obs_list[oi], res_scale * self.spectral_resolution_list[oi])
    
    def get_line_params(self, theta, l, R=None, R_idx=None, get_convolved=True):
        line_params = {}
        
        if hasattr(l, "asymmetric_Gaussian"):
            line_params["a_asym"] = theta[self.params.index("a_asym_{}".format(l.name))]
        
        if hasattr(l, "fixed_line_ratio"):
            line_params["amplitude"] = l.fixed_line_ratio["ratio"] * theta[self.params.index("amplitude_{}".format(l.fixed_line_ratio["rline"].name))]
        elif hasattr(l, "var_line_ratio"):
            line_params["amplitude"] = theta[self.params.index("relative_amplitude_{}".format(l.name))] * theta[self.params.index("amplitude_{}".format(l.var_line_ratio["rline"].name))]
        else:
            if "amplitude_{}".format(l.name) in self.params:
                line_params["amplitude"] = theta[self.params.index("amplitude_{}".format(l.name))]
            else:
                line_params["amplitude"] = np.nan
        
        if "delta_v_{}".format(l.name) in self.params:
            line_params["dv"] = theta[self.params.index("delta_v_{}".format(l.name))]
        elif hasattr(l, "delta_v_prior") and l.delta_v_prior["type"].lower() == "fixed":
            line_params["dv"] = l.delta_v_prior["params"][0]
        elif hasattr(l, "coupled_delta_v_line"):
            line_params["dv"] = theta[self.params.index("delta_v_{}".format(l.coupled_delta_v_line.name))]
        else:
            line_params["dv"] = 0
        
        # Convert velocity offset in km/s to rest-frame wavelength
        line_params["wl0"] = l.wl / (1.0 - line_params["dv"]/299792.458)
        
        # Intrinsic line width (in km/s)
        if "sigma_v_{}".format(l.name) in self.params:
            line_params["sigma_v"] = theta[self.params.index("sigma_v_{}".format(l.name))]
        elif hasattr(l, "sigma_v_prior") and l.sigma_v_prior["type"].lower() == "fixed":
            line_params["sigma_v"] = l.sigma_v_prior["params"][0]
        elif hasattr(l, "coupled_sigma_v_line"):
            line_params["sigma_v"] = theta[self.params.index("sigma_v_{}".format(l.coupled_sigma_v_line.name))]
        elif "sigma_v" in self.params:
            line_params["sigma_v"] = theta[self.params.index("sigma_v")]
        else:
            line_params["sigma_v"] = self.sigma_default
        
        # Convert line width in km/s to rest-frame wavelength
        if line_params["sigma_v"] > 0:
            line_params["sigma_l_deconvolved"] = l.wl / (299792.458/line_params["sigma_v"] - 1.0)
        else:
            line_params["sigma_l_deconvolved"] = np.nan
        
        if get_convolved:
            # Add instrumental dispersion (if not specified, will take the maximum resolution)
            if R is None:
                z = self.fixed_redshift if self.fixed_redshift else theta[self.params.index("redshift")]
                obs_indices = range(self.n_obs) if R_idx is None else [R_idx]
                for oi in obs_indices:
                    FWHM_instrument = line_params["wl0"] / np.interp(line_params["wl0"]*(1.0+z), *self.get_spectral_resolution(theta, oi), left=np.nan, right=np.nan)
                    line_params["sigma_l_convolved_res{:d}".format(oi)] = np.sqrt(line_params["sigma_l_deconvolved"]**2 + (FWHM_instrument/sigma2fwhm)**2)
                line_params["sigma_l_convolved"] = np.nanmin([line_params["sigma_l_convolved_res{:d}".format(oi)] for oi in obs_indices])
            else:
                sigma_instrument = line_params["wl0"] / R / sigma2fwhm
                line_params["sigma_l_convolved"] = np.sqrt(line_params["sigma_l_deconvolved"]**2 + sigma_instrument**2)
            line_params["sigma_l"] = line_params["sigma_l_convolved" + ('' if R_idx is None else "_res{:d}".format(R_idx))]
        else:
            assert self.full_convolution
            line_params["sigma_l"] = line_params["sigma_l_deconvolved"]

        line_params["physical"] = np.isfinite(line_params["amplitude"]) and np.isfinite(line_params["wl0"]) and line_params["sigma_l"] > 0

        return line_params

    def get_line_overview(self, specific_lines=[], R=None):
        assert self.fitting_complete and hasattr(self, "samples")
        n_samples = self.samples.shape[0]
        z = self.fixed_redshift if self.fixed_redshift else np.nanmedian(self.samples[:, self.params.index("redshift")])

        specific_lines = specific_lines if specific_lines else self.line_list
        line_overview = {"line_names": [l.name for l in specific_lines], "line_uplims": [l.upper_limit for l in specific_lines],
                            "asymmetric_Gaussian_lines": [], "report_ratios": []}
        
        for l in specific_lines:
            if hasattr(l, "asymmetric_Gaussian"):
                line_overview["asymmetric_Gaussian_lines"].append(l.name)
            if hasattr(l, "fixed_line_ratio"):
                line_overview["fixed_line_ratio_{}_ratio".format(l.name)] = l.fixed_line_ratio["ratio"]
                line_overview["fixed_line_ratio_{}_rline".format(l.name)] = l.fixed_line_ratio["rline"].name
            if hasattr(l, "var_line_ratio"):
                line_overview["var_line_ratio_{}_ratio_min".format(l.name)] = l.var_line_ratio["ratio_min"]
                line_overview["var_line_ratio_{}_ratio_max".format(l.name)] = l.var_line_ratio["ratio_max"]
                line_overview["var_line_ratio_{}_rline".format(l.name)] = l.var_line_ratio["rline"].name
            if hasattr(l, "delta_v_prior"):
                line_overview["delta_v_prior_{}_type".format(l.name)] = l.delta_v_prior["type"]
                line_overview["delta_v_prior_{}_params".format(l.name)] = l.delta_v_prior["params"]
            if hasattr(l, "coupled_delta_v_line"):
                line_overview["delta_v_prior_{}_cline".format(l.name)] = l.coupled_delta_v_line.name
            if hasattr(l, "sigma_v_prior"):
                line_overview["sigma_v_prior_{}_type".format(l.name)] = l.sigma_v_prior["type"]
                line_overview["sigma_v_prior_{}_params".format(l.name)] = l.sigma_v_prior["params"]
            if hasattr(l, "coupled_sigma_v_line"):
                line_overview["sigma_v_prior_{}_cline".format(l.name)] = l.coupled_sigma_v_line.name
            
            # Obtain parameters for each line
            l.parameters = list(key for key in self.get_line_params(self.samples[0], l=l, R=R).keys() if key != "physical")
            
            sample_indices_rank = [np.arange(corei, n_samples, self.mpi_ncores) for corei in range(self.mpi_ncores)]
            line_samples = np.tile(np.nan, (n_samples, len(l.parameters)))

            self.mpi_synchronise(self.mpi_comm)
            for si in sample_indices_rank[self.mpi_rank]:
                line_params = self.get_line_params(self.samples[si], l=l, R=R)
                if line_params["physical"] or l.upper_limit:
                    line_samples[si] = [line_params[param] for param in l.parameters]
            
            self.mpi_synchronise(self.mpi_comm)
            if self.mpi_run:
                # Use gather to concatenate arrays from all ranks on the master rank
                line_samples_full = np.zeros((self.mpi_ncores, n_samples, len(l.parameters))) if self.mpi_rank == 0 else None
                self.mpi_comm.Gather(line_samples, line_samples_full, root=0)
                if self.mpi_rank == 0:
                    for corei in range(1, self.mpi_ncores):
                        line_samples[sample_indices_rank[corei]] = line_samples_full[corei, sample_indices_rank[corei]]
            
            l.line_samples = line_samples
            if self.mpi_rank == 0:
                line_params_perc = np.nanpercentile(l.line_samples, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)], axis=0)
                
                for pi, param in enumerate(l.parameters):
                    line_overview[l.name + "_{}_perc".format(param)] = line_params_perc[:, pi]
                
                if l.upper_limit:
                    assert np.all(np.isnan(line_overview[l.name + "_amplitude_perc"]))
                    flux_uplims = np.tile(np.nan, self.n_obs)
                    wl0_med = line_overview[l.name + "_wl0_perc"][1]
                    
                    for oi in range(self.n_obs):
                        wl_obs = self.wl_obs_list[oi]
                        flux = self.flux_list[oi]
                        flux_err = self.flux_err_list[oi]

                        wl_emit = wl_obs / (1.0 + z)
                        if l.wl > np.max(wl_emit) or l.wl < np.min(wl_emit):
                            continue
                        
                        sig_med = line_overview[l.name + ("_sigma_l_convolved_res{:d}_perc".format(oi) if R is None else "sigma_l_convolved")][1]
                        line_overview[l.name + "_sigma_l_convolved_perc"] = [np.nan, sig_med, np.nan]
                        FWHM_select = (wl_emit >= wl0_med - 1.5*sig_med) * (wl_emit <= wl0_med + 1.5*sig_med)
                        
                        if np.sum(FWHM_select) > 2:
                            if self.no_cont_flux:
                                flux_contsub = flux[FWHM_select] * (1.0 + z)
                            else:
                                flux_contsub = flux[FWHM_select] * (1.0 + z) - self.get_cont_flux(wl_emit=wl_emit[FWHM_select])
                            
                            # Only place upper limit if the line would be resolved
                            dl_emit = np.interp(wl_emit[FWHM_select], 0.5*(wl_emit[FWHM_select][1:]+wl_emit[FWHM_select][:-1]), np.diff(wl_emit[FWHM_select]))
                            flux_uplims[oi] = max(0, np.sum(flux_contsub * dl_emit)) + 3*np.sqrt(np.sum((flux_err[FWHM_select] * dl_emit)**2))
                    
                    line_overview[l.name + "_amplitude_perc"][1] = np.nanmin(flux_uplims)
        
        if self.mpi_rank == 0:
            for l in specific_lines:
                # Calculate observed EW (in the rest frame); use median of underlying continuum
                if l.name == "HI1216":
                    l_min, l_max = 1216.5, 1225.0
                else:
                    l_min = line_overview[l.name + "_wl0_perc"][1] / (1.0 + 1.5*line_overview[l.name + "_sigma_l_convolved_perc"][1]/299792.458)
                    l_max = line_overview[l.name + "_wl0_perc"][1] / (1.0 - 1.5*line_overview[l.name + "_sigma_l_convolved_perc"][1]/299792.458)
                
                line_overview[l.name + "_EW_perc"] = line_overview[l.name + "_amplitude_perc"] / np.median(self.get_cont_flux(wl_emit=np.linspace(l_min, l_max, 10)))

                if hasattr(l, "report_ratios"):
                    num_lines = l.report_ratios.get("num_lines", [[l]] * len(l.report_ratios["den_lines"]))
                    
                    for nline_set, dline_set in zip(num_lines, l.report_ratios["den_lines"]):
                        ratio_kind = "ratio"
                        numerator_samples = np.zeros(n_samples)
                        for li in nline_set:
                            if li not in specific_lines:
                                numerator_samples = np.tile(np.nan, n_samples)
                                continue

                            if li.upper_limit:
                                ratio_kind = "upper_limit"
                                if any(lj.upper_limit for lj in dline_set):
                                    ratio_kind = "undefined"
                                    numerator_samples = np.tile(np.nan, n_samples)
                                    continue
                                numerator_samples += line_overview[li.name + "_amplitude_perc"][1]
                            else:
                                numerator_samples += li.line_samples[:, li.parameters.index("amplitude")]
                        
                        denominator_samples = np.zeros(n_samples)
                        for lj in dline_set:
                            if lj not in specific_lines:
                                denominator_samples = np.tile(np.nan, n_samples)
                                continue

                            if lj.upper_limit:
                                ratio_kind = "lower_limit"
                                denominator_samples += line_overview[lj.name + "_amplitude_perc"][1]
                            else:
                                denominator_samples += lj.line_samples[:, lj.parameters.index("amplitude")]
                        
                        ratio_samples = numerator_samples / denominator_samples
                        
                        ratio_name = '_'.join([li.name for li in nline_set]) + "_to_" + '_'.join([lj.name for lj in dline_set]) + "_ratio"
                        line_overview["report_ratios"].append(ratio_name)
                        line_overview[ratio_name + "_kind"] = ratio_kind
                        line_overview[ratio_name + "_perc"] = np.nanpercentile(ratio_samples, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)], axis=0)
                        line_overview[ratio_name.replace("ratio", "logratio") + "_perc"] = np.nanpercentile(np.log10(ratio_samples), [0.5*(100-68.2689), 50, 0.5*(100+68.2689)], axis=0)
        
        for l in specific_lines:
            if hasattr(l, "line_samples"):
                del l.line_samples
        
        if self.mpi_run:
            line_overview = self.mpi_comm.bcast(line_overview, root=0)
        self.mpi_synchronise(self.mpi_comm)
        
        return line_overview
    
    def get_line_spectra(self, line_overview=None, R=None, R_idx=None):
        assert self.fitting_complete and hasattr(self, "samples")
        n_samples = self.samples.shape[0]
        if R_idx is not None:
            assert R is None
            assert isinstance(R_idx, int)
        
        new_line_overview = line_overview is None

        line_spectra = {}
        for l in self.line_list:
            if new_line_overview:
                # Obtain full samples of the line parameters
                line_overview = self.get_line_overview(specific_lines=[l], R=R)
            
            sigma_l_conv_key = "sigma_l_convolved{}".format('' if R_idx is None else "_res{:d}".format(R_idx))
            sigma_l_conv_perc_key = l.name + "_{}_perc".format(sigma_l_conv_key)

            if not np.isfinite(line_overview[sigma_l_conv_perc_key][1]):
                line_spectra[l.name + "_wl_range_fit" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = np.array([])
                line_spectra[l.name + "_line_spec_median" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = np.array([])
                line_spectra[l.name + "_line_spec_lowerr" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = np.array([])
                line_spectra[l.name + "_line_spec_uperr" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = np.array([])
                continue
            
            wl_range_fit = np.arange(line_overview[l.name + "_wl0_perc"][1] - 5*line_overview[sigma_l_conv_perc_key][1]*sigma2fwhm,
                                        line_overview[l.name + "_wl0_perc"][1] + 5*line_overview[sigma_l_conv_perc_key][1]*sigma2fwhm,
                                        l.wl/10000.0)
            line_spectra[l.name + "_wl_range_fit" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = wl_range_fit
            
            if l.upper_limit:
                # Pass on the upper limit on the line to create the profile
                l.line_params = {key: line_overview[l.name + "_{}_perc".format(key)][1] for key in ["amplitude", "wl0", sigma_l_conv_key]}
                l.line_params["physical"] = np.isfinite(l.line_params["amplitude"]) and np.isfinite(l.line_params["wl0"]) and l.line_params[sigma_l_conv_key] > 0
            
            sample_indices_rank = [np.arange(corei, n_samples, self.mpi_ncores) for corei in range(self.mpi_ncores)]
            line_spec_samples = np.tile(np.nan, (n_samples, wl_range_fit.size))

            self.mpi_synchronise(self.mpi_comm)
            for si in sample_indices_rank[self.mpi_rank]:
                line_spec_samples[si] = self.create_model(self.samples[si], specific_lines=[l], wl_emit=wl_range_fit, R=R, R_idx=R_idx)
            
            self.mpi_synchronise(self.mpi_comm)
            if self.mpi_run:
                # Use gather to concatenate arrays from all ranks on the master rank
                line_spec_samples_full = np.zeros((self.mpi_ncores, n_samples, wl_range_fit.size)) if self.mpi_rank == 0 else None
                self.mpi_comm.Gather(line_spec_samples, line_spec_samples_full, root=0)
                if self.mpi_rank == 0:
                    for corei in range(1, self.mpi_ncores):
                        line_spec_samples[sample_indices_rank[corei]] = line_spec_samples_full[corei, sample_indices_rank[corei]]
            
            if l.upper_limit:
                del l.line_params
            
            if self.mpi_rank == 0:
                line_spec_median = np.median(line_spec_samples, axis=0)
                line_spectra[l.name + "_line_spec_median" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = line_spec_median
                line_spectra[l.name + "_line_spec_lowerr" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = line_spec_median - np.percentile(line_spec_samples, 0.5*(100-68.2689), axis=0)
                line_spectra[l.name + "_line_spec_uperr" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = np.percentile(line_spec_samples, 0.5*(100+68.2689), axis=0) - line_spec_median
        
        if self.mpi_run:
            line_spectra = self.mpi_comm.bcast(line_spectra, root=0)
        self.mpi_synchronise(self.mpi_comm)
        
        return line_spectra

    def create_model(self, theta, z=None, specific_lines=[], wl_emit=None, R=None, R_idx=None):
        bespoke_z = z is not None
        bespoke_wl = wl_emit is not None
        bespoke_R = R is not None or R_idx is not None
        predetermined_wl = not (specific_lines or bespoke_z or bespoke_wl or bespoke_R or not self.fixed_redshift)

        if not bespoke_z:
            if self.fixed_redshift:
                z = self.fixed_redshift
            else:
                assert "redshift" in self.params
                z = theta[self.params.index("redshift")]
        
        if specific_lines:
            line_list = specific_lines
        else:
            line_list = [l for l in self.line_list if not l.upper_limit]
        
        R_idx_key = '' if R_idx is None else "_res{:d}".format(R_idx)
        if bespoke_wl:
            wl_emit_list = [wl_emit]
            if bespoke_R:
                if R_idx is None:
                    assert R is not None
                    spectral_resolution_list = [R if hasattr(R, "__len__") else np.tile(R, wl_emit.size)]
                else:
                    assert R is None
                    spectral_resolution_list = [np.interp(wl_emit * (1.0 + z), *self.get_spectral_resolution(theta, R_idx))]
            else:
                # Only consider the highest resolution as default when specifying a wavelength array
                spectral_resolution_list = [np.interp(wl_emit * (1.0 + z), *self.get_spectral_resolution(theta, np.argmax(self.med_spectral_resolution_list)))]
        else:
            assert R is None and R_idx is None
            wl_emit_list = self.wl_emit_list.copy() if predetermined_wl else [wl_obs / (1.0 + z) for wl_obs in self.wl_obs_list]
            spectral_resolution_list = [self.get_spectral_resolution(theta, oi)[1] for oi in range(self.n_obs)]
        n_models = len(wl_emit_list)

        if predetermined_wl:
            n_res_list = self.n_res_list
            wl_emit_model_list = self.wl_emit_model_list
            model_profile_list = [model_profile.copy() for model_profile in self.model_profile_list]
        else:
            # Create a model wavelength grid covering a spectral resolution element with an adaptively chosen number of bins
            n_res_list = []
            wl_emit_model_list = []

            for oi in range(n_models):
                wl_emit = wl_emit_list[oi]
                spectral_resolution = spectral_resolution_list[oi]

                # If the line is (much) narrower than a spectral resolution element, need to increase the number of wavelength bins
                if self.full_convolution:
                    n_res = self.def_n_res * max(1.0, *[l.wl / np.interp(l.wl, wl_emit, spectral_resolution) / \
                                        self.get_line_params(theta, l=l, R=np.interp(l.wl, wl_emit, spectral_resolution))["sigma_l_convolved"] for l in line_list])
                else:
                    n_res = self.def_n_res
                n_res_list.append(n_res)

                wl_emit_model_list.append(self.get_highres_wl_array([wl_emit], [spectral_resolution], specific_lines=specific_lines, n_res=n_res)[1])

            if specific_lines or self.no_cont_flux:
                model_profile_list = [np.zeros_like(wl_emit_model) for wl_emit_model in wl_emit_model_list]
            else:
                # Add underlying continuum
                model_profile_list = [self.get_cont_flux(wl_emit=wl_emit_model) for wl_emit_model in wl_emit_model_list]
        
        unphysical_model = False
        
        for oi in range(n_models):
            wl_emit_model = wl_emit_model_list[oi]
            model_profile = model_profile_list[oi]

            # Add an (asymmetric) Gaussian component for each line
            for l in line_list:
                if l.upper_limit:
                    assert hasattr(l, "line_params")
                    l.line_params["sigma_l"] = l.line_params["sigma_l_deconvolved" if self.full_convolution else "sigma_l_convolved{}".format(R_idx_key)]
                else:
                    R = np.interp(l.wl, wl_emit_list[oi], spectral_resolution_list[oi])
                    l.line_params = self.get_line_params(theta, l=l, R=R, get_convolved=not self.full_convolution)
                
                if l.line_params["physical"]:
                    # Create line profile in rest frame and add line model to the full model spectrum
                    if (wl_emit_model[-1] < l.line_params["wl0"] - 5.0 * l.line_params["sigma_l"]) or \
                        (wl_emit_model[0] > l.line_params["wl0"] + 5.0 * l.line_params["sigma_l"]):
                        continue
                    
                    idx0 = np.searchsorted(wl_emit_model, l.line_params["wl0"] - 5.0 * l.line_params["sigma_l"], side="left")
                    idx1 = np.searchsorted(wl_emit_model, l.line_params["wl0"] + 5.0 * l.line_params["sigma_l"], side="right")
                    if idx1 > idx0:
                        model_profile[idx0:idx1] += self.line_profile(wl=wl_emit_model[idx0:idx1], l=l)
                    else:
                        assert idx0 == idx1
                        dl_emit = np.interp(wl_emit_model[idx0], 0.5*(wl_emit_model[1:]+wl_emit_model[:-1]), np.diff(wl_emit_model))
                        model_profile[idx0] += l.line_params["amplitude"] / dl_emit
                else:
                    unphysical_model = True
            
            model_profile_list[oi] = model_profile
        
        for oi in range(n_models):
            model_profile = model_profile_list[oi]
            if self.full_convolution:
                # Fully convolve flux profile to simulate instrumental effect, with wavelength-dependent smoothing:
                # standard devation derived from the number of bins covering a resolution element
                model_profile = gaussian_filter1d(model_profile, sigma=n_res_list[oi]/sigma2fwhm, mode="nearest", truncate=4.0)

            if bespoke_wl:
                # Interpolate to specified wavelength array and scale flux density by (1+z) to account for observed redshifting
                model_profile_list[oi] = np.interp(wl_emit_list[oi], wl_emit_model_list[oi], model_profile) / (1.0 + z)
            else:
                # Rebin to observed wavelength array and scale flux density by (1+z) to account for observed redshifting
                model_profile_list[oi] = spectres(wl_emit_list[oi], wl_emit_model_list[oi], model_profile, fill=np.nan, verbose=False) / (1.0 + z)
        
        if bespoke_wl:
            assert len(model_profile_list) == 1
            return model_profile_list[0]
        else:
            return (model_profile_list, unphysical_model)
    
    def set_prior(self):
        self.priors = []
        self.params = []
        
        if not self.fixed_redshift:
            self.priors.append(self.redshift_prior)
            self.params.append("redshift")
        if not self.sigma_prior["type"].lower() == "fixed" and \
            not np.all([hasattr(l, "sigma_v_prior") or hasattr(l, "coupled_sigma_v_line") for l in self.line_list if not l.upper_limit]):
            self.priors.append(self.sigma_prior)
            self.params.append("sigma_v")
        for rsi, res_scale_prior in enumerate(self.res_scale_priors):
            if not res_scale_prior["type"].lower() == "fixed":
                self.priors.append(res_scale_prior)
                self.params.append("res_scale_{:d}".format(rsi))

        for l in self.line_list:
            if not l.upper_limit:
                if hasattr(l, "asymmetric_Gaussian"):
                    self.priors.append({"type": l.asymmetric_Gaussian["prior"]["type"], "params": l.asymmetric_Gaussian["prior"]["params"]})
                    self.params.append("a_asym_{}".format(l.name))
                
                if hasattr(l, "fixed_line_ratio"):
                    assert not hasattr(l, "var_line_ratio")
                elif hasattr(l, "var_line_ratio"):
                    self.priors.append({"type": l.var_line_ratio.get("prior_type", "uniform"), "params": [l.var_line_ratio["ratio_min"], l.var_line_ratio["ratio_max"]]})
                    self.params.append("relative_amplitude_{}".format(l.name))
                else:
                    if hasattr(l, "amplitude_prior"):
                        self.priors.append({"type": l.amplitude_prior["type"], "params": l.amplitude_prior["params"]})
                    else:
                        self.priors.append({"type": "loguniform", "params": [np.log10(self.min_amplitude), np.log10(self.max_amplitude)]})
                    self.params.append("amplitude_{}".format(l.name))
                
                if hasattr(l, "delta_v_prior") and not l.delta_v_prior["type"].lower() == "fixed":
                    self.priors.append(l.delta_v_prior)
                    self.params.append("delta_v_{}".format(l.name))
                
                if hasattr(l, "sigma_v_prior") and not l.sigma_v_prior["type"].lower() == "fixed":
                    self.priors.append(l.sigma_v_prior)
                    self.params.append("sigma_v_{}".format(l.name))

        self.n_dims = len(self.params)
        assert self.n_dims == len(self.priors)

    def Prior(self, cube):
        assert hasattr(self, "priors")
        assert hasattr(self, "params")

        # Scale the input unit cube to apply priors across all parameters
        for di in range(len(cube)):
            if self.priors[di]["type"].lower() == "uniform":
                # Uniform distribution as prior
                cube[di] = cube[di] * (self.priors[di]["params"][1] - self.priors[di]["params"][0]) + self.priors[di]["params"][0]
            elif self.priors[di]["type"].lower() == "loguniform":
                # Log-uniform distribution as prior
                cube[di] = 10**(cube[di] * (self.priors[di]["params"][1] - self.priors[di]["params"][0]) + self.priors[di]["params"][0])
            elif self.priors[di]["type"].lower() == "normal":
                # Normal distribution as prior
                cube[di] = norm.ppf(cube[di], loc=self.priors[di]["params"][0], scale=self.priors[di]["params"][1])
            elif self.priors[di]["type"].lower() == "lognormal":
                # Log-normal distribution as prior
                cube[di] = 10**norm.ppf(cube[di], loc=self.priors[di]["params"][0], scale=self.priors[di]["params"][1])
            elif self.priors[di]["type"].lower() == "gamma":
                # Gamma distribution as prior (prior belief: unlikely to be very high)
                cube[di] = gamma.ppf(cube[di], a=self.priors[di]["params"][0], loc=0, scale=self.priors[di]["params"][1])
            else:
                raise TypeError("prior type '{}' of parameter {:d}, {}, not recognised".format(self.priors[di]["type"], di+1, self.params[di]))
        
        return cube

    def LogLikelihood(self, theta):
        logL = 0.0
        
        model_profile_list, unphysical_model = self.create_model(theta)
        if unphysical_model:
            return -np.inf
        
        for oi in range(self.n_obs):
            if self.fixed_redshift and not self.full_convolution:
                fit_select = self.fit_select_list[oi]
                logL += -0.5 * np.nansum(((self.flux_list[oi][fit_select] - model_profile_list[oi][fit_select]) / self.flux_err_list[oi][fit_select])**2)
            else:
                logL += -0.5 * np.nansum(((self.flux_list[oi] - model_profile_list[oi]) / self.flux_err_list[oi])**2)
        
        return logL
