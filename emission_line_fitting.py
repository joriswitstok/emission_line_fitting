import numpy as np

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
    def __init__(self, line_list, wl_obs_list, flux_list, flux_err_list, cont_flux=None, conv=1, sigma_default=100.0, spectral_resolution_list=None, full_convolution=False,
                    redshift_prior={"type": "uniform", "params": [0, 20]}, sigma_prior={"type": "uniform", "params": [0, 500]}, res_scale_priors=None,
                    mpi_run=False, mpi_comm=None, mpi_rank=0, mpi_ncores=1, mpi_synchronise=None,
                    fit_name=None, print_setup=True, plot_setup=False, mpl_style=None, **solv_kwargs):
        self.mpi_run = mpi_run
        self.mpi_comm = mpi_comm
        self.mpi_rank = mpi_rank
        self.mpi_ncores = mpi_ncores
        self.mpi_synchronise = lambda _: None if mpi_synchronise is None else mpi_synchronise

        self.fit_name = fit_name
        self.mpl_style = mpl_style
        if self.mpi_rank == 0:
            print("Initialising MultiNest Solver object{}{}...".format(" for {}".format(self.fit_name) if self.fit_name else '',
                                                                        " with {} cores".format(self.mpi_ncores) if self.mpi_run else ''))

        self.line_list = line_list
        self.wl_obs_list = wl_obs_list
        self.flux_list = flux_list
        self.flux_err_list = flux_err_list
        self.cont_flux = cont_flux
        self.sigma_default = sigma_default
        
        for l in self.line_list:
            if not hasattr(l, "upper_limit"):
                l.upper_limit = False
        
        self.conv = conv
        self.n_obs = len(self.wl_obs_list)
        assert len(self.flux_list) == self.n_obs and len(self.flux_err_list) == self.n_obs
        
        if spectral_resolution_list is None:
            # Default to a spectral resolution of R = 100000
            spectral_resolution_list = [np.tile(1e5, wl_obs.size) for wl_obs in self.wl_obs_list]
        self.spectral_resolution_list = spectral_resolution_list
        self.med_spectral_resolution_list = [np.median(spectral_resolution) for spectral_resolution in self.spectral_resolution_list]
        
        self.full_convolution = full_convolution
        
        self.redshift_prior = redshift_prior
        self.sigma_prior = sigma_prior
        if res_scale_priors is None:
            self.res_scale_priors = [{"type": "fixed", "params": [1.0]} for _ in range(self.n_obs)]
        else:
            assert len(res_scale_priors) == self.n_obs
            self.res_scale_priors = res_scale_priors

        self.set_prior()

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
            model_list = [self.create_model(params)[0] for params in params_list]

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
            ax_lab.set_ylabel(r"$F_\lambda \, (10^{{{:d}}} \, \mathrm{{erg \, s^{{-1}} \, cm^{{-2}} \, \AA^{{-1}}}})$".format(int(-np.log10(self.conv))), labelpad=30)

            axes[0, 0].annotate(text=r"Intrinsic ($R = 100000$)", xy=(0, 1), xytext=(4, -4),
                                    xycoords="axes fraction", textcoords="offset points",
                                    va="top", ha="left", size="xx-small",
                                    bbox=dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))

            wl_emit = np.arange(np.min([0.95*l.wl for l in self.line_list]), np.max([1.05*l.wl for l in self.line_list]), np.min([l.wl for l in self.line_list])/5000.0)
            
            for coli, params, label in zip(range(n_cols), params_list, labels_list):
                z = params[self.params.index("redshift")] if "redshift" in self.params else self.redshift_prior["params"][0]
                axes[0, coli].plot(wl_emit*(1.0 + z), self.create_model(params, wl_emit=wl_emit, R=1e5), drawstyle="steps-mid",
                                    color=colors[0], alpha=0.8, label=label)

            for ax in axes[0]:
                ax.axhline(y=0, color='k', alpha=0.8)
                ax.legend(loc="upper right", fontsize="xx-small")
            
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
                    
                    ax.plot(wl_obs, model_list[coli][oi], color=colors[0], drawstyle="steps-mid", alpha=0.8)
                    
                    if coli == 1:
                        ax.legend(loc="upper right", fontsize="xx-small")
            
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

    def line_profile(self, x, l, R_idx=None):
        assert hasattr(l, "line_params")
        sigma = l.line_params["sigma_l_deconvolved" if self.full_convolution else "sigma_l_convolved"]
        
        if hasattr(l, "asymmetric_Gaussian"):
            line_f_l_rest = self.asym_Gaussian(A=l.line_params["amplitude"], x0=l.line_params["x0"], y0=0, sigma=sigma,
                                                a_asym=l.line_params["a_asym"], x=x)
        else:
            line_f_l_rest = self.Gaussian(A=l.line_params["amplitude"], x0=l.line_params["x0"], y0=0, sigma=sigma, x=x)
        
        return line_f_l_rest

    def get_cont_flux(self, wl_emit, z=None):
        if self.cont_flux is None:
            cont_flux = np.tile(np.nan, wl_emit.size)
        elif isinstance(self.cont_flux, dict):
            # Model continuum as a power-law slope (in the rest frame)
            cont_flux = np.where(wl_emit < 1215.6701, 0.0, self.cont_flux['C'] * (wl_emit/self.cont_flux["wl0"])**self.cont_flux["beta"])
        else:
            # Calculate observed continuum at given rest-frame wavelengths, then shift flux density into rest frame
            assert z is not None
            cont_flux = self.cont_flux(wl_emit * (1.0 + z)) * (1.0 + z)
        
        return cont_flux
    
    def get_prior_extrema(self, prior):
        if prior["type"].lower() == "uniform":
            minimum = prior["params"][0]
            maximum = prior["params"][1]
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
    
    def get_res_scale(self, theta, oi):
        return 1.0 if self.res_scale_priors[oi]["type"].lower() == "fixed" else theta[self.params.index("res_scale_{:d}".format(oi))]
    
    def get_line_params(self, theta, l, R=None):
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
        line_params["x0"] = l.wl / (1.0 - line_params["dv"]/299792.458)
        
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
        
        # Add instrumental dispersion (if not specified, will take the maximum resolution)
        if R is None:
            z = np.nanmedian(self.samples[:, self.params.index("redshift")]) if "redshift" in self.params else self.redshift_prior["params"][0]
            for oi, wl_obs, spectral_resolution in zip(range(self.n_obs), self.wl_obs_list, self.spectral_resolution_list):
                sigma_instrument = line_params["x0"] / np.interp(line_params["x0"]*(1.0+z), wl_obs, spectral_resolution * self.get_res_scale(theta, oi), left=np.nan, right=np.nan) / (2.0 * np.sqrt(2.0 * np.log(2)))
                line_params["sigma_l_convolved_res{:d}".format(oi)] = np.sqrt(line_params["sigma_l_deconvolved"]**2 + sigma_instrument**2)
            line_params["sigma_l_convolved"] = np.nanmin([line_params["sigma_l_convolved_res{:d}".format(oi)] for oi in range(self.n_obs)])
        else:
            sigma_instrument = line_params["x0"] / R / (2.0 * np.sqrt(2.0 * np.log(2)))
            line_params["sigma_l_convolved"] = np.sqrt(line_params["sigma_l_deconvolved"]**2 + sigma_instrument**2)
        
        line_params["physical"] = np.isfinite(line_params["amplitude"]) and np.isfinite(line_params["x0"]) and line_params["sigma_l_convolved"] > 0

        return line_params

    def get_line_overview(self, specific_lines=[], R=None):
        assert self.fitting_complete and hasattr(self, "samples")
        n_samples = self.samples.shape[0]
        z = np.nanmedian(self.samples[:, self.params.index("redshift")]) if "redshift" in self.params else self.redshift_prior["params"][0]

        specific_lines = specific_lines if specific_lines else self.line_list
        line_overview = {"line_names": [l.name for l in specific_lines], "line_uplims": [l.upper_limit for l in specific_lines],
                            "asymmetric_Gaussian_lines": {l.name for l in specific_lines if hasattr(l, "asymmetric_Gaussian")},
                            "fixed_line_ratios": {l.name: l.fixed_line_ratio for l in specific_lines if hasattr(l, "fixed_line_ratio")},
                            "var_line_ratios": {l.name: l.var_line_ratio for l in specific_lines if hasattr(l, "var_line_ratio")},
                            "delta_v_priors": {l.name: l.delta_v_prior for l in specific_lines if hasattr(l, "delta_v_prior")},
                            "coupled_delta_v_lines": {l.name: l.coupled_delta_v_line for l in specific_lines if hasattr(l, "var_line_ratio")},
                            "sigma_v_priors": {l.name: l.sigma_v_prior for l in specific_lines if hasattr(l, "sigma_v_prior")},
                            "coupled_sigma_v_lines": {l.name: l.coupled_sigma_v_line for l in specific_lines if hasattr(l, "var_line_ratio")},
                            "report_ratios": []}
        
        for l in specific_lines:
            # Obtain parameters for each line
            l.parameters = list(self.get_line_params(self.samples[0], l, R=R).keys())
            
            sample_indices_rank = [np.arange(corei, n_samples, self.mpi_ncores) for corei in range(self.mpi_ncores)]
            line_samples = np.tile(np.nan, (n_samples, len(l.parameters)))

            self.mpi_synchronise(self.mpi_comm)
            for si in sample_indices_rank[self.mpi_rank]:
                line_params = self.get_line_params(self.samples[si], l, R=R)
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
                    dv_med = np.nanmedian(l.line_samples[:, l.parameters.index("dv")])
                    x0_med = l.wl / (1.0 - dv_med/299792.458)
                    
                    for oi, wl_obs, flux_err in zip(range(self.n_obs), self.wl_obs_list, self.flux_err_list):
                        wl_emit = wl_obs / (1.0 + z)
                        if l.wl > np.max(wl_emit) or l.wl < np.min(wl_emit):
                            continue
                        
                        sig_med = np.nanmedian(l.line_samples[:, l.parameters.index("sigma_l_convolved_res{:d}".format(oi) if R is None else "sigma_l_convolved")])
                        FWHM_select = (wl_emit >= x0_med - 1.5*sig_med) * (wl_emit <= x0_med + 1.5*sig_med)
                        
                        if np.sum(FWHM_select) > 2:
                            # Only place upper limit if the line would be resolved
                            dl_obs = np.interp(wl_obs, 0.5*(wl_obs[1:]+wl_obs[:-1]), np.diff(wl_obs))
                            flux_uplims[oi] = np.mean((flux_err * dl_obs)[FWHM_select]) * np.sqrt(np.sum(FWHM_select))
                    
                    line_overview[l.name + "_amplitude_perc"][1] = np.nanmin(flux_uplims)
        
        if self.mpi_rank == 0:
            for l in specific_lines:
                # Calculate observed EW (in the rest frame)
                line_overview[l.name + "_EW_perc"] = line_overview[l.name + "_amplitude_perc"] / self.get_cont_flux(wl_emit=l.wl, z=z)

                if hasattr(l, "report_ratios"):
                    ratio_kind = "ratio"
                    num_lines = l.report_ratios.get("num_lines", [[l]] * len(l.report_ratios["den_lines"]))
                    
                    for nline_set, dline_set in zip(num_lines, l.report_ratios["den_lines"]):
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
                        logratio_samples = np.log10(ratio_samples)
                        
                        ratio_name = '_'.join([li.name for li in nline_set]) + "_to_" + '_'.join([lj.name for lj in dline_set]) + "_ratio"
                        line_overview["report_ratios"].append(ratio_name)
                        line_overview[ratio_name + "_kind"] = ratio_kind
                        line_overview[ratio_name + "_perc"] = np.nanpercentile(ratio_samples, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)], axis=0)
                        line_overview[ratio_name.replace("ratio", "logratio") + "_perc"] = np.nanpercentile(logratio_samples, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)], axis=0)
        
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

        line_spectra = {}
        for l in self.line_list:
            if line_overview is None:
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
            
            wl_range_fit = np.arange(line_overview[l.name + "_x0_perc"][1] - 5*line_overview[sigma_l_conv_perc_key][1]*(2.0*np.sqrt(2.0*np.log(2))),
                                        line_overview[l.name + "_x0_perc"][1] + 5*line_overview[sigma_l_conv_perc_key][1]*(2.0*np.sqrt(2.0*np.log(2))),
                                        l.wl/10000.0)
            line_spectra[l.name + "_wl_range_fit" + ('' if R_idx is None else "_res{:d}".format(R_idx))] = wl_range_fit
            
            if l.upper_limit:
                # Pass on the upper limit on the line to create the profile
                l.line_params = {key: line_overview[l.name + "_{}_perc".format(key)][1] for key in ["amplitude", "x0", sigma_l_conv_key]}
                l.line_params["physical"] = np.isfinite(l.line_params["amplitude"]) and np.isfinite(l.line_params["x0"]) and l.line_params[sigma_l_conv_key] > 0
            
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
        if "redshift" in self.params:
            z = theta[self.params.index("redshift")]
        else:
            assert self.redshift_prior["type"].lower() == "fixed"
            z = self.redshift_prior["params"][0]
        
        if specific_lines:
            line_list = specific_lines
        else:
            line_list = [l for l in self.line_list if not l.upper_limit]
        
        bespoke_wl = wl_emit is not None
        bespoke_R = R is not None
        if bespoke_wl:
            if bespoke_R:
                assert R_idx is None
                wl_emit_list = [wl_emit]
                spectral_resolution_list = [R if hasattr(R, "__len__") else np.tile(R, wl_emit.size)]
            else:
                wl_emit_list = []
                spectral_resolution_list = []
                for oi, wl_obs, spectral_resolution in zip(range(self.n_obs), self.wl_obs_list, self.spectral_resolution_list):
                    if (R_idx is None and oi != np.argmax(self.med_spectral_resolution_list)) or (R_idx is not None and oi != R_idx):
                        # Only consider the specified resolution (highest as default) when specifying a wavelength array
                        continue
                    wl_emit_list.append(wl_emit)
                    spectral_resolution_list.append(np.interp(wl_emit * (1.0 + z), wl_obs, spectral_resolution * self.get_res_scale(theta, oi)))
        else:
            assert R is None and R_idx is None
            wl_emit_list = [wl_obs / (1.0 + z) for wl_obs in self.wl_obs_list]
            spectral_resolution_list = [spectral_resolution * self.get_res_scale(theta, oi) for oi, spectral_resolution in zip(range(self.n_obs), self.spectral_resolution_list)]

        if self.full_convolution:
            # Create a model wavelength grid covering a spectral resolution element with an adaptively chosen number of bins
            n_res_list = []
            wl_emit_model_list = []

            for oi, wl_emit, spectral_resolution in zip(range(self.n_obs), wl_emit_list, spectral_resolution_list):
                # If the line is (much) narrower than a spectral resolution element, need to increase the number of wavelength bins
                n_res = 3.0 * max(1.0, *[l.wl / np.interp(l.wl, wl_emit, spectral_resolution) / \
                                            self.get_line_params(theta, l, R=np.inf)["sigma_l_deconvolved"] for l in line_list])

                if wl_emit.size > 1:
                    # Check that wavelength array is ascending
                    assert np.all(np.diff(wl_emit) > 0)

                wl_emit_bin_edges = []
                wl = wl_emit[0] - 15 * wl_emit[0] / np.min(spectral_resolution)
                
                while wl < wl_emit[-1] + 15 * wl_emit[-1] / np.min(spectral_resolution):
                    wl_emit_bin_edges.append(wl)
                    wl += wl / (n_res * np.interp(wl, wl_emit, spectral_resolution))
                
                wl_emit_bin_edges = np.array(wl_emit_bin_edges)
                assert wl_emit_bin_edges.size > 1
                
                n_res_list.append(n_res)
                wl_emit_model_list.append(0.5 * (wl_emit_bin_edges[:-1] + wl_emit_bin_edges[1:]))
        else:
            n_res_list = len(wl_emit_list) * [np.nan]
            wl_emit_model_list = wl_emit_list
        
        model_profile_list = [np.zeros(wl_emit_model.size) for wl_emit_model in wl_emit_model_list]
        unphysical_model = False
        
        if not specific_lines and isinstance(self.cont_flux, dict):
            for oi, wl_emit_model in zip(range(len(wl_emit_list)), wl_emit_model_list):
                # Add underlying continuum
                model_profile_list[oi] = model_profile_list[oi] + self.get_cont_flux(wl_emit=wl_emit_model)
        
        # Add an (asymmetric) Gaussian component for each line
        for l in line_list:
            for oi, wl_emit, wl_emit_model, spectral_resolution in zip(range(len(wl_emit_list)), wl_emit_list, wl_emit_model_list, spectral_resolution_list):
                R = np.interp(l.wl, wl_emit, spectral_resolution)
                if l.upper_limit:
                    assert hasattr(l, "line_params")
                    if R_idx is not None:
                        l.line_params["sigma_l_convolved"] = l.line_params["sigma_l_convolved{}".format('' if R_idx is None else "_res{:d}".format(R_idx))]
                else:
                    l.line_params = self.get_line_params(theta, l, R)
                
                if l.line_params["physical"]:
                    # Create line profile in rest frame and add line model to the full model spectrum
                    model_profile_list[oi] = model_profile_list[oi] + self.line_profile(x=wl_emit_model, l=l)
                else:
                    unphysical_model = True
        
        for oi, wl_emit, wl_emit_model, n_res in zip(range(len(wl_emit_list)), wl_emit_list, wl_emit_model_list, n_res_list):
            if self.full_convolution:
                # Fully convolve flux profile to simulate instrumental effect, with wavelength-dependent smoothing:
                # standard devation derived from the number of bins covering a resolution element
                model_profile_list[oi] = gaussian_filter1d(model_profile_list[oi], sigma=n_res/(2.0 * np.sqrt(2.0 * np.log(2))),
                                                            mode="nearest", truncate=5.0)
            
            # Scale flux density by (1+z) to account for observed redshifting
            model_profile_list[oi] = model_profile_list[oi] / (1.0 + z)
            
            # Rebin to observed wavelength array
            model_profile_list[oi] = spectres(wl_emit, wl_emit_model, model_profile_list[oi])
        
        if bespoke_wl:
            assert len(model_profile_list) == 1
            return model_profile_list[0]
        else:
            return (model_profile_list, unphysical_model)
    
    def set_prior(self):
        self.priors = []
        self.params = []
        
        if not self.redshift_prior["type"].lower() == "fixed":
            self.priors.append(self.redshift_prior)
            self.params.append("redshift")
        if not self.sigma_prior["type"].lower() == "fixed" and \
            not np.all([hasattr(l, "sigma_v_prior") or hasattr(l, "coupled_sigma_v_line") for l in self.line_list if not l.upper_limit]):
            self.priors.append(self.sigma_prior)
            self.params.append("sigma_v")
        for oi, res_scale_prior in enumerate(self.res_scale_priors):
            if not res_scale_prior["type"].lower() == "fixed":
                self.priors.append(res_scale_prior)
                self.params.append("res_scale_{:d}".format(oi))

        for l in self.line_list:
            if not l.upper_limit:
                z_min, z_max = self.get_prior_extrema(self.redshift_prior)
                
                if hasattr(l, "asymmetric_Gaussian"):
                    self.priors.append({"type": l.asymmetric_Gaussian["prior"]["type"], "params": l.asymmetric_Gaussian["prior"]["params"]})
                    self.params.append("a_asym_{}".format(l.name))
                
                if hasattr(l, "fixed_line_ratio"):
                    assert not hasattr(l, "var_line_ratio")
                elif hasattr(l, "var_line_ratio"):
                    self.priors.append({"type": "uniform", "params": [l.var_line_ratio["ratio_min"], l.var_line_ratio["ratio_max"]]})
                    self.params.append("relative_amplitude_{}".format(l.name))
                else:
                    max_amplitude = None
                    for wl_obs, flux, spectral_resolution in zip(self.wl_obs_list, self.flux_list, self.spectral_resolution_list):
                        R_min = np.min(spectral_resolution) * self.get_prior_extrema(self.res_scale_priors[oi])[0]
                        dv_min, dv_max = self.get_prior_extrema(l.delta_v_prior) if hasattr(l, "delta_v_prior") else (-max(3000.0, 3*299792.458/R_min), max(3000.0, 3*299792.458/R_min))
                        wl_obs_min = (1.0 + z_min) * l.wl / (1.0 - dv_min/299792.458)
                        wl_obs_minidx = max(0, np.argmin(np.abs(wl_obs - wl_obs_min)) - 1)
                        wl_obs_max = (1.0 + z_max) * l.wl / (1.0 - dv_max/299792.458)
                        wl_obs_maxidx = min(wl_obs.size - 1, np.argmin(np.abs(wl_obs - wl_obs_max)) + 1)

                        if wl_obs_min < np.max(wl_obs) and wl_obs_max > np.min(wl_obs):
                            sigma_prior = l.sigma_v_prior if hasattr(l, "sigma_v_prior") else self.sigma_prior
                            sigma_max = (1.0 + z_max) * np.sqrt((l.wl / (299792.458/self.get_prior_extrema(sigma_prior)[1] - 1.0))**2 + (l.wl / R_min / (2.0 * np.sqrt(2.0 * np.log(2))))**2)
                
                            max_flux = np.max([np.nanmax(flux[wl_obs_minidx:wl_obs_maxidx+1])])
                            max_amplitude = max_flux * sigma_max * 2.0 * np.sqrt(2.0 * np.log(2))
                    
                    if max_amplitude:
                        self.priors.append({"type": "uniform", "params": [0, max_amplitude]})
                        self.params.append("amplitude_{}".format(l.name))
                    else:
                        raise ValueError("Warning: {} is not observed in the provided spectrum".format(l.estlabel))
                
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
        
        for flux, model_profile, flux_err in zip(self.flux_list, model_profile_list, self.flux_err_list):
            logL += -0.5 * np.nansum(((flux - model_profile) / flux_err)**2)
        
        return logL
