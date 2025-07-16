import jax
import jaxopt
import numpy as np
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels, transforms

jax.config.update("jax_enable_x64", True)

def prepare_gp_inputs(times, wavelengths, fluxes, flux_errors):
    X = (times, wavelengths)

    # normalise fluxes - values have to be ideally above zero
    y_norm = np.copy(fluxes.max()) 
    y = (fluxes / y_norm).copy() 
    yerr = (flux_errors / y_norm).copy()

    return X, y, yerr, y_norm
    
def fit_gp_model(times, wavelengths, fluxes, flux_errors, k1='Matern52', fit_mean=False, 
                 time_scale=None, wave_scale=None, add_noise=True):
    
    assert k1 in ['Matern52', 'Matern32', 'ExpSquared'], "Not a valid kernel"
    def build_gp(params):
        """Creates a Gaussian Process model.
        """
        nonlocal time_scale, wave_scale, add_noise  # import from main function
        if time_scale is None:
            log_time_scale = params["log_scale"][0]
        else:
            log_time_scale = np.log(time_scale)
        if wave_scale is None:
            log_wave_scale = params["log_scale"][-1]
        else:
            log_wave_scale = np.log(wave_scale)
        if add_noise is True:
            noise = jnp.exp(2 * params["log_noise"])
        else:
            noise = 0.0

        # select time-axis kernel
        if k1 == 'Matern52':
            kernel1 = transforms.Subspace(0, kernels.Matern52(scale=jnp.exp(log_time_scale)))
        elif k1 == 'Matern32':
            kernel1 = transforms.Subspace(0, kernels.Matern32(scale=jnp.exp(log_time_scale)))
        else:
            kernel1 = transforms.Subspace(0, kernels.ExpSquared(scale=jnp.exp(log_time_scale)))
        # wavelength-axis kernel
        kernel2 = transforms.Subspace(1, kernels.ExpSquared(scale=jnp.exp(log_wave_scale)))
        
        kernel = jnp.exp(params["log_amp"]) * kernel1 * kernel2
        diag = yerr ** 2 + noise
        
        if fit_mean is True:
            mean = jnp.exp(params["log_mean"])
        else:
            mean = None

        return GaussianProcess(kernel, X, diag=diag, mean=mean)

    @jax.jit
    def loss(params):
        """Loss function for the Gaussian Process hyper-parameters optimisation.
        """
        return -build_gp(params).condition(y).log_probability
    
    X, y, yerr, _ = prepare_gp_inputs(times, wavelengths, fluxes, flux_errors)

    # GP hyper-parameters
    scales = np.array([30, 10_000]) # units: days, angstroms
    if time_scale is not None:
        scales = np.delete(scales, 0)
    if wave_scale is not None:
        scales = np.delete(scales, -1)
    
    params = {
        "log_amp": jnp.log(y.var()),
        "log_noise": jnp.log(np.mean(yerr)),
    }
    if len(scales) != 0:
        params.update({"log_scale": jnp.log(scales)})
    if fit_mean is True:
        params.update({"log_mean": jnp.log(np.average(y, weights=1/yerr**2))})

    # Train the GP model
    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params)
    gp_model = build_gp(soln.params)
    gp_model.__dict__["init_params"] = params
    gp_model.__dict__["params"] = soln.params
    
    return gp_model

def gp_predict(times_pred, wavelengths_pred, ratio_pred, error_pred, gp_model, return_cov: bool = False):
    X_test, y, _, y_norm = prepare_gp_inputs(times_pred, wavelengths_pred, 
                                             ratio_pred, error_pred, 
                                            )
    if return_cov is True:
        return_var = False
    else:
        return_var = True
            
    # GP prediction
    mu, cov = gp_model.predict(y, X_test=X_test, return_var=return_var, return_cov=True)
    # renormalise outputs and convert jax-Array to numpy-array
    mu = np.array(mu) * y_norm
    cov = np.array(cov) * (y_norm ** 2)

    return mu, cov