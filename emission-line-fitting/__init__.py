try:
    from pymultinest.solve import Solver
except:
    import warnings
    warnings.warn(
        'pymultinest not installed. Fitting not available', UserWarning)
else:
    from . import emission_line_fitting