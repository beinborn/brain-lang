import nilearn.signal

def clean(data):
    # check data format
    return nilearn.signal.clean(data, sessions=None,
                         detrend=True, standardize=True,
                         confounds=None, low_pass=None,
                         high_pass=0.005, t_r=2.0, ensure_finite=False)



def standardize(data):
    # TODO standardize data to have mean 0
    return data
