"""Basic class for fMRI data readers.
"""


class FmriReader(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_all_events(self, subject_ids=None, **kwargs):
        raise NotImplementedError()
