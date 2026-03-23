from astropy.io import fits


class HSTModel:
    """
    Base class for importing HST data.
    """


    def __init__(self, filepath, meta : dict | None = None):
        
        self.filepath = filepath

        self.init_meta(meta=meta)

        self.on_init()

    @property
    def dirname(self):
        """
        Returns the directory of the HSTModel if present.
        """
        return os.path.dirname(self.filepath) if self.filepath else None

    @property
    def basename(self):
        """
        Returns the base filename of the HSTModel if present.
        """
        return os.path.basename(self.filepath) if self.filepath else None

    def init_meta(self, meta : dict | None = None):
        """
        Initialize the metadata and optionally merge in provided metadata.

        Args:
            meta (dict, optional): Metadata to merge in. Defaults to None.
        """
        self.meta = {}
        if meta is not None:
            self.meta.update(meta)

    def on_init(self):
        """
        Hook called after the HSTModel is initialized.
        """
        self.meta.model_type = self.__class__.__name__
        if self.filepath is not None:
            self.meta.filename = os.path.basename(self.filepath)

