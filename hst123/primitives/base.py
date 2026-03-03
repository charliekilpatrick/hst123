"""Base class for pipeline primitives; subclasses get the pipeline as self._p."""


class BasePrimitive:
    """Primitive helpers attach to the pipeline and use self._p for options/state."""

    def __init__(self, pipeline):
        if pipeline is None:
            raise TypeError("BasePrimitive requires a pipeline instance")
        self._p = pipeline

    @property
    def pipeline(self):
        """Pipeline instance (same as self._p)."""
        return self._p
