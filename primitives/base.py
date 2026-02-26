"""
Base primitive class for all pipeline helper primitives.

Every helper (FitsHelper, PhotometryHelper, etc.) inherits from BasePrimitive
and receives the main hst123 pipeline instance so it can read options, state,
and delegate back to the pipeline when needed.
"""


class BasePrimitive:
    """
    Base class for primitive helpers used by the hst123 pipeline.

    Subclasses receive the pipeline instance and store it as _p. Use
    self._p to access pipeline options (e.g. self._p.options), state
    (e.g. self._p.coord), or other helpers (e.g. self._p._fits).
    """

    def __init__(self, pipeline):
        if pipeline is None:
            raise TypeError("BasePrimitive requires a pipeline instance")
        self._p = pipeline

    @property
    def pipeline(self):
        """The hst123 pipeline instance this primitive is attached to."""
        return self._p
