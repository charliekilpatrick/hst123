"""
Base class for pipeline *primitives* (FITS, astrometry, DOLPHOT, etc.).

Each primitive receives the live :class:`~hst123._pipeline.hst123` instance as
``self._p`` and should use :meth:`_primitive_cleanup` after writing outputs.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from hst123.utils.logging import get_logger


class BasePrimitive:
    """
    Attach to the pipeline and share options/state via ``self._p``.

    Parameters
    ----------
    pipeline
        The running :class:`~hst123._pipeline.hst123` instance (required).

    Attributes
    ----------
    _p
        Same object as :attr:`pipeline`.
    _log
        Logger named after the primitive module (``hst123.primitives…``).

    Raises
    ------
    TypeError
        If ``pipeline`` is None.
    """

    def __init__(self, pipeline):
        if pipeline is None:
            raise TypeError("BasePrimitive requires a pipeline instance")
        self._p = pipeline
        self._log = get_logger(self.__class__.__module__)
        self._log.debug("attached %s", self.__class__.__name__)

    @property
    def pipeline(self):
        """
        Pipeline instance (alias of ``self._p``).

        Returns
        -------
        object
        """
        return self._p

    def _primitive_cleanup(
        self,
        step_name: str,
        *,
        work_dir: str | None = None,
        remove_globs: Sequence[str] = (),
        remove_paths: Sequence[str] = (),
        validate_fits_paths: Sequence[str] = (),
        validate_text_paths: Sequence[str] = (),
        require_basic_wcs: bool = False,
        text_min_size: int = 1,
        skip_interstitial_removal: bool = False,
        validation_notes: dict[str, Any] | None = None,
        custom_validators: Sequence[Callable[[], None]] = (),
        validate_tables: Sequence[Any] | None = None,
        respect_keep_artifacts: bool = True,
    ) -> None:
        """
        After a primitive step: optionally remove interstitial files, validate outputs.

        All actions are logged under ``[primitive cleanup] <step_name>: ...`` for
        transparency. Subclasses should call this at the end of methods that
        write FITS, catalogs, or temporary files.

        Parameters
        ----------
        work_dir
            Directory for glob-based interstitial removal (default: pipeline work_dir).
        respect_keep_artifacts
            When True and ``--keep-drizzle-artifacts`` is set, interstitial removal
            is skipped (validation still runs).
        """
        from hst123.primitives.primitive_cleanup import run_primitive_cleanup

        wd = work_dir
        if wd is None:
            try:
                wd = getattr(self._p.options["args"], "work_dir", None) or None
            except Exception:
                wd = None

        run_primitive_cleanup(
            self._log,
            step_name,
            work_dir=wd,
            remove_globs=remove_globs,
            remove_paths=remove_paths,
            validate_fits_paths=validate_fits_paths,
            validate_text_paths=validate_text_paths,
            require_basic_wcs=require_basic_wcs,
            text_min_size=text_min_size,
            skip_interstitial_removal=skip_interstitial_removal,
            validation_notes=validation_notes,
            custom_validators=custom_validators,
            validate_tables=validate_tables,
            pipeline=self._p,
            respect_keep_artifacts=respect_keep_artifacts,
        )
