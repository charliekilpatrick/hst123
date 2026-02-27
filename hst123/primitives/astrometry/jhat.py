"""
JHAT primitive: run JHAT to align JWST images to Gaia or a JWST source catalog.

Extracted from jwst123 (https://github.com/aswinsuresh24/jwst123).
Requires the optional `jhat` package (not installed with hst123 by default).
"""


def run_jhat(
    align_image,
    outdir,
    params,
    gaia=False,
    photfilename=None,
    xshift=0,
    yshift=0,
    Nbright=800,
    verbose=False,
):
    """
    Run JHAT to align a JWST image to Gaia or a JWST source catalog.

    Parameters
    ----------
    align_image : str
        Image to align.
    outdir : str
        Output directory.
    params : dict
        Parameters for JHAT (e.g. strict_gaia_params, strict_jwst_params).
        Passed as keyword arguments to ``st_wcs_align().run_all()``.
    gaia : bool, optional
        If True, align to Gaia. Default is False.
    photfilename : str, optional
        Photometry file name (required when gaia is False).
    xshift, yshift : float, optional
        x and y shift in pixels. Default 0.
    Nbright : int, optional
        Number of bright stars to use. Default 800.
    verbose : bool, optional
        Verbose output. Default False.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If the `jhat` package is not installed.
    ValueError
        If gaia is False and photfilename is None.
    """
    try:
        from jhat import st_wcs_align
    except ImportError as e:
        raise ImportError(
            "run_jhat requires the jhat package. Install with: pip install jhat"
        ) from e

    wcs_align = st_wcs_align()
    if gaia:
        wcs_align.run_all(
            align_image,
            outsubdir=outdir,
            refcatname="Gaia",
            pmflag=True,
            use_dq=False,
            verbose=verbose,
            xshift=xshift,
            yshift=yshift,
            **params,
        )
    else:
        if photfilename is None:
            raise ValueError("Input photometric catalog is required when gaia=False")
        wcs_align.run_all(
            align_image,
            outsubdir=outdir,
            refcatname=photfilename,
            use_dq=False,
            verbose=verbose,
            xshift=xshift,
            yshift=yshift,
            Nbright=Nbright,
            **params,
        )
