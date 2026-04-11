import string
from numbers import Integral
from enum import IntFlag

import numpy as np
from astropy import wcs as pywcs
from astropy.io import fits
from stsci.tools import fileutil as fu

from astropy import log


default_log_level = log.getEffectiveLevel()


__all__ = ["archive_wcs", "ArchiveMode", "available_wcskeys", "deleteWCS",
           "next_wcskey", "pc2cd", "restoreWCS", "wcskeys", "wcsnames",
           "wcs_from_key"]


altwcskw = ['WCSAXES', 'CRVAL', 'CRPIX', 'PC', 'CDELT', 'CD', 'CTYPE', 'CUNIT',
            'PV', 'PS']

# List non-standard WCS keywords (such as those created, e.g., by TweakReg)
# that need to be archived/restored with the rest of WCS here:
STWCS_KWDS = ['WCSTYPE', 'RMS_RA', 'RMS_DEC', 'NMATCH', 'FITNAME', 'HDRNAME']

# These are keywords that may be present in headers created by WCSLIB.
# However, HST keeps them (or similar information) in the primary header and so
# they should not be managed by STWCS when copying/archiving an HST WCS:
EXCLUDE_WCSLIB_KWDS = ['EQUINOX', 'LONPOLE', 'LATPOLE', 'RESTWAV', 'RESTFRQ']

_DEFAULT_WCSNAME = 'ARCHIVED_WCS'

# file operations


class ArchiveMode(IntFlag):
    NO_CONFLICT = 0  # Fail if WCS key or name already exists (unless identical WCS)
    AUTO_RENAME = 1  # rename primary WCS if same name is already used by an Alt WCS (to keep Alt WCS names unique)
    OVERWRITE_KEY = 2  # overwrite an existing Alt WCS
    QUIET_ABORT = 4  # quit if conflicts without raising exceptions

AM = ArchiveMode


def archive_wcs(fname, ext, wcskey=None, wcsname=None, mode=ArchiveMode.NO_CONFLICT):
    """
    Copy the primary WCS to the header as an alternate WCS
    with the specified ``wcskey`` and name ``wcsname``.
    It loops over all extensions in 'ext'.

    Parameters
    ----------
    fname :  string or `astropy.io.fits.HDUList`
        File name or a `astropy.io.fits.HDUList` whose primary WCS need to be
        archived.

    ext : int, tuple, str, or list of integers or tuples (e.g.('sci',1))
        Specifies FITS extensions whose primary WCS should be archived.

        If a string is provided, it should specify the EXTNAME of extensions
        with WCSs to be archived

    wcskey : {'A'-'Z'}, None, optional
        When ``wcskey`` is `None`, and ``wcsname`` is not `None` and an
        alternate WCS exist with the same name as ``wcsname``, the key of
        this alernate WCS will be used for ``wcskey``. If ``wcsname`` is unique,
        the next available key will be used. If ``wcskey`` is a letter ``'A'-'Z'``,
        the primary WCS will be archived to the indicated key. If an
        alternate WCS with the specified key already exist an exception will be
        raisen unless mode includes either flag ``ArchiveMode.OVERWRITE_KEY`` or
        ``ArchiveMode.QUIET_ABORT``.

    wcsname : str, None, optional
        Name of alternate WCS description. If `None`, the WCS name will be
        initially set to the name of the primary WCS if available. If primary
        WCS does not have ``'WCSNAME'`` set and provided ``wcskey`` is not
        `None` and a WCS with this key already exists in the header, the name
        of this alternate WCS will be used. I this alternate WCS does not have
        ``'WCSNAME'`` set and provided, a default WCS name will be used.

        .. note::
            An exception will be raisen if an alternate WCS with provided
            name already exists in image's headers unless ``mode`` includes
            ``ArchiveMode.AUTO_RENAME`` flag in which case ``wcsname`` will be
            automatically adjusted to be unique.

    mode : int, optional
        An integer mask containing one or more flags from ``ArchiveMode``:

        - ``NO_CONFLICT``: Archive only when provided WCS key or name
          do not conflict with existing alternate WCS or when existing
          alternate WCS is identical to the primary WCS being archived.

        - ``AUTO_RENAME``: Rename primary WCS if same name is already used
          by an alternate WCS (in order to keep alternate WCS names unique).

        - ``OVERWRITE_KEY``: Overwrite an existing alternate WCS with the
          same key as ``wcskey``.

        - ``QUIET_ABORT``: Stop archiving and quit if conflicts exists
          without raising exceptions.

    Returns
    -------
    wcs_id : tuple (str or None, str or None)
        Returns effective WCS key and name used for the archived WCS.
        If archival failed due to conflicts, and ``mode`` was set to
        ``ArchiveMode.QUIET_ABORT`` a tuple of ``(None, None)`` will be returned.

    """
    if wcsname is not None and not isinstance(wcsname, str):
        raise TypeError("'wcsname' must be a string")

    if (wcskey is not None and
            ((not isinstance(wcskey, str) or len(wcskey) != 1 or
              wcskey.strip() not in string.ascii_uppercase))):
        raise ValueError(
            "Parameter 'wcskey' must be a character - one of 'A'-'Z' or ' '."
        )

    if isinstance(fname, str):
        # open file:
        h = fits.open(fname, mode='update')
        close_hdulist = True
    else:
        h = fname
        close_hdulist = False

        # validate file object and open mode:
        if not isinstance(h, fits.HDUList):
            if close_hdulist:
                h.close()
            raise TypeError("'fname' must be either a FITS file object or a string file name.")

        if h.fileinfo(0) is not None and h.fileinfo(0)['filemode'] != 'update':
            if close_hdulist:
                h.close()
            raise ValueError("'fname' must be a file object opened in 'update' mode.")

    # validate and interpret extension(s):
    try:
        ext = _buildExtlist(h, ext)
    except ValueError as e:
        if close_hdulist:
            h.close()
        raise e

    wcsext = ext[0]
    hdr = h[wcsext].header

    wcs_names = _alt_wcs_names(hdr)
    wcs_names_u = {k: v.upper() for k, v in wcs_names.items()}
    wcsname_u = None if wcsname is None else wcsname.upper()
    wcs_keys = wcskeys(h, wcsext)

    # validate and process mode:
    if mode & ~sum(AM):
        raise ValueError("Unrecognized 'mode' flag.")
    overwrite_key = bool(mode & AM.OVERWRITE_KEY)
    auto_rename = bool(mode & AM.AUTO_RENAME)

    wcs_id = (None, None)

    # do not overwrite OPUS WCS if present regardless of 'mode':
    if wcskey == 'O' and wcskey in wcs_keys:
        if close_hdulist:
            h.close()
        log.debug("WCS name 'OPUS' already exists and cannot be overwritten.")
        return wcs_id

    elif wcsname == 'OPUS' and wcsname in wcs_names_u.values():
        if close_hdulist:
            h.close()
        log.debug("WCS name 'OPUS' already exists and cannot be overwritten.")
        return wcs_id

    if wcsname is None:
        if 'WCSNAME' in hdr:
            # use WCS name of the primary WCS if available:
            wcsname = hdr['WCSNAME']
            if not wcsname.strip():
                wcsname = _DEFAULT_WCSNAME
                auto_rename = True

        elif wcs_names:
            # use the name of the Alt WCS with largest key:
            wcsname = wcs_names[max(wcs_names.keys())]
            auto_rename = True

        else:
            # assign default name
            wcsname = _DEFAULT_WCSNAME
            auto_rename = True

    elif wcsname_u in wcs_names_u.values():
        if (wcskey is not None and wcskey in wcs_names and
                wcsname_u != wcs_names_u[wcskey] and not auto_rename):
            msg = (
                "'wcsname' must be unique in image header. Provided 'wcsname' "
                "was already used for an alternate WCS with a different key "
                "than the supplied 'wcskey'. Add ArchiveMode.AUTO_RENAME flag "
                "to 'mode' in order to allow 'archive_wcs()' to modify "
                "'wcsname' to be unique."
            )

            if AM.QUIET_ABORT:
                log.warning(msg)
                return wcs_id
            else:
                raise ValueError(msg)

        elif wcskey is None:
            # pick the key that corresponds to the provided WCS name:
            key_match = sorted([k for k, v in wcs_names_u.items() if v == wcsname_u])[-1]

            if _test_wcs_equal(h, wcsext, ' ', key_match) or overwrite_key:
                wcskey = key_match

            elif not auto_rename:
                msg = (
                    "'wcsname' must be unique in image header. Provided 'wcsname' "
                    "was already used by an alternate WCS. Add either "
                    "ArchiveMode.AUTO_RENAME flag to 'mode' in order to allow "
                    "'archive_wcs()' to modify 'wcsname' to be unique or "
                    "ArchiveMode.OVERWRITE_KEY flag in order to allow "
                    "'archive_wcs()' to overwrite an existing WCS."
                )
                if AM.QUIET_ABORT:
                    log.warning(msg)
                    return wcs_id
                else:
                    raise ValueError(msg)

        else:
            if wcskey in wcs_keys and not overwrite_key:
                msg = (f"Alternate WCS with WCS key '{key_match}' is "
                       "already present in header. To overwrite an "
                       "existing WCS add ArchiveMode.OVERWRITE_KEY flag "
                       "to 'mode'."
                )
                if AM.QUIET_ABORT:
                    log.warning(msg)
                    return wcs_id
                else:
                    raise ValueError(msg)

    else:
        auto_rename = False

    if wcskey is None:
        wcskey = _next_wcskey(hdr)

    # check if name is present in Alt WCS and if renaming is allowed:
    if auto_rename:
        wcsname = _auto_increment_wcsname(wcsname, wcs_names.values())

    log.setLevel('WARNING')

    wcs_id = (wcskey, wcsname)

    for e in ext:
        hwcs = wcs_from_key(h, e, from_key=' ', to_key=wcskey)

        if hwcs is None:
            continue

        # remove old WCS if present:
        if wcskey in wcs_keys:
            hwcs = wcs_from_key(h, e, from_key=wcskey)
            for k in hwcs:
                if k in h[e].header:
                    del h[e].header[k]

        hwcs.set(f'WCSNAME{wcskey:.1s}', wcsname, 'Coordinate system title', before=0)

        h[e].header.update(hwcs)

    log.setLevel(default_log_level)
    if close_hdulist:
        h.close()

    return wcs_id


def _auto_increment_wcsname(wcsname, alt_names):
    """
    Examples
    --------
    >>> altwcs._auto_increment_wcsname('IDC', ['IDC_A'])
    'IDC'

    >>> altwcs._auto_increment_wcsname('IDC', ['IDC_A', 'IDC'])
    'IDC-1'

    >>> altwcs._auto_increment_wcsname('IDC', ['IDC_A', 'IDC', 'IDC-1'])
    'IDC-2'

    >>> altwcs._auto_increment_wcsname('IDC-1', ['IDC_A', 'IDC', 'IDC-1'])
    'IDC-2'

    >>> altwcs._auto_increment_wcsname('IDC-2', ['IDC_A', 'IDC', 'IDC-1'])
    'IDC-2'

    >>> altwcs._auto_increment_wcsname('IDC_A', ['IDC_A', 'IDC', 'IDC-1'])
    'IDC_A-1'

    """
    separators = ['-', '_', ' ']  # in this order. First separator defines default
    wcsname_u = wcsname.upper()
    alt_names_u = [n.upper() for n in alt_names]
    if wcsname_u not in alt_names_u:
        return wcsname

    def find_counters(wname_u):
        num = {k: [] for k in separators}
        wname_u_len = len(wname_u)
        for name in alt_names_u:
            if name.startswith(wname_u) and len(name) > wname_u_len:
                for sep in separators:
                    if name[wname_u_len] == sep:
                        num_str = name[wname_u_len+1:]
                        if '_' not in num_str:
                            try:
                                num[sep].append(int(num_str))
                                break
                            except:
                                pass
        return num

    # figure out if wcsname ends in numbers:
    idx = max(wcsname.rfind(sep) for sep in separators)

    if idx >= 0:
        try:
            num_str = wcsname[idx + 1:].replace('_', '-')
            wcs_num = int(num_str)
            separator = wcsname[idx]
            wcsname_root = wcsname[:idx]
            wcsname_root_u = wcsname_root.upper()

            num = find_counters(wcsname_root_u)
            for sep in separators:
                if num[sep]:
                    max_num = max(num[sep]) + 1
                    return f'{wcsname_root}{separator}{max_num:d}'
            return f'{wcsname_root}{separator}{wcs_num + 1:d}'

        except Exception as e:
            pass

    num = find_counters(wcsname_u)
    for sep in separators:
        if num[sep]:
            max_num = max(num[sep]) + 1
            return f'{wcsname}{sep}{max_num:d}'

    return f'{wcsname:s}{separators[0]}1'


def _test_wcs_equal(h, ext, wcskey1, wcskey2):
    hwcs1 = wcs_from_key(h, ext, from_key=wcskey1, to_key=' ')
    if 'WCSNAME' in hwcs1:
        del hwcs1['WCSNAME']

    hwcs2 = wcs_from_key(h, ext, from_key=wcskey2, to_key=' ')
    if 'WCSNAME' in hwcs2:
        del hwcs2['WCSNAME']

    sdiff = set(c.image for c in hwcs1.cards).symmetric_difference(
        c.image for c in hwcs2.cards
    )

    return not bool(sdiff)


def restore_from_to(f, fromext=None, toext=None, wcskey=" ", wcsname=" "):
    """
    Copy an alternate WCS from one extension as a primary WCS of another extension

    Reads in a WCS defined with wcskey and saves it as the primary WCS.
    Goes sequentially through the list of extensions in ext.
    Alternatively uses 'fromext' and 'toext'.

    Parameters
    ----------
    f:       string or `astropy.io.fits.HDUList`
             a file name or a file object
    fromext: string
             extname from which to read in the alternate WCS, for example 'SCI'
    toext:   string or python list
             extname or a list of extnames to which the WCS will be copied as
             primary, for example ['SCI', 'ERR', 'DQ']
    wcskey:  a charater
             "A"-"Z" - Used for one of 26 alternate WCS definitions.
             or " " - find a key from WCSNAMe value
    wcsname: string (optional)
             if given and wcskey is " ", will try to restore by WCSNAME value

    See Also
    --------
    archive_wcs - Copy primary WCS as an alternate WCS
    restoreWCS - Copy a WCS with key "WCSKEY" to the primary WCS

    """
    if isinstance(f, str):
        fobj = fits.open(f, mode='update')
    else:
        fobj = f

    if not _parpasscheck(fobj, ext=None, wcskey=wcskey, fromext=fromext, toext=toext):
        closefobj(f, fobj)
        raise ValueError("Input parameters problem")

    # Interpret input 'ext' value to get list of extensions to process
    # ext = _buildExtlist(fobj, ext)

    if isinstance(toext, str):
        toext = [toext]

    # the case of an HDUList object in memory without an associated file

    # if fobj.filename() is not None:
    #        name = fobj.filename()

    wcskeyext = 0 if fu.isFits(fobj)[1] == 'simple' else 1

    if wcskey == " ":
        if wcsname.strip():
            wkey = getKeyFromName(fobj[wcskeyext].header, wcsname)
            if not wkey:
                closefobj(f, fobj)
                raise KeyError(f"Could not get a key from wcsname '{wcsname}'.")
    else:
        if wcskey not in wcskeys(fobj, ext=wcskeyext):
            print(f"Could not find alternate WCS with key '{wcskey}' in this file")
            closefobj(f, fobj)
            return
        wkey = wcskey

    countext = fu.countExtn(fobj, fromext)
    if countext:
        for i in range(1, countext + 1):
            for toe in toext:
                _restore(fobj, fromextnum=i, fromextnam=fromext, toextnum=i,
                         toextnam=toe, ukey=wkey)
    else:
        raise KeyError(f"File does not have extension with extname {fromext:s}")

    if fobj.filename() is not None:
        closefobj(f, fobj)


def restoreWCS(f, ext, wcskey=" ", wcsname=" "):
    """
    Copy a WCS with key "WCSKEY" to the primary WCS

    Reads in a WCS defined with wcskey and saves it as the primary WCS.
    Goes sequentially through the list of extensions in ext.
    Alternatively uses 'fromext' and 'toext'.


    Parameters
    ----------
    f : str or `astropy.io.fits.HDUList`
        file name or a file object
    ext : int, tuple, str, or list of integers or tuples (e.g.('sci',1))
        fits extensions to work with
        If a string is provided, it should specify the EXTNAME of extensions
        with WCSs to be archived
    wcskey : str
        "A"-"Z" - Used for one of 26 alternate WCS definitions.
        or " " - find a key from WCSNAMe value
    wcsname : str
        (optional) if given and wcskey is " ", will try to restore by WCSNAME value

    See Also
    --------
    archive_wcs - copy the primary WCS as an alternate WCS
    restore_from_to

    """
    if isinstance(f, str):
        fobj = fits.open(f, mode='update')
    else:
        fobj = f

    if not _parpasscheck(fobj, ext=ext, wcskey=wcskey):
        closefobj(f, fobj)
        raise ValueError("Input parameters problem")

    # Interpret input 'ext' value to get list of extensions to process
    ext = _buildExtlist(fobj, ext)

    # the case of an HDUList object in memory without an associated file

    wcskeyext = 0 if fu.isFits(fobj)[1] == 'simple' else 1

    if wcskey == " ":
        if wcsname.strip():
            wcskey = getKeyFromName(fobj[wcskeyext].header, wcsname)
            if not wcskey:
                closefobj(f, fobj)
                raise KeyError(f"Could not get a key from wcsname '{wcsname}'.")

    for e in ext:
        if wcskey in wcskeys(fobj, ext=e):
            _restore(fobj, wcskey, fromextnum=e, verbose=False)

    if fobj.filename() is not None:
        closefobj(f, fobj)


def deleteWCS(fname, ext, wcskey=" ", wcsname=" "):
    """
    Delete an alternate WCS defined with wcskey.
    If wcskey is " " try to get a key from WCSNAME.

    Parameters
    ----------
    fname : str or a `astropy.io.fits.HDUList`
    ext : int, tuple, str, or list of integers or tuples (e.g.('sci',1))
        fits extensions to work with
        If a string is provided, it should specify the EXTNAME of extensions
        with WCSs to be archived
    wcskey : str
        one of 'A'-'Z' or " "
    wcsname : str
        Name of alternate WCS description
    """
    if isinstance(fname, str):
        fobj = fits.open(fname, mode='update')
    else:
        fobj = fname

    if not _parpasscheck(fobj, ext, wcskey, wcsname):
        closefobj(fname, fobj)
        raise ValueError("Input parameters problem")

    # Interpret input 'ext' value to get list of extensions to process
    ext = _buildExtlist(fobj, ext)
    # Do not allow deleting the original WCS.
    if wcskey == 'O':
        print("Wcskey 'O' is reserved for the original WCS and should not be deleted.")
        closefobj(fname, fobj)
        return

    wcskeyext = ext[0]

    if not wcskeys and not wcsname:
        raise KeyError("Either wcskey or wcsname should be specified")

    if wcskey == " ":
        # try getting the key from WCSNAME
        wkey = getKeyFromName(fobj[wcskeyext].header, wcsname)
        if not wkey:
            closefobj(fname, fobj)
            raise KeyError(f"Could not get a key: wcsname '{wcsname}' not found in header.")
    else:
        if wcskey not in wcskeys(fobj[wcskeyext].header):
            closefobj(fname, fobj)
            raise KeyError(f"Could not find alternate WCS with key '{wcskey}' in this file")
        wkey = wcskey

    prexts = []
    for i in ext:
        hdr = fobj[i].header
        # set exclude_special=False to delete those keywords from the header
        # (if there were in the header before) when removing an Alt WCS
        hwcs = wcs_from_key(fobj, i, from_key=wkey, exclude_special=False)
        if hwcs:
            for k in hwcs:
                if k in hdr:
                    del hdr[k]
            prexts.append(i)

    if prexts:
        print(f'Deleted all instances of WCS with key {wkey:s} in extensions {prexts}')
    else:
        print(f"Did not find WCS with key {wkey:s} in any of the extensions {prexts}")
    closefobj(fname, fobj)


def _buildExtlist(fobj, ext, _single=False):
    """
    Utility function to interpret the provided value of 'ext' and return a list
    of 'valid' values which can then be used by the rest of the functions in
    this module.

    .. note::
       This function does not check that extension list contains extensions
       that identify the same HDUs. For example, for a typical ACS/WFC image
       it would not detect that 1 is the same as ``('SCI', 1)``.

    Parameters
    ----------
    fobj: HDUList
        file to be examined
    ext: an int, a tuple, string, list of integers or tuples (e.g.('sci',1))
        FITS extensions to work with. If a string is provided,
        it should specify the EXTNAME of extensions with WCSs to be archived.
    _single: bool
        Do not allow lists of simple extensions.

    """
    if not _single and isinstance(ext, list):
        ext_list = []
        for e in ext:
            ext_list.extend(_buildExtlist(fobj, e, _single=True))

    else:
        if isinstance(ext, str):
            ext_list = []
            for extn in range(1, len(fobj)):
                hdr = fobj[extn].header
                if 'extname' in hdr and hdr['extname'].upper() == ext.upper():
                    exti = (ext, hdr.get('extver', 1))
                    ext_list.append(extn if exti in ext_list else exti)

        elif isinstance(ext, Integral):
            ext_list = [int(ext)]

        elif (isinstance(ext, tuple) and len(ext) == 2 and
              isinstance(ext[0], str) and isinstance(ext[1], Integral)):
            ext_list = [(ext[0], int(ext[1]))]

        else:
            raise ValueError(
                "'ext' must be int, str, or a tuple of the form (extname, extver), "
                "where, extname is a string and extver is an integer number."
            )

    return ext_list


def _restore(fobj, ukey, fromextnum,
             toextnum=None, fromextnam=None, toextnam=None, verbose=True):
    """
    fobj: string of HDUList
    ukey: string 'A'-'Z'
          wcs key
    fromextnum: int
                extver of extension from which to copy WCS
    fromextnam: string
                extname of extension from which to copy WCS
    toextnum: int
              extver of extension to which to copy WCS
    toextnam: string
              extname of extension to which to copy WCS
    """
    # create an extension tuple, e.g. ('SCI', 2)
    fromextension = (fromextnam, fromextnum) if fromextnam else fromextnum

    if toextnum:
        if toextnam:
            toextension = (toextnam, toextnum)
        else:
            toextension = toextnum
    else:
        toextension = fromextension

    hwcs = wcs_from_key(fobj, fromextension, from_key=ukey, to_key=' ')
    fobj[toextension].header.update(hwcs)

    if ukey == 'O' and 'TDDALPHA' in fobj[toextension].header:
        fobj[toextension].header['TDDALPHA'] = 0.0
        fobj[toextension].header['TDDBETA'] = 0.0

    if 'ORIENTAT' in fobj[toextension].header:
        norient = np.rad2deg(np.arctan2(hwcs[f'CD1_2'], hwcs[f'CD2_2']))
        fobj[toextension].header['ORIENTAT'] = norient

    # Reset 2014 TDD keywords prior to computing new values (if any are computed)
    for kw in ['TDD_CYA', 'TDD_CYB', 'TDD_CXA', 'TDD_CXB']:
        if kw in fobj[toextension].header:
            fobj[toextension].header[kw] = 0.0


# header operations


def _check_headerpars(fobj, ext):
    if not isinstance(fobj, fits.Header) and not isinstance(fobj, fits.HDUList) \
            and not isinstance(fobj, str):
        raise ValueError("Expected a file name, a file object or a header\n")

    if not isinstance(fobj, fits.Header):
        if not isinstance(ext, Integral) and not isinstance(ext, tuple):
            raise ValueError("Expected ext to be a number or a tuple, e.g. ('SCI', 1)\n")


def _getheader(fobj, ext):
    if isinstance(fobj, str):
        hdr = fits.getheader(fobj, ext)
    elif isinstance(fobj, fits.Header):
        hdr = fobj
    else:
        hdr = fobj[ext].header
    return hdr


def wcs_from_key(fobj, ext, from_key=' ', to_key=None, exclude_special=True):
    """
    Read in WCS with a given ``from_key`` from the specified extension of
    the input file object and return a FITS header representing this WCS
    using desired WCS key specified by ``to_key``.

    Parameters
    ----------
    fobj : str, `astropy.io.fits.HDUList`
        FITS filename or `astropy.io.fits.HDUList` object containing
        a header with an alternate/primary WCS to be read.

    ext : int, str or tuple of (str, int)
        Extension specification identifying image HDU from which WCS should be
        loaded. If ``ext`` is a tuple, it is of the form ``(extname, extver)``
        where ``extname`` is a `str` extension name and ``extver`` is
        an integer extension version.
        An integer ``ext`` indicates "extension number". Finally, a single
        `str` extension name is interpretted as ``(ext, 1)``.

    from_key : {' ', 'A'-'Z'}, optional
        A 1 character string that is either a space character indicating the
        primary WCS, or one of the 26 ASCII letters (``'A'``-``'Z'``)
        indicating alternate WCS to be loaded from specified header.

    to_key : {None, ' ', 'A'-'Z'}, optional
        The key of the primary/alternate WCS to be used in the returned header.
        When ``to_key`` is `None`, the returned header describes a WCS with the
        same key as the one read in using ``from_key``. A space character or
        a single ASCII letter indicates the key to be used for the returned
        WCS (see ``from_key`` for details).

    exclude_special : bool, optional
        When `True`, HST-specific keywords that are intended to be present in
        the primary header and not in ``'SCI'`` extensions, will be filtered
        out from the returned WCS header.

    Returns
    -------
    hdr: astropy.io.fits.Header
        Header object with FITS representation for specified primary or
        alternate WCS.

    """
    if len(from_key) != 1 or from_key.strip() not in string.ascii_uppercase:
        raise ValueError(
            "Parameter 'from_key' must be a character - one of 'A'-'Z' or ' '."
        )

    if to_key is None:
        to_key = from_key

    elif len(to_key) != 1 or to_key.strip() not in string.ascii_uppercase:
        raise ValueError(
            "Parameter 'to_key' must be a character - one of 'A'-'Z' or ' '."
        )

    if isinstance(fobj, str):
        fobj = fits.open(fobj)
        close_fobj = True
    else:
        close_fobj = False

    hdr = _getheader(fobj, ext)

    try:
        w = pywcs.WCS(hdr, fobj=fobj, key=from_key)
    except KeyError as e:
        log.warning(f'wcs_from_key: Could not read WCS with key {from_key:s}')
        fname = fobj.filename()
        ftype = 'file'
        if fname is None:
            fname = f'{repr(fobj)}'
            ftype = 'in-memory file'
        log.warning(f"              Skipping {ftype} '{fname:s}[{ext}]'")
        log.warning(f"              {e.args[0]}")
        return fits.Header()
    finally:
        if close_fobj:
            fobj.close()

    hwcs = w.to_header(key=to_key)
    if exclude_special:
        exclude_hst_specific(hwcs, wcskey=to_key)

    if w.wcs.has_cd():
        hwcs = pc2cd(hwcs, key=to_key)

    # preserve CTYPE values:
    for k, ctype in enumerate(w.wcs.ctype):
        hwcs[f'CTYPE{k + 1:d}{to_key:.1s}'] = ctype

    # include non-standard (i.e., tweakreg-specific) keywords
    from_key_s = from_key.strip()
    to_key_s = to_key.strip()
    for kwd in STWCS_KWDS:
        from_kwd = kwd + from_key_s
        if from_kwd in hdr:
            # preserves comments and empty/null str values
            idx = hdr.index(from_kwd)
            hdri = hdr[idx:idx+1]
            hdri.rename_keyword(from_kwd, kwd + to_key_s, force=True)
            hwcs.update(hdri)

    return hwcs


def wcskeys(fobj, ext=None):
    """
    Returns a list of characters used in the header for alternate
    WCS description with WCSNAME keyword

    Parameters
    ----------
    fobj : str, `astropy.io.fits.HDUList` or `astropy.io.fits.Header`
          fits file name, fits file object or fits header
    ext : int or None
         extension number
         if None, fobj must be a header
    """
    _check_headerpars(fobj, ext)
    hdr = _getheader(fobj, ext)

    wcs_kwd_list = ['WCSNAME', 'CTYPE1', 'CRPIX1', 'CRVAL1', 'RADESYS']

    wkeys = set()

    # check primary:
    for kwd in wcs_kwd_list:
        if kwd in hdr:
            wkeys.add(' ')
            break

    # check Alt WCS:
    for kwd in wcs_kwd_list:
        alt_kwds = hdr[kwd + '?']
        alt_keys = [key[-1].upper() for key in alt_kwds if key[-1] in string.ascii_letters]
        wkeys.update(alt_keys)

    return sorted(wkeys)


def _alt_wcs_names(hdr, del_opus=True):
    """ Return a dictionary of all alternate WCS keys with names except ``OPUS``

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        An image header.

    del_opus : bool
        Indicates whether to remove ``OPUS`` entry (WCS key ``'O'``) from
        returned key-name dictionary.

    Returns
    -------
    wnames : dict
        A dictionary of **Alt** WCS keys and names (as values):

    """
    names = hdr["WCSNAME?"]

    if del_opus and 'WCSNAMEO' in names and names['WCSNAMEO'] == 'OPUS':
        # remove OPUS name
        del names['WCSNAMEO']

    wnames = {kwd[-1].upper(): val for kwd, val in names.items()
              if kwd[-1] in string.ascii_letters}

    return wnames


def wcsnames(fobj, ext=None, include_primary=True):
    """
    Returns a dictionary of wcskey: WCSNAME pairs

    Parameters
    ----------
    fobj : stri, `astropy.io.fits.HDUList` or `astropy.io.fits.Header`
          fits file name, fits file object or fits header
    ext : int or None
         extension number
         if None, fobj must be a header

    """
    _check_headerpars(fobj, ext)
    hdr = _getheader(fobj, ext)
    names = _alt_wcs_names(hdr, del_opus=False)
    if include_primary and 'WCSNAME' in hdr:
        names[' '] = hdr['WCSNAME']
    return names


def available_wcskeys(fobj, ext=None):
    """
    Returns a list of characters which are not used in the header
    with WCSNAME keyword. Any of them can be used to save a new
    WCS.

    Parameters
    ----------
    fobj : str, `astropy.io.fits.HDUList` or `astropy.io.fits.Header`
        fits file name, fits file object or fits header
    ext : int or None
        extension number
        if None, fobj must be a header
    """
    _check_headerpars(fobj, ext)
    hdr = _getheader(fobj, ext)
    return sorted(set(string.ascii_uppercase).difference(wcskeys(hdr)))


def next_wcskey(fobj, ext=None):
    """
    Returns next available character to be used for an alternate WCS

    Parameters
    ----------
    fobj : str, `astropy.io.fits.HDUList` or `astropy.io.fits.Header`
        fits file name, fits file object or fits header
    ext : int or None
        extension number
        if None, fobj must be a header
    """
    _check_headerpars(fobj, ext)
    hdr = _getheader(fobj, ext)
    return _next_wcskey(hdr)


def _next_wcskey(hdr):
    """
    Returns next available character to be used for an alternate WCS

    Parameters
    ----------
    hdr : `astropy.io.fits.Header`
        FITS header.

    Returns
    -------
    key : str, None
        Next available character to be used as an alternate WCS key or `None`
        if none is available.

    """
    all_keys = available_wcskeys(hdr)
    key = all_keys[0] if all_keys else None
    return key


def getKeyFromName(header, wcsname):
    """
    If WCSNAME is found in header, return its key, else return
    None. This is used to update an alternate WCS
    repeatedly and not generate new keys every time.

    Parameters
    ----------
    header :  `astropy.io.fits.Header`
    wcsname : str
        value of WCSNAME
    """
    names = wcsnames(header)
    wkeys = [key for key, name in names.items() if name.lower() == wcsname.lower()]
    if wkeys:
        wkey = max(wkeys)
    else:
        wkey = None
    return wkey


def pc2cd(hdr, key=' '):
    """
    Convert a PC matrix to a CD matrix.

    WCSLIB (and PyWCS) recognizes CD keywords as input
    but converts them and works internally with the PC matrix.
    to_header() returns the PC matrix even if the input was a
    CD matrix. To keep input and output consistent we check
    for has_cd and convert the PC back to CD.

    Parameters
    ----------
    hdr: `astropy.io.fits.Header`

    """
    key = key.strip()
    cdelt1 = hdr.pop(f'CDELT1{key:.1s}', 1)
    cdelt2 = hdr.pop(f'CDELT2{key:.1s}', 1)
    hdr[f'CD1_1{key:.1s}'] = (cdelt1 * hdr.pop(f'PC1_1{key:.1s}', 1),
                              'partial of first axis coordinate w.r.t. x')
    hdr[f'CD1_2{key:.1s}'] = (cdelt1 * hdr.pop(f'PC1_2{key:.1s}', 0),
                              'partial of first axis coordinate w.r.t. y')
    hdr[f'CD2_1{key:.1s}'] = (cdelt2 * hdr.pop(f'PC2_1{key:.1s}', 0),
                              'partial of second axis coordinate w.r.t. x')
    hdr[f'CD2_2{key:.1s}'] = (cdelt2 * hdr.pop(f'PC2_2{key:.1s}', 1),
                              'partial of second axis coordinate w.r.t. y')
    return hdr


def _parpasscheck(fobj, ext, wcskey, fromext=None, toext=None, reusekey=False):
    """
    Check input parameters to altwcs functions

    fobj : str or `astropy.io.fits.HDUList` object
        a file name or a file object
    ext : int, a tuple,a python list of integers or a python list
        of tuples (e.g.('sci',1))
        fits extensions to work with
    wcskey : str
        "A"-"Z" or " "- Used for one of 26 alternate WCS definitions
    wcsname : str
        (optional)
        if given and wcskey is " ", will try to restore by WCSNAME value
    reusekey : bool
        A flag which indicates whether to reuse a wcskey in the header
    """
    if not isinstance(fobj, fits.HDUList):
        print("First parameter must be a fits file object or a file name.")
        return False

    # first one covers the case of an object created in memory
    # (e.g. headerlet) for which fileinfo returns None
    if fobj.fileinfo(0) is None:
        pass
    else:
        # an HDUList object with associated file
        if fobj.fileinfo(0)['filemode'] != 'update':
            print("First parameter must be a file name or a file object opened in 'update' mode.")
            return False

    if not isinstance(ext, Integral) and not isinstance(ext, tuple) \
        and not isinstance(ext, str) \
        and not isinstance(ext, list) and ext is not None:
        print("Ext must be integer, tuple, string,a list of int extension "
              "numbers,\nor a list of tuples representing a fits extension, "
              "for example ('sci', 1).")
        return False

    if not isinstance(fromext, str) and fromext is not None:
        print("fromext must be a string representing a valid extname")
        return False

    if not isinstance(toext, list) and not isinstance(toext, str) and \
                        toext is not None:
        print("toext must be a string or a list of strings representing extname")
        return False

    if len(wcskey) > 1 or wcskey.strip() not in string.ascii_letters:
        print('Parameter wcskey must be a character - one of "A"-"Z" or " "')
        return False

    return True


def closefobj(fname, f):
    """
    Functions in this module accept as input a file name or a file object.
    If the input was a file name (string) we close the object. If the user
    passed a file object we leave it to the user to close it.
    """
    if isinstance(fname, str):
        f.close()


def mapFitsExt2HDUListInd(fname, extname):
    """
    Map FITS extensions with 'EXTNAME' to HDUList indexes.
    """

    if not isinstance(fname, fits.HDUList):
        f = fits.open(fname)
        close_file = True
    else:
        f = fname
        close_file = False

    d = {}
    for hdu in f:
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME'] == extname:
            extver = hdu.header['EXTVER']
            d[(extname, extver)] = f.index_of((extname, extver))

    if close_file:
        f.close()
    return d


def exclude_hst_specific(hdr, wcskey=' '):
    """ Remove HST-specific keywords from header ``hdr`` that are not supposed
    to be present in SCI headers.

    """
    if (wcskey is not None and
        ((not isinstance(wcskey, str) or len(wcskey) != 1 or
          wcskey.strip() not in string.ascii_uppercase))):
        raise ValueError(
            "Parameter 'wcskey' must be a character - one of 'A'-'Z' or ' '."
        )

    if 'EXTNAME' in hdr and hdr['EXTNAME'] not in ['SCI', 'DQ', 'ERR']:
        logger.warning("Input header must be either 'SCI', 'DQ', or 'ERR' image headers.")
        logger.warning("HST-specific keywords will not be excluded from the header.")
        return hdr

    if wcskey is None or wcskey == ' ':
        wcskey = ''

    for kwd in EXCLUDE_WCSLIB_KWDS:
        kwda = kwd + wcskey
        if kwda in hdr:
            del hdr[kwda]

    return hdr
