import copy
import numpy as np
from astropy.io import fits

from . import altwcs
from .hstwcs import HSTWCS
from ..updatewcs import utils
from stsci.tools import fileutil

DEFAULT_WCS_KEYS = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                    'CTYPE1', 'CTYPE2', 'ORIENTAT']
DEFAULT_PRI_KEYS = ['HDRNAME', 'SIPNAME', 'NPOLNAME', 'D2IMNAME', 'DESCRIP']
COL_FITSKW_DICT = {'RMS_RA': 'sci.crder1', 'RMS_DEC': 'sci.crder2',
                   'NMatch': 'sci.nmatch', 'Catalog': 'sci.catalog'}

###
### WCSEXT table related keyword archive functions
###
def init_wcscorr(input, force=False):
    """
    This function will initialize the WCSCORR table if it is not already present,
    and look for WCS keywords with a prefix of 'O' as the original OPUS
    generated WCS as the initial row for the table or use the current WCS
    keywords as initial row if no 'O' prefix keywords are found.

    This function will NOT overwrite any rows already present.

    This function works on all SCI extensions at one time.
    """
    # TODO: Create some sort of decorator or (for Python2.5) context for
    # opening a FITS file and closing it when done, if necessary
    if not isinstance(input, fits.HDUList):
        # input must be a filename, so open as `astropy.io.fits.HDUList` object
        fimg = fits.open(input, mode='update')
        need_to_close = True
    else:
        fimg = input
        need_to_close = False

    # Do not try to generate a WCSCORR table for a simple FITS file
    numsci = fileutil.countExtn(fimg)
    if len(fimg) == 1 or numsci == 0 or 'NDRIZIM' in fimg[0].header or 'D001DATA' in fimg[0].header:
        if need_to_close:
            fimg.close()
        return

    enames = []
    for e in fimg: enames.append(e.name)
    if 'WCSCORR' in enames:
        if not force:
            if need_to_close:
                fimg.close()
            return
        else:
            del fimg['wcscorr']
    print('Initializing new WCSCORR table for ', fimg.filename())

    used_wcskeys = altwcs.wcskeys(fimg['SCI', 1].header)

    # define the primary columns of the WCSEXT table with initial rows for each
    # SCI extension for the original OPUS solution
    numwcs = max(1, len(used_wcskeys))

    # create new table with more rows than needed initially to make it easier to
    # add new rows later
    wcsext = create_wcscorr(descrip=True, numrows=numsci, padding=(numsci * numwcs) + numsci * 4)
    # Assign the correct EXTNAME value to this table extension
    wcsext.header['TROWS'] = (numsci * 2, 'Number of updated rows in table')
    wcsext.header['EXTNAME'] = ('WCSCORR', 'Table with WCS Update history')
    wcsext.header['EXTVER'] = 1

    # define set of WCS keywords which need to be managed and copied to the table
    wcs1 = HSTWCS(fimg, ext=('SCI', 1))
    idc2header = wcs1.idcscale is not None
    wcs_keywords = list(wcs1.wcs2header(idc2hdr=idc2header).keys())

    prihdr = fimg[0].header
    prihdr_keys = DEFAULT_PRI_KEYS
    pri_funcs = {'SIPNAME': utils.build_sipname,
                 'NPOLNAME': utils.build_npolname,
                 'D2IMNAME': utils.build_d2imname}

    # Now copy original OPUS values into table
    for extver in range(1, numsci + 1):
        rowind = find_wcscorr_row(wcsext.data,
                                  {'WCS_ID': 'OPUS', 'EXTVER': extver,
                                   'WCS_key': 'O'})
        # There should only EVER be a single row for each extension with OPUS values
        rownum = np.where(rowind)[0][0]
        # print 'Archiving OPUS WCS in row number ',rownum,' in WCSCORR table for SCI,',extver

        hdr = fimg['SCI', extver].header
        # define set of WCS keywords which need to be managed and copied to the table
        if used_wcskeys is None:
            used_wcskeys = altwcs.wcskeys(hdr)
        # Check to see whether or not there is an OPUS alternate WCS present,
        # if so, get its values directly, otherwise, archive the PRIMARY WCS
        # as the OPUS values in the WCSCORR table
        if 'O' not in used_wcskeys:
            altwcs.archive_wcs(fimg, ('SCI', extver), wcskey='O', wcsname='OPUS')
        wkey = 'O'

        wcs = HSTWCS(fimg, ext=('SCI', extver), wcskey=wkey)
        wcshdr = wcs.wcs2header(idc2hdr=idc2header)

        if wcsext.data.field('CRVAL1')[rownum] != 0:
            # If we find values for these keywords already in the table, do not
            # overwrite them again
            print('WCS keywords already updated...')
            break

        for kwd in wcs_keywords:
            alt_kwd = (kwd + wkey)[:8]
            if kwd in wcsext.data.names and alt_kwd in wcshdr:
                wcsext.data.field(kwd)[rownum] = wcshdr[alt_kwd]

        # Now get any keywords from PRIMARY header needed for WCS updates
        for kwd in prihdr_keys:
            wcsext.data.field(kwd)[rownum] = prihdr.get(kwd, '')

    # Now that we have archived the OPUS alternate WCS, remove it from the list
    # of used_wcskeys
    if 'O' in used_wcskeys:
        used_wcskeys.remove('O')

    # Now copy remaining alternate WCSs into table
    # TODO: Much of this appears to be redundant with update_wcscorr; consider
    # merging them...
    for uwkey in used_wcskeys:
        for extver in range(1, numsci + 1):
            hdr = fimg['SCI', extver].header
            wcs = HSTWCS(fimg, ext=('SCI', extver),
                                       wcskey=uwkey)
            wcshdr = wcs.wcs2header()
            if 'WCSNAME' + uwkey not in wcshdr:
                wcsid = utils.build_default_wcsname(fimg[0].header['idctab'])
            else:
                wcsid = wcshdr['WCSNAME' + uwkey]

            # identify next empty row
            rowind = find_wcscorr_row(wcsext.data,
                                      selections={'wcs_id': ['', '0.0']})
            rows = np.where(rowind)
            if len(rows[0]) > 0:
                rownum = np.where(rowind)[0][0]
            else:
                print('No available rows found for updating. ')

            # Update selection columns for this row with relevant values
            wcsext.data.field('WCS_ID')[rownum] = wcsid
            wcsext.data.field('EXTVER')[rownum] = extver
            wcsext.data.field('WCS_key')[rownum] = uwkey

            # Look for standard WCS keyword values
            for key in wcs_keywords:
                if key in wcsext.data.names:
                    wcsext.data.field(key)[rownum] = wcshdr[key + uwkey]
            # Now get any keywords from PRIMARY header needed for WCS updates
            for key in prihdr_keys:
                if key in pri_funcs:
                    val = pri_funcs[key](fimg)[0]
                else:
                    val = prihdr.get(key, '')
                wcsext.data.field(key)[rownum] = val

    # Append this table to the image FITS file
    fimg.append(wcsext)
    # force an update now
    # set the verify flag to 'warn' so that it will always succeed, but still
    # tell the user if PyFITS detects any problems with the file as a whole
    utils.updateNEXTENDKw(fimg)

    fimg.flush('warn')

    if need_to_close:
        fimg.close()


def find_wcscorr_row(wcstab, selections):
    """
    Return an array of indices from the table (NOT HDU) 'wcstab' that matches the
    selections specified by the user.

    The row selection criteria must be specified as a dictionary with
    column name as key and value(s) representing the valid desired row values.
    For example, {'wcs_id':'OPUS','extver':2}.
    """

    mask = None
    for i in selections:
        icol = wcstab.field(i)
        #if isinstance(icol, np.chararray): icol = icol.rstrip()
        selecti = selections[i]
        if not isinstance(selecti, list):
            if isinstance(selecti, str):
                selecti = selecti.rstrip()
            bmask = (icol == selecti)
            if mask is None:
                mask = bmask.copy()
            else:
                mask = np.logical_and(mask, bmask)
            del bmask
        else:
            for si in selecti:
                if isinstance(si, str):
                    si = si.rstrip()
                bmask = (icol == si)
                if mask is None:
                    mask = bmask.copy()
                else:
                    mask = np.logical_or(mask, bmask)
                del bmask

    return mask


def archive_wcs_file(image, wcs_id=None):
    """
    Update WCSCORR table with rows for each SCI extension to record the
    newly updated WCS keyword values.
    """

    if not isinstance(image, fits.HDUList):
        fimg = fits.open(image, mode='update')
        close_image = True
    else:
        fimg = image
        close_image = False

    update_wcscorr(fimg, wcs_id=wcs_id)

    if close_image:
        fimg.close()


def update_wcscorr(dest, source=None, extname='SCI', wcs_id=None, active=True):
    """
    Update WCSCORR table with a new row or rows for this extension header. It
    copies the current set of WCS keywords as a new row of the table based on
    keyed WCSs as per Paper I Multiple WCS standard).

    Parameters
    ----------
    dest : HDUList
        The HDU list whose WCSCORR table should be appended to (the WCSCORR HDU
        must already exist)
    source : HDUList, optional
        The HDU list containing the extension from which to extract the WCS
        keywords to add to the WCSCORR table.  If None, the dest is also used
        as the source.
    extname : str, optional
        The extension name from which to take new WCS keywords.  If there are
        multiple extensions with that name, rows are added for each extension
        version.
    wcs_id : str, optional
        The name of the WCS to add, as in the WCSNAMEa keyword.  If
        unspecified, all the WCSs in the specified extensions are added.
    active: bool, optional
        When True, indicates that the update should reflect an update of the
        active WCS information, not just appending the WCS to the file as a
        headerlet
    """
    if not isinstance(dest, fits.HDUList):
        dest = fits.open(dest, mode='update')
    fname = dest.filename()

    if source is None:
        source = dest

    if extname == 'PRIMARY':
        return

    numext = fileutil.countExtn(source, extname)
    if numext == 0:
        raise ValueError('No %s extensions found in the source HDU list.'
                         % extname)
    # Initialize the WCSCORR table extension in dest if not already present
    init_wcscorr(dest)
    try:
        dest.index_of('WCSCORR')
    except KeyError:
        return

    # check to see whether or not this is an up-to-date table
    # replace with newly initialized table with current format
    old_table = dest['WCSCORR']
    wcscorr_cols = ['WCS_ID', 'EXTVER', 'SIPNAME',
                    'HDRNAME', 'NPOLNAME', 'D2IMNAME']

    for colname in wcscorr_cols:
        if colname not in old_table.data.columns.names:
            print("WARNING:    Replacing outdated WCSCORR table...")
            #outdated_table = old_table.copy()
            del dest['WCSCORR']
            init_wcscorr(dest)
            old_table = dest['WCSCORR']
            break

    # Current implementation assumes the same WCS keywords are in each
    # extension version; if this should not be assumed then this can be
    # modified...
    wcs_keys = altwcs.wcskeys(source[(extname, 1)].header)
    if 'O' in wcs_keys:
        wcs_keys.remove('O')  # 'O' is reserved for original OPUS WCS
    if ' ' not in wcs_keys: wcs_keys.append(' ')  # Insure that primary WCS gets used
    # apply logic for only updating WCSCORR table with specified keywords
    # corresponding to the WCS with WCSNAME=wcs_id
    if wcs_id is not None:
        wcs_id_up = wcs_id.upper()
        wnames = altwcs._alt_wcs_names(source[(extname, 1)].header)
        wkeys = [key for key, name in wnames.items() if name.upper() == wcs_id_up]
        if len(wkeys) > 1 and ' ' in wkeys:
            wkeys.remove(' ')
        wcs_keys = wkeys
    wcshdr = HSTWCS(source, ext=(extname, 1)).wcs2header()
    wcs_keywords = list(wcshdr.keys())

    # create new table for hdr and populate it with the newly updated values
    new_table = create_wcscorr(descrip=True, numrows=0, padding=len(wcs_keys) * numext)
    prihdr = source[0].header

    # Get headerlet related keywords here
    sipname, idctab = utils.build_sipname(source, fname, "None")
    npolname, npolfile = utils.build_npolname(source, None)
    d2imname, d2imfile = utils.build_d2imname(source, None)
    hdrname = prihdr.get('hdrname', '')

    idx = -1
    for wcs_key in wcs_keys:
        wcs_key_hdr = wcs_key.strip()

        for extver in range(1, numext + 1):
            extn = (extname, extver)
            if 'SIPWCS' in extname and not active:
                tab_extver = 0  # Since it has not been added to the SCI header yet
            else:
                tab_extver = extver
            hdr = source[extn].header
            wcsname_kwd = 'WCSNAME' + wcs_key_hdr
            if wcsname_kwd in hdr:
                wcsname = hdr[wcsname_kwd]
            else:
                wcsname = utils.build_default_wcsname(hdr['idctab'])

            selection = {'WCS_ID': wcsname, 'EXTVER': tab_extver,
                         'SIPNAME': sipname, 'HDRNAME': hdrname,
                         'NPOLNAME': npolname, 'D2IMNAME': d2imname
                         }

            # Ensure that an entry for this WCS is not already in the dest
            # table; if so just skip it
            rowind = find_wcscorr_row(old_table.data, selection)
            if np.any(rowind):
                continue

            idx += 1

            wcs = HSTWCS(source, ext=extn, wcskey=wcs_key)
            wcshdr = wcs.wcs2header()

            # Update selection column values
            for key, val in selection.items():
                if key in new_table.data.names:
                    new_table.data.field(key)[idx] = val

            for key in wcs_keywords:
                if key in new_table.data.names:
                    new_table.data.field(key)[idx] = wcshdr[key + wcs_key_hdr]

            for key in DEFAULT_PRI_KEYS:
                if key in new_table.data.names and key in prihdr:
                    new_table.data.field(key)[idx] = prihdr[key]

            # Now look for additional, non-WCS-keyword table column data
            for key, fitkw in COL_FITSKW_DICT.items():
                # Interpret any 'pri.hdrname' or
                # 'sci.crpix1' formatted keyword names
                if '.' in fitkw:
                    srchdr, fitkw = fitkw.split('.')
                    if 'pri' in srchdr.lower():
                        srchdr = prihdr
                    else:
                        srchdr = source[extn].header
                else:
                    srchdr = source[extn].header

                if fitkw + wcs_key_hdr in srchdr:
                    new_table.data.field(key)[idx] = srchdr[fitkw + wcs_key_hdr]

    # If idx was never incremented, no rows were added, so there's nothing else
    # to do...
    if idx < 0:
        return

    # Now, we need to merge this into the existing table
    rowind = find_wcscorr_row(old_table.data, {'wcs_id': ['', '0.0']})
    old_nrows = np.where(rowind)[0][0]
    new_nrows = new_table.data.shape[0]

    # check to see if there is room for the new row
    if (old_nrows + new_nrows) > old_table.data.shape[0] - 1:
        pad_rows = 2 * new_nrows
        # if not, create a new table with 'pad_rows' new empty rows
        upd_table = fits.BinTableHDU.from_columns(old_table.columns, header=old_table.header,
                                                  nrows=old_table.data.shape[0] + pad_rows)
    else:
        upd_table = old_table
        pad_rows = 0
    # Now, add
    for name in old_table.columns.names:
        if name in new_table.data.names:
            # reset the default values to ones specific to the row definitions
            for i in range(pad_rows):
                upd_table.data.field(name)[old_nrows + i] = old_table.data.field(name)[-1]
            # Now populate with values from new table
            upd_table.data.field(name)[old_nrows:old_nrows + new_nrows] = \
                new_table.data.field(name)
    upd_table.header['TROWS'] = old_nrows + new_nrows

    # replace old extension with newly updated table extension
    dest['WCSCORR'] = upd_table


def restore_file_from_wcscorr(image, id='OPUS', wcskey=''):
    """ Copies the values of the WCS from the WCSCORR based on ID specified by user.
    The default will be to restore the original OPUS-derived values to the Primary WCS.
    If wcskey is specified, the WCS with that key will be updated instead.
    """
    wcskey = wcskey.strip()
    if not isinstance(image, fits.HDUList):
        fimg = fits.open(image, mode='update')
        close_image = True
    else:
        fimg = image
        close_image = False
    numsci = fileutil.countExtn(fimg)
    wcs_table = fimg['WCSCORR']
    orig_rows = (wcs_table.data.field('WCS_ID') == 'OPUS')
    # create an HSTWCS object to figure out what WCS keywords need to be updated
    wcsobj = HSTWCS(fimg, ext=('sci', 1))
    wcshdr = wcsobj.wcs2header()
    for extn in range(1, numsci + 1):
        # find corresponding row from table
        ext_rows = (wcs_table.data.field('EXTVER') == extn)
        erow = np.where(np.logical_and(ext_rows, orig_rows))[0][0]
        for key in wcshdr:
            if key in wcs_table.data.names:  # insure that keyword is column in table
                tkey = key

                if 'orient' in key.lower():
                    key = 'ORIENT'
                if wcskey == '':
                    skey = key
                else:
                    skey = key[:7] + wcskey
                fimg['sci', extn].header[skey] = wcs_table.data.field(tkey)[erow]
        for key in DEFAULT_PRI_KEYS:
            if key in wcs_table.data.names:
                if wcskey == '':
                    pkey = key
                else:
                    pkey = key[:7] + wcskey
                fimg[0].header[pkey] = wcs_table.data.field(key)[erow]

    utils.updateNEXTENDKw(fimg)

    # close the image now that the update has been completed.
    if close_image:
        fimg.close()


def create_wcscorr(descrip=False, numrows=1, padding=0):
    """
    Return the basic definitions for a WCSCORR table.
    The dtype definitions for the string columns are set to the maximum allowed so
    that all new elements will have the same max size which will be automatically
    truncated to this limit upon updating (if needed).

    The table is initialized with rows corresponding to the OPUS solution
    for all the 'SCI' extensions.
    """

    trows = numrows + padding
    # define initialized arrays as placeholders for column data
    # TODO: I'm certain there's an easier way to do this... for example, simply
    # define the column names and formats, then create an empty array using
    # them as a dtype, then create the new table from that array.
    def_float64_zeros = np.array([0.0] * trows, dtype=np.float64)
    def_float64_ones = def_float64_zeros + 1.0
    def_float_col = {'format': 'D', 'array': def_float64_zeros.copy()}
    def_float1_col = {'format': 'D', 'array': def_float64_ones.copy()}
    def_str40_col = {'format': '40A',
                     'array': np.array([''] * trows, dtype='S40')}
    def_str24_col = {'format': '24A',
                     'array': np.array([''] * trows, dtype='S24')}
    def_int32_col = {'format': 'J',
                     'array': np.array([0] * trows, dtype=np.int32)}

    # If more columns are needed, simply add their definitions to this list
    col_names = [('HDRNAME', def_str24_col), ('SIPNAME', def_str24_col),
                 ('NPOLNAME', def_str24_col), ('D2IMNAME', def_str24_col),
                 ('CRVAL1', def_float_col), ('CRVAL2', def_float_col),
                 ('CRPIX1', def_float_col), ('CRPIX2', def_float_col),
                 ('CD1_1', def_float1_col), ('CD1_2', def_float_col),
                 ('CD2_1', def_float_col), ('CD2_2', def_float1_col),
                 ('CTYPE1', def_str24_col), ('CTYPE2', def_str24_col),
                 ('ORIENTAT', def_float_col), ('PA_V3', def_float_col),
                 ('RMS_RA', def_float_col), ('RMS_Dec', def_float_col),
                 ('NMatch', def_int32_col), ('Catalog', def_str40_col)]

    # Define selector columns
    id_col = fits.Column(name='WCS_ID', format='40A',
                         array=np.array(['OPUS'] * numrows + [''] * padding,
                                        dtype='S24'))
    extver_col = fits.Column(name='EXTVER', format='I',
                             array=np.array(list(range(1, numrows + 1)),
                                            dtype=np.int16))
    wcskey_col = fits.Column(name='WCS_key', format='A',
                             array=np.array(['O'] * numrows + [''] * padding,
                                            dtype='S'))
    # create list of remaining columns to be added to table
    col_list = [id_col, extver_col, wcskey_col]  # start with selector columns

    for c in col_names:
        cdef = copy.deepcopy(c[1])
        col_list.append(fits.Column(name=c[0], format=cdef['format'],
                        array=cdef['array']))

    if descrip:
        col_list.append(
            fits.Column(name='DESCRIP', format='128A',
                        array=np.array(
                            ['Original WCS computed by OPUS'] * numrows,
                            dtype='S128')))

    # Now create the new table from the column definitions
    newtab = fits.BinTableHDU.from_columns(fits.ColDefs(col_list), nrows=trows)
    # The fact that setting .name is necessary should be considered a bug in
    # pyfits.
    # TODO: Make sure this is fixed in pyfits, then remove this
    newtab.name = 'WCSCORR'

    return newtab


def delete_wcscorr_row(wcstab, selections=None, rows=None):
    """
    Sets all values in a specified row or set of rows to default values

    This function will essentially erase the specified row from the table
    without actually removing the row from the table. This avoids the problems
    with trying to resize the number of rows in the table while preserving the
    ability to update the table with new rows again without resizing the table.

    Parameters
    ----------
    wcstab: object
        PyFITS binTable object for WCSCORR table
    selections: dict
        Dictionary of wcscorr column names and values to be used to select
        the row or set of rows to erase
    rows: int, list
        If specified, will specify what rows from the table to erase regardless
        of the value of 'selections'
    """

    if selections is None and rows is None:
        print('ERROR: Some row selection information must be provided!')
        print('       Either a row numbers or "selections" must be provided.')
        raise ValueError

    delete_rows = None
    if rows is None:
        if 'wcs_id' in selections and selections['wcs_id'] == 'OPUS':
            delete_rows = None
            print('WARNING: OPUS WCS information can not be deleted from WCSCORR table.')
            print('         This row will not be deleted!')
        else:
            rowind = find_wcscorr_row(wcstab, selections=selections)
            delete_rows = np.where(rowind)[0].tolist()
    else:
        if not isinstance(rows, list):
            rows = [rows]
        delete_rows = rows

    # Insure that rows pointing to OPUS WCS do not get deleted, even by accident
    for row in delete_rows:
        if wcstab['WCS_key'][row] == 'O' or wcstab['WCS_ID'][row] == 'OPUS':
            del delete_rows[delete_rows.index(row)]

    if delete_rows is None:
        return

    # identify next empty row
    rowind = find_wcscorr_row(wcstab, selections={'wcs_id': ['', '0.0']})
    last_blank_row = np.where(rowind)[0][-1]

    # copy values from blank row into user-specified rows
    for colname in wcstab.names:
        wcstab[colname][delete_rows] = wcstab[colname][last_blank_row]


def update_wcscorr_column(wcstab, column, values, selections=None, rows=None):
    """
    Update the values in 'column' with 'values' for selected rows

    Parameters
    ----------
    wcstab: object
        PyFITS binTable object for WCSCORR table
    column: string
        Name of table column with values that need to be updated
    values: string, int, or list
        Value or set of values to copy into the selected rows for the column
    selections: dict
        Dictionary of wcscorr column names and values to be used to select
        the row or set of rows to erase
    rows: int, list
        If specified, will specify what rows from the table to erase regardless
        of the value of 'selections'
    """
    if selections is None and rows is None:
        print('ERROR: Some row selection information must be provided!')
        print('       Either a row numbers or "selections" must be provided.')
        raise ValueError

    if not isinstance(values, list):
        values = [values]

    update_rows = None
    if rows is None:
        if 'wcs_id' in selections and selections['wcs_id'] == 'OPUS':
            update_rows = None
            print('WARNING: OPUS WCS information can not be deleted from WCSCORR table.')
            print('         This row will not be deleted!')
        else:
            rowind = find_wcscorr_row(wcstab, selections=selections)
            update_rows = np.where(rowind)[0].tolist()
    else:
        if not isinstance(rows, list):
            rows = [rows]
        update_rows = rows

    if update_rows is None:
        return

    # Expand single input value to apply to all selected rows
    if len(values) > 1 and len(values) < len(update_rows):
        print('ERROR: Number of new values', len(values))
        print('       does not match number of rows', len(update_rows), ' to be updated!')
        print('       Please enter either 1 value or the same number of values')
        print('       as there are rows to be updated.')
        print('    Table will not be updated...')
        raise ValueError

    if len(values) == 1 and len(values) < len(update_rows):
        values = values * len(update_rows)
    # copy values from blank row into user-specified rows
    for row in update_rows:
        wcstab[column][row] = values[row]
