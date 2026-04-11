"""Interface for Astrometry database service.

This module contains interface functions for the AstrometryDB Restful service
based on interfaces/code provided by B. McLean 11-Oct-2017.

The code checks for the existence of environmental variables to help control
the operation of this interface; namely,

RAISE_PIPELINE_ERRORS - boolean to specify whether to raise exceptions
                        during processing or simply log errors and quit
                        gracefully.  If not set, default behavior will be
                        to log errors and quit gracefully.

ASTROMETRY_SERVICE_URL - URL pointing to user-specified web service that
                        will provide updated astrometry solutions for
                        observations being processed.  This will replace
                        the built-in URL included in the base class.  This
                        value will also be replaced by any URL provided by
                        the user as an input parameter `url`.

ASTROMETRY_STEP_CONTROL - String specifying whether or not to perform the
                          astrometry update processing at all.
                          Valid Values: "ON", "On", "on", "OFF", "Off", "off"
                          If not set, default value is "ON".

GSSS_WEBSERVICES_URL - URL point to user-specified web service which provides
                       information on the guide stars used for taking HST
                       observations. This value will replace the default URL
                       included in this module as the `gsss_url` variable.

"""
import os
import sys
import time
import atexit
import hashlib

import requests
from requests.exceptions import ConnectionError
from io import BytesIO
from lxml import etree

from astropy.io import fits
from astropy.coordinates import SkyCoord

from stsci.tools import fileutil
from ..wcsutil import headerlet
from ..wcsutil import HSTWCS
from ..wcsutil import altwcs
from ..distortion import utils
from . import updatehdr

import logging

logger = logging.getLogger('stwcs.updatewcs.astrometry_utils')
for h in logger.handlers:
    if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout:
        break
else:
    logger.handlers.append(logging.StreamHandler(sys.stdout))
atexit.register(logging.shutdown)

# Definitions of environment variables used by this step
astrometry_db_envvar = "ASTROMETRY_SERVICE_URL"
gsss_url_envvar = "GSSS_WEBSERVICES_URL"
pipeline_error_envvar = "RAISE_PIPELINE_ERRORS"
astrometry_control_envvar = "ASTROMETRY_STEP_CONTROL"

gsss_url = 'https://gsss.stsci.edu/webservices'

class AstrometryDB:
    """Base class for astrometry database interface."""

    serviceLocation = 'https://mast.stsci.edu/portal/astrometryDB/'
    headers = {'Content-Type': 'text/xml'}

    available = True
    available_code = {'code': "", 'text': ""}

    def __init__(self, url=None, raise_errors=None, perform_step=True,
                 write_log=False, testing=False):
        """Initialize class with user-provided URL.

        Parameters
        ==========
        url : str
            User-provided URL for astrometry dB web-service to replaced
            default web-service.  Any URL specified here will override
            any URL specified as the environment variable
            `ASTROMETRY_SERVICE_URL`.  This parameter value, if specified,
            and the environmental variable will replace the built-in default
            URL included in this class.

        raise_errors : bool, optional
             User can specify whether or not to turn off raising exceptions
             upon errors in getting or applying new astrometry solutions.
             This will override the environment variable
             'RAISE_PIPELINE_ERRORS' if set.

        perform_step : bool, optional
            Specify whether or not to perform this step.  This is
            overriden by the setting of the `ASTROMETRY_STEP_CONTROL`
            environment variable.  Default value: True.

        write_log : bool, optional
            Specify whether or not to write a log file during processing.
            Default: False

        testing : bool, optional
            Shortens time between retries when checking for service availability.
            Default: False

        """
        self.perform_step = perform_step
        self.testing = testing
        # Check to see whether an environment variable has been set
        if astrometry_control_envvar in os.environ:
            val = os.environ[astrometry_control_envvar].lower()
            if val == 'off':
                self.perform_step = False
            elif val == 'on':
                self.perform_step = True
            else:
                l = "Environment variable {} not set to valid value".\
                    format(astrometry_control_envvar)
                l += "\t Valid values: on or off (case-insensitive)"
                raise ValueError(l)
            logger.info("Astrometry step operation set to {}".
                        format(self.perform_step))
        if not self.perform_step:
            logger.info("Astrometry update step has been turned off")
            logger.info("\tNo updates will be performed!")
            return

        if write_log:
            formatter = logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            log_filename = 'astrometry.log'
            fh = logging.FileHandler(log_filename, mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        # check to see whether any URL has been specified as an
        # environmental variable.
        if astrometry_db_envvar in os.environ:
            self.serviceLocation = os.environ[astrometry_db_envvar]

        if url is not None:
            self.serviceLocation = url
        #
        # Implement control over behavior for error conditions
        # User provided input will always take precedent
        # Environment variable will also be recognized if no user-variable set
        # otherwise, it will turn off raising Exceptions
        #
        self.raise_errors = False
        if pipeline_error_envvar in os.environ:
            val = os.environ[pipeline_error_envvar].lower()
            if val in ["true"]:
                self.raise_errors = True
            elif val in ["false"]:
                self.raise_errors = False
            else:
                l = f"Invalid environment variable setting for {pipeline_error_envvar}."
                l += "\t Valid values: True or False (case-insensitive)"
                raise ValueError(l)
            logger.debug(f"{pipeline_error_envvar} set to {self.raise_errors}")
        if raise_errors is not None:
            self.raise_errors = raise_errors
            logger.info("Setting `raise_errors` to {}".format(raise_errors))

        self.isAvailable()  # determine whether service is available

        # Initialize attribute to keep track of type of observation
        self.new_observation = False
        self.deltas = None

    def updateObs(self, obsname, all_wcs=False, remove_duplicates=True):
        """Update observation with any available solutions.

        Parameters
        ==========
        obsname : str
           Filename for observation to be updated

        all_wcs : bool
            If True, all solutions from the Astrometry database
            are appended to the input file as separate FITS
            extensions.  If False, only those solutions based on the
            same IDCTAB will be appended.

        remove_duplicates : bool
            If True, any headerlet extensions with the same
            HDRNAME are found, the copies will
            be deleted until only the first version added remains.
        """
        if not self.perform_step:
            return

        obs_open = False
        # User provided only an input filename, so open in 'update' mode
        if isinstance(obsname, str):
            obsfile = obsname
            obsname = fits.open(obsfile, mode='update')
            obs_open = True
        elif isinstance(obsname, fits.HDUList):
            obsfile = obsname.filename()
            # User provided an HDUList - make sure it is opened in 'update' mode
            if obsname.fileinfo(0)['filemode'] != 'update':
                # Not opened in 'update' mode, so close and re-open
                obsname.close()
                logger.info("Opening {} in 'update' mode to append new WCSs".format(obsfile))
                obsname = fits.open(obsfile, mode='update')
        else:
            # We do not know what kind of input this is, so raise an Exception with an explanation.
            error_msg = "Input not valid!  Please provide either a filename or fits.HDUList object"
            logger.error(error_msg)
            raise ValueError(error_msg)

        obsroot = obsname[0].header.get('rootname', None)
        observationID = obsroot.split('_')[:1][0]
        logger.info("Updating astrometry for {}".format(observationID))

        # take inventory of what hdrlets are already appended to this file
        wcsnames = headerlet.get_headerlet_kw_names(obsname, 'wcsname')

        # Get all the WCS solutions available from the astrometry database
        # for this observation, along with what was flagged as the 'best'
        # solution.  The 'best' solution should be the one that aligns the
        # observation closest to the GAIA frame.
        headerlets, best_solution_id = self.getObservation(observationID)
        if headerlets is None:
            logger.warning("Problems getting solutions from database")
            logger.warning(" NO Updates performed for {}".format(
                           observationID))
            if self.raise_errors:
                raise ValueError("No new solution found in AstrometryDB.")
            else:
                return

        # Get IDCTAB filename from file header
        idctab = obsname[0].header.get('IDCTAB', None)
        idcroot = os.path.basename(fileutil.osfn(idctab)).split('_')[0]

        # Determine what WCSs to append to this observation
        # If headerlet found in database, update file with all new WCS solutions
        # according to the 'all_wcs' parameter
        apriori_added = False
        if not self.new_observation:
            # Attach new unique hdrlets to file...
            logger.info("Updating {} with:".format(observationID))
            for h in headerlets:
                newname = headerlets[h][0].header['wcsname']
                # Only append the WCS from the database if `all_wcs` was turned on,
                # or the WCS was based on the same IDCTAB as in the image header.
                append_wcs = True if ((idcroot in newname) or all_wcs or newname == 'OPUS') else False
                if append_wcs and (idcroot in newname):
                    apriori_added = True

                # Check to see whether this WCS has already been appended or
                # if it was never intended to be appended.  If so, skip it.
                if newname in wcsnames:
                    continue  # do not add duplicate hdrlets
                # Add solution as an alternate WCS
                if append_wcs:
                    try:
                        logger.info("\tHeaderlet with WCSNAME={}".format(
                                    newname))
                        headerlets[h].attach_to_file(obsname)
                    except ValueError:
                        pass

        if remove_duplicates:
            hdr_kw = headerlet.get_headerlet_kw_names(obsname, kw='HDRNAME')
            for hname in [kwd for kwd in set(hdr_kw) if hdr_kw.count(kwd) > 1]:
                headerlet.delete_headerlet([obsname], hdrname=hname, keep_first=True)
                logger.warn(f"Duplicate headerlet with 'HDRNAME'='{hname}' found.")
                logger.warn("Duplicate headerlets have been removed.")

        # Obtain the current primary WCS name
        current_wcsname = obsname[('sci', 1)].header['wcsname']

        # At this point, we have appended all applicable headerlets from the database
        # However, if no database-provided headerlet was applicable, we need to
        # compute a new a priori WCS based on the IDCTAB from the observation header.
        # This will also re-define the 'best_solution_id'.
        if not apriori_added:
            # No headerlets were appended from the database, so we need to define
            # a new a priori solution and apply it as the new 'best_solution_id'
            self.apply_new_apriori(obsname)

        else:
            # Once all the new headerlet solutions have been added as new extensions
            # Apply the best solution, if one was specified, as primary WCS
            # This needs to be separate logic in order to work with images which have already
            # been updated with solutions from the database, and we are simply resetting.
            if best_solution_id and best_solution_id != current_wcsname:
                # get full list of all headerlet extensions now in the file
                hdrlet_extns = headerlet.get_extname_extver_list(obsname, 'hdrlet')

                for h in hdrlet_extns:
                    hdrlet = obsname[h].headerlet
                    wcsname = hdrlet[0].header['wcsname']
                    if wcsname == best_solution_id:
                        # replace primary WCS with this solution
                        hdrlet.init_attrs()
                        hdrlet.apply_as_primary(obsname, attach=False, force=True)
                        logger.info('Replacing primary WCS with')
                        logger.info('\tHeaderlet with WCSNAME={}'.format(
                                     newname))
                        break

        # Insure changes are written to the file and that the file is closed.
        if obs_open:
            obsname.close()

    def findObservation(self, observationID):
        """Find whether there are any entries in the AstrometryDB for
        the observation with `observationID`.

        Parameters
        ==========
        observationID : str
            base rootname for observation to be updated (eg., `iab001a1q`)

        Return
        ======
        entry : obj
            Database entry for this observation, if found.
            It will return None if there was an error in accessing the
            database and `self.raise_errors` was not set to True.
        """
        if not self.perform_step:
            return None

        serviceEndPoint = self.serviceLocation + \
            'observation/read/' + observationID

        try:
            logger.info('Accessing AstrometryDB service :')
            logger.info('\t{}'.format(serviceEndPoint))
            r = requests.get(serviceEndPoint, headers=self.headers)
            if r.status_code == requests.codes.ok:
                logger.info('AstrometryDB service call succeeded')
            elif r.status_code == 404:
                # This code gets returned if exposure is not found in database
                # Never fail for this case since all new observations
                # will result in this error
                logger.info("No solutions found in database for {}".
                            format(observationID))
                self.new_observation = True
            else:
                logger.warning(" AstrometryDB service call failed")
                logger.warning("    Status: {}".format(r.status_code))
                logger.warning("    {}".format(r.reason))
                if self.raise_errors:
                    e = "AstrometryDB service could not be connected!"
                    raise requests.RequestException(e)
                else:
                    return None
        except Exception:
            logger.warning('AstrometryDB service call failed')
            logger.warning("    Status: {}".format(r.status_code))
            logger.warning("    {}".format(r.reason))

            if self.raise_errors:
                l = 'AstrometryDB service call failed with reason:\n\t"{}"'.\
                    format(r.reason)
                l += '\n\tStatus code = {}'.format(r.status_code)
                raise requests.RequestException(l)
            else:
                return None
        return r

    def getObservation(self, observationID):
        """Get solutions for observation from AstrometryDB.

        Parameters
        ==========
        observationID : str
            base rootname for observation to be updated (eg., `iab001a1q`)

        Return
        ======
        headerlets : dict
            Dictionary containing all solutions found for exposure in the
            form of headerlets labelled by the name given to the solution in
            the database.

        best_solution_id : str
            WCSNAME of the WCS solution flagged as 'best' in the astrometry
            database for the observation.  The 'best' solution should be the
            one that aligns the observation as close to the GAIA frame as
            possible.

        """
        if not self.perform_step:
            return None, None

        if not self.available:
            logger.warning("AstrometryDB not available.")
            logger.warning("NO Updates performed for {}".format(observationID))
            if self.raise_errors:
                raise ConnectionError("AstrometryDB not accessible.")
            else:
                return None, None

        r = self.findObservation(observationID)
        best_solution_id = None

        if r is None or self.new_observation:
            return r, best_solution_id
        else:
            # Now, interpret return value for observation into separate
            # headerlets to be appended to observation
            headerlets = {}
            tree = BytesIO(r.content)
            root = etree.parse(tree)

            # Convert returned solutions specified in XML into dictionaries
            solutions = []
            for solution in root.iter('solution'):
                sinfo = {}
                for field in solution.iter():
                    if field.tag != 'solution':
                        sinfo[field.tag] = field.text
                solutions.append(sinfo)

            # interpret bestSolutionID from tree
            for bestID in root.iter('bestSolutionID'):
                best_solution_id = bestID.text

            # Now use these names to get the actual updated solutions
            headers = {'Content-Type': 'application/fits'}
            for solution_info in solutions:
                solutionID = solution_info['solutionID']
                wcsName = solution_info['wcsName']
                if solutionID is None:
                    continue
                # Translate bestSolutionID into wcsName, if one is specified
                if best_solution_id and best_solution_id == solutionID:
                    best_solution_id = wcsName
                serviceEndPoint = self.serviceLocation + \
                    'observation/read/' + observationID + \
                    '?wcsname=' + wcsName
                logger.info('Retrieving astrometrically-updated WCS "{}" for observation "{}"'.format(wcsName, observationID))
                r_solution = requests.get(serviceEndPoint, headers=headers)
                if r_solution.status_code == requests.codes.ok:
                    hlet_bytes = BytesIO(r_solution.content).getvalue()
                    hlet = headerlet.Headerlet(file=hlet_bytes)
                    hlet.init_attrs()
                    if hlet[0].header['hdrname'] == 'OPUS':
                        hdrdate = hlet[0].header['date'].split('T')[0]
                        hlet[0].header['hdrname'] += hdrdate
                    headerlets[solutionID] = hlet

            if not solutions:
                logger.warning("No new WCS's found for {}".format(observationID))
                logger.warning("No updates performed...")

            return headerlets, best_solution_id

    def apply_new_apriori(self, obsname):
        """ Compute and apply a new a priori WCS based on offsets from astrometry database.

        Parameters
        -----------
        obsname : str
            Full filename or `astropy.io.fits.HDUList` object \
            for the observation to be corrected

        Returns
        -------
        wcsname : str
            Value of WCSNAME keyword for this new WCS

        """
        filename = os.path.basename(obsname.filename())

        # Start by archiving and writing out pipeline-default based on new IDCTAB
        # Save this new WCS as a headerlet extension and separate headerlet file
        wname = obsname[('sci', 1)].header['wcsname']
        hlet_extns = headerlet.get_headerlet_kw_names(obsname, kw='EXTVER')

        # newly processed data will not have any hlet_extns, so we need to account for that
        newhlt = max(hlet_extns) + 1 if len(hlet_extns) > 0 else 1
        hlet_names = [obsname[('hdrlet', e)].header['wcsname'] for e in hlet_extns]

        if wname not in hlet_names:
            wname_hash = hashlib.sha1(wname.encode()).hexdigest()[:6]
            hdrname = "{}_{}".format(filename.replace('.fits', ''), wname_hash)
            # Create full filename for headerlet:
            hfilename = "{}_hlet.fits".format(hdrname)
            logger.info("Archiving pipeline-default WCS {} to {}".format(wname, filename))
            descrip = "Pipeline-default WCS"
            numext = len(obsname)
            headerlet.archive_as_headerlet(obsname, hfilename,
                                           sciext='SCI',
                                           wcskey="PRIMARY",
                                           author="stwcs.updatewcs",
                                           descrip=descrip)
            obsname[numext].header['EXTVER'] = newhlt

            # Now, write out pipeline-default WCS to a unique headerlet file
            logger.info("Writing out pipeline-default WCS {} to headerlet file: {}".format(wname, hfilename))
            headerlet.extract_headerlet(obsname, hfilename, extnum=numext, clobber=True)

        # We need to create new apriori WCS based on new IDCTAB
        # Get guide star offsets from DB
        # Getting observationID (rootname) from header to avoid
        # potential issues with actual filename being changed
        pix_offsets = find_gsc_offset(obsname)

        # Determine rootname for IDCTAB
        idctab = obsname[0].header['IDCTAB']
        idcroot = os.path.basename(fileutil.osfn(idctab)).split('_')[0]
        # Create WCSNAME for this new a priori WCS
        if pix_offsets['catalog']:
            wname = 'IDC_{}-{}'.format(idcroot, pix_offsets['catalog'])
        else:
            wname = 'IDC_{}'.format(idcroot)
        # Compute and add new solution if it is not already an alternate WCS
        # Save this new WCS as a headerlet extension and separate headerlet file
        wname_hash = hashlib.sha1(wname.encode()).hexdigest()[:6]
        hdrname = "{}_{}".format(filename.replace('.fits', ''), wname_hash)
        # Create full filename for headerlet:
        hfilename = "{}_hlet.fits".format(hdrname)

        # apply offsets to image using the same tangent plane
        # which was used to compute the offsets
        updatehdr.updatewcs_with_shift(obsname, pix_offsets['expwcs'],
                                       hdrname=hfilename,
                                       wcsname=wname, reusename=True,
                                       fitgeom='rscale', rot=0.0, scale=1.0,
                                       xsh=pix_offsets['delta_x'],
                                       ysh=pix_offsets['delta_y'],
                                       verbose=False, force=True)

        sci_extns = updatehdr.get_ext_list(obsname, extname='SCI')

        # Update list of alternate WCSs
        alt_wnames = _get_alt_wcsnames(obsname)
        # Remove any alternate WCS solutions which are not based on the current IDCTAB
        for alt_key, alt_name in alt_wnames.items():
            if idcroot not in alt_name and alt_key not in [' ', 'O']:
                for sci_extn in sci_extns:
                    altwcs.deleteWCS(obsname, sci_extn, wcskey=alt_key)

        if wname not in alt_wnames.values():
            for sci_ext in sci_extns:
                # Create alternate WCS for this new WCS
                _, wname = altwcs.archive_wcs(obsname, sci_ext,
                                                   wcsname=wname,
                                                   mode=altwcs.ArchiveMode.QUIET_ABORT)
                logger.info('Archived {} in {}'.format(wname, sci_ext))

        # Get updated list of headerlet names
        hlet_extns = headerlet.get_headerlet_kw_names(obsname, kw='EXTVER')
        hlet_names = [obsname[('hdrlet', e)].header['wcsname'] for e in hlet_extns]
        if wname not in hlet_names:
            newhlt += 1
            descrip = "A Priori WCS based on ICRS guide star positions"
            logger.info("Appending a priori WCS {} to {}".format(wname, filename))
            headerlet.archive_as_headerlet(obsname, hfilename,
                                           sciext='SCI',
                                           wcskey="PRIMARY",
                                           author="stwcs.updatewcs",
                                           descrip=descrip)

            hlet_extns = headerlet.find_headerlet_HDUs(obsname, strict=False)
            newext = max(hlet_extns)

            obsname[newext].header['EXTVER'] = newext
            # Update a priori headerlet with offsets used to compute new WCS
            apriori_hdr = obsname[newext].headerlet[0].header
            apriori_hdr['D_RA'] = pix_offsets['delta_ra']
            apriori_hdr['D_DEC'] = pix_offsets['delta_dec']
            apriori_hdr['D_ROLL'] = pix_offsets['roll']
            apriori_hdr['D_SCALE'] = pix_offsets['scale']
            apriori_hdr['NMATCH'] = 2
            apriori_hdr['CATALOG'] = pix_offsets['catalog']

        if not os.path.exists(hfilename):
            # Now, write out new a priori WCS to a unique headerlet file
            logger.info("Writing out a priori WCS {} to headerlet file: {}".format(wname, hfilename))
            try:
                newext = headerlet.find_headerlet_HDUs(obsname, hdrname=hfilename)[0]
            except ValueError:
                newext = headerlet.find_headerlet_HDUs(obsname, hdrname=hdrname)[0]
            headerlet.extract_headerlet(obsname, hfilename, extnum=newext)

        return wname

    def isAvailable(self, max_tries=3, force_timeout=False):
        """Tests the availability of the astrometry database

        Parameters
        ----------
        max_tries : int, optional
            Number of attempts to query the db, by default 3

        force_timeout : bool, optional
            If True, forces a timeout to test error handling, by default False

        Raises
        ------
        ConnectionRefusedError
            If the service is unavailable after max_tries attempts and raise_errors is True
        ConnectionError
            If there is a network-related error during the request and raise_errors is True
        """
        if not self.perform_step:
            return

        # added max_tries limit to 10
        if max_tries > 10:
            max_tries = 10
            logger.warning("max_tries limited to 10")

        # Set timeout based on testing flag
        timeout = 1e-15 if force_timeout else 5.0  # values in seconds.

        service_endpoint = self.serviceLocation + "availability"
        logger.info(f"AstrometryDB URL: {service_endpoint}")

        for attempt in range(max_tries):
            try:
                response = requests.get(
                    service_endpoint, headers=self.headers, timeout=timeout
                )

                if response.status_code == requests.codes.ok:
                    logger.info("AstrometryDB service available")
                    self._set_availability_status(
                        response.status_code, "Available", True
                    )
                    return
                else:
                    self._log_service_failure(response.status_code, response.text)
                    self._set_availability_status(
                        response.status_code, response.text, False
                    )

                    if self._is_final_attempt(attempt, max_tries):
                        self._handle_final_failure("AstrometryDB service unavailable!")
                        return

                    self._log_retry_message(attempt, max_tries)

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException,
            ) as err:
                logger.warning(f"AstrometryDB connection failed: {service_endpoint}")
                logger.warning(f"Error: {err}")
                self.available = False

                if self._is_final_attempt(attempt, max_tries):
                    if self.raise_errors:
                        raise ConnectionError(
                            f"Failed to connect to AstrometryDB: {err}"
                        ) from err
                    else:
                        logger.warning("AstrometryDB service unavailable!")
                        return

                self._log_retry_message(attempt, max_tries)
            except Exception as e:
                logger.warning(f"Unexpected error: {e}")
                self.available = False
                if self.raise_errors:
                    raise
                return

    def _set_availability_status(self, status_code, status_text, is_available):
        """Helper method to set availability status consistently."""
        self.available_code["code"] = status_code
        self.available_code["text"] = status_text
        self.available = is_available

    def _log_service_failure(self, status_code, response_text):
        """Helper method to log service failure details."""
        logger.warning(f"AstrometryDB service call failed:")
        logger.warning(f"  URL: {self.serviceLocation}")
        logger.warning(f"  Status: {status_code}")
        logger.warning(f"  Response: {response_text}")

    def _is_final_attempt(self, current_attempt, max_tries):
        """Check if this is the final attempt."""
        return current_attempt == (max_tries - 1)

    def _handle_final_failure(self, error_message):
        """Handle the final failure attempt."""
        if self.raise_errors:
            raise ConnectionRefusedError(error_message)
        else:
            logger.warning(error_message)

    def _log_retry_message(self, attempt, max_tries):
        """Log retry message with consistent formatting."""
        # shorten time between retries if testing
        if self.testing:
            wait_time = 0.1
        else:
            wait_time = 60
        remaining_attempts = max_tries - attempt - 1
        if remaining_attempts > 0:
            logger.warning(
                f"AstrometryDB service unavailable! Retrying in {wait_time} seconds " +
                f"({remaining_attempts} attempt(s) remaining)"
            )
            time.sleep(wait_time)


#
# Supporting functions
#
def find_gsc_offset(obsname):
    """Find the GSC to GAIA offset based on guide star coordinates

    Parameters
    ----------
    obsname : str
        Full filename or (preferably)`astropy.io.fits.HDUList` object of
        image to be processed.

    NOTES
    ------
    The default transform is GSC2-GAIA. The options were primarily for transforming
    individual objects from the catalogs and that is not specified in the limited
    documentation. The ipppssoot input is a special case where it pulls the gsids,
    epoch and refframe from the dms databases and overrides the transform using this logic::

        REFFRAME=GSC1 sets GSC1-GAIA
        REFFRAME=ICRS and EPOCH < 2017.75 sets GSC2-GAIA
        REFFRAME=ICRS and EPOCH > 2017.75 sets no-offset since it's already in GAIA frame

    Returns
    -------
    response : dict
        Dict of offset, roll and scale in decimal degrees and pixels for image
        based on correction to guide star coordinates relative to GAIA.
        Keys: ``delta_x``, ``delta_y``, ``delta_ra``, ``delta_dec``, ``roll``, ``scale``,
        ``expwcs``, ``catalog``, ``dGSinputRA``, ``dGSoutputRA``, ``dGSinputDEC``, ``dGSoutputDEC``
    """
    # check to see whether any URL has been specified as an
    # environmental variable.
    if gsss_url_envvar in os.environ:
        gsss_serviceLocation = os.environ[gsss_url_envvar]
    else:
        gsss_serviceLocation = gsss_url

    # Insure input is a fits.HDUList object, if originally provided as a filename(str)
    close_obj = False
    if isinstance(obsname, str):
        obsname = fits.open(obsname)
        close_obj = True

    if 'rootname' in obsname[0].header:
        ippssoot = obsname[0].header['rootname'].upper()
    else:
        ippssoot = fileutil.buildNewRootname(obsname).upper()

    expwcs = build_reference_wcs(obsname)
    # Initialize variables for cases where no offsets are available.
    response = {
        "delta_ra": 0.0,
        "delta_dec": 0.0,
        "roll": 0.0,
        "scale": 1.0,
        "dGSinputRA": 0.0,
        "dGSoutputRA": 0.0,
        "dGSinputDEC": 0.0,
        "dGSoutputDEC": 0.0,
        "catalog": None,
        "message": "",
        "expwcs": expwcs,
        "delta_x": 0.0,
        "delta_y": 0.0,
    }
    # Define what service needs to be used to get the offsets
    serviceType = "GSCConvert/GSCconvert.aspx"
    spec_str = "IPPPSSOOT={}"
    spec = spec_str.format(ippssoot)
    serviceUrl = "{}/{}?{}".format(gsss_serviceLocation, serviceType, spec)
    try:
        rawcat = requests.get(serviceUrl)
    except ConnectionError:
        logger.warning("Problem accessing service")
        return response
    if not rawcat.ok:
        logger.warning("Problem accessing service with:\n{}".format(serviceUrl))

        # It's possible rawcat.status_code to be 200 and rawcat.ok to be False
        return response
    if rawcat.status_code == requests.codes.ok:
        logger.info("gsReference service retrieved {}".format(ippssoot))
        refXMLtree = etree.fromstring(rawcat.content)
        message = refXMLtree.findtext('msg')
        response["message"] = message
        if message.split()[0] == "Success":
            response = {
                "delta_ra": float(refXMLtree.findtext('deltaRA')),
                "delta_dec": float(refXMLtree.findtext('deltaDEC')),
                "roll": float(refXMLtree.findtext('deltaROLL')),
                "scale": float(refXMLtree.findtext('deltaSCALE')),
                "dGSinputRA": float(refXMLtree.findtext('dGSinputRA')),
                "dGSinputDEC": float(refXMLtree.findtext('dGSinputDEC')),
                "dGSoutputRA": float(refXMLtree.findtext('dGSoutputRA')),
                "dGSoutputDEC": float(refXMLtree.findtext('dGSoutputDEC')),
                "catalog": refXMLtree.findtext('outputCatalog'),
                "message": message,
                "expwcs": expwcs,
                "delta_x": 0.0,
                "delta_y": 0.0,
                }
        else:
            # status_code == 200 but message indicates "Failure"
            return response

    # Use GS coordinate as reference point
    old_gs = (response["dGSinputRA"], response["dGSinputDEC"])
    new_gs = (response["dGSoutputRA"], response["dGSoutputDEC"])

    # This check is a workaround an issue with the service where delta_ra/dec are 0
    # but the computed scale is NaN.
    if response["delta_ra"] != 0.0 and response["delta_dec"] != 0.0:

        # Compute tangent plane for this observation
        wcsframe = expwcs.wcs.radesys.lower()

        # Use WCS to compute offset in pixels of shift applied to WCS Reference pixel
        # RA,Dec of ref pixel in decimal degrees
        crval = SkyCoord(expwcs.wcs.crval[0], expwcs.wcs.crval[1],
                         unit='deg', frame=wcsframe)

        # Define SkyCoord for Guide Star using old/original coordinates used to
        # originally compute WCS for exposure
        old_gs_coord = SkyCoord(old_gs[0], old_gs[1], unit='deg', frame=wcsframe)
        sof_old = old_gs_coord.skyoffset_frame()
        # Define new SkyOffsetFrame based on new GS coords
        new_gs_coord = SkyCoord(new_gs[0], new_gs[1], unit='deg',
                           frame=wcsframe)
        # Determine offset from old GS position to the new GS position
        sof_new = new_gs_coord.transform_to(sof_old)
        # Compute new CRVAL position as old CRVAL+GS offset (sof_new)
        new_crval_coord = SkyCoord(sof_new.lon.arcsec, sof_new.lat.arcsec,
                             unit='arcsecond',
                             frame=crval.skyoffset_frame())
        # Return RA/Dec for new/updated CRVAL position
        new_crval = new_crval_coord.icrs

        # Compute offset in pixels for new CRVAL
        newpix = expwcs.all_world2pix(new_crval.ra.value, new_crval.dec.value, 1)
        deltaxy = expwcs.wcs.crpix - newpix  # offset from ref pixel position
        response["delta_x"] = deltaxy[0]
        response["delta_y"] = deltaxy[1]

    else:
        logger.warning("GSC returned zero offsets in RA, DEC for guide star")

    if close_obj:
        obsname.close()
    return response


def build_reference_wcs(input, sciname='sci'):
    """Create the reference WCS based on all the inputs for a field

    Parameters
    -----------
    input : str or `astropy.io.fits.HDUList` object or list
        Full filename or `fits.HDUList` object
         of the observation to use in building a tangent plane WCS.
         If a list of filenames or HDUList objects are provided, then all
         of them will be used to generate the reference WCS for the entire
         field of view.

    sciname : str
        EXTNAME of extensions which have WCS information for the observation

    """
    # Insure that input is a list at all times.
    # If a single filename (str) or single HDUList is provided, wrap it as a list.
    if not isinstance(input, list) or isinstance(input, fits.HDUList):
        input = [input]

    # Create a composite field-of-view for all inputs
    wcslist = []
    for img in input:
        nsci = fileutil.countExtn(img)
        for num in range(nsci):
            extname = (sciname, num + 1)
            if sciname == 'sci':
                extwcs = HSTWCS(img, ext=extname)
            else:
                # Working with HDRLET as input and do the best we can...
                extwcs = read_hlet_wcs(img, ext=extname)

            wcslist.append(extwcs)

    # This default output WCS will have the same plate-scale and orientation
    # as the first chip in the list, which for WFPC2 data means the PC.
    # Fortunately, for alignment, this doesn't matter since no resampling of
    # data will be performed
    outwcs = utils.output_wcs(wcslist)

    return outwcs

def read_hlet_wcs(filename, ext):
    """Insure `~stwcs.wcsutil.HSTWCS` includes all attributes of a full image WCS.

    For headerlets, the WCS does not contain information about the size of the
    image, as the image array is not present in the headerlet.
    """
    hstwcs = HSTWCS(filename, ext=ext)
    if hstwcs.naxis1 is None:
        hstwcs.naxis1 = int(hstwcs.wcs.crpix[0] * 2.)  # Assume crpix is center of chip
        hstwcs.naxis2 = int(hstwcs.wcs.crpix[1] * 2.)

    return hstwcs

def _get_alt_wcsnames(hdu, sci_extn=('SCI', 1)):
    keys = altwcs.wcskeys(hdu, sci_extn)
    keys.remove(' ')
    keys.remove('O')
    alt_wnames = {key: hdu[('sci', 1)].header['WCSNAME%s' % key] for key in keys}

    return alt_wnames

def apply_astrometric_updates(obsnames, **pars):
    """Apply new astrometric solutions to observation.

    Functional stand-alone interface for applying new astrometric solutions
    found in the astrometry dB to the given observation(s).

    Parameters
    ==========
    obsnames : str, list
        Filename or list of filenames of observation(s) to be updated

    url : str, optional
        URL of astrometry database web-interface to use.
        If None (default), it will use built-in URL for STScI web-interface

    raise_errors : bool, optional
        Specify whether or not to raise Exceptions and stop processing
        when an error in either accessing the database,
        retrieving a solution from the database or applying the new
        solution to the observation. If None, it will look to see whether
        the environmental variable `RAISE_PIPELINE_ERRORS` was set,
        otherwise, it will default to 'False'.

    """
    if not isinstance(obsnames, list):
        obsnames = [obsnames]

    url = pars.get('url', None)
    raise_errors = pars.get('raise_errors', None)

    db = AstrometryDB(url=url, raise_errors=raise_errors)
    for obs in obsnames:
        db.updateObs(obs)
