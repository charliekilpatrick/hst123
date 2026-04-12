"""Photometry table display and formatting (instrument names, SNANA output)."""


def format_instrument_display_name(inst):
    """Return display string for instrument (e.g. WFC3/UVIS, ACS/WFC)."""
    inst = inst.lower()
    if "wfc3" in inst and "uvis" in inst:
        return "WFC3/UVIS"
    if "wfc3" in inst and "ir" in inst:
        return "WFC3/IR"
    if "acs" in inst and "wfc" in inst:
        return "ACS/WFC"
    if "acs" in inst and "hrc" in inst:
        return "ACS/HRC"
    if "acs" in inst and "sbc" in inst:
        return "ACS/SBC"
    if "wfpc2" in inst:
        return "WFPC2"
    if "_" in inst:
        return inst.upper().replace("_full", "").replace("_", "/")
    return inst.upper()


def show_photometry_data(
    phottable, form, header, units, log, file=None, avg=False, log_rows=True
):
    """Log and optionally write photometry rows (form uses date, inst, filt, exp, mag, err, optional lim)."""
    if avg:
        if log_rows:
            log.info("# Average photometry")
        if file:
            file.write("\n# Average Photometry \n")
    else:
        for key in phottable.meta.keys():
            out = "# {key} = {val}"
            line = out.format(key=key, val=phottable.meta[key])
            if file:
                file.write(line + "\n")
            if log_rows:
                log.info(line)

    if header:
        if log_rows:
            log.info(header)
    if file and header:
        file.write(header + "\n")
    if units:
        if log_rows:
            log.info(units)
    if file and units:
        file.write(units + "\n")

    for row in phottable:
        inst = format_instrument_display_name(row["INSTRUMENT"])
        date = "%7.5f" % row["MJD"]
        if row["MJD"] == 99999.0:
            date = "-----------"

        datakeys = {
            "date": date,
            "inst": inst,
            "filt": row["FILTER"].upper(),
            "exp": "%7.4f" % row["EXPTIME"],
            "mag": "%3.4f" % row["MAGNITUDE"],
            "err": "%3.4f" % row["MAGNITUDE_ERROR"],
        }
        if "lim" in form:
            datakeys["lim"] = "%3.4f" % row["LIMIT"]

        line = form.format(**datakeys)
        if log_rows:
            log.info(line)
        if file:
            file.write(line + "\n")

    if file:
        file.write("\n")


def write_snana_photometry(phottable, file, coord, object_name, output_zpt):
    """Write photometry to file in SNANA format (header + VARLIST + OBS lines)."""
    header = "SNID: {obj} \nRA: {ra} \nDECL: {dec} \n\n"
    header = header.format(
        obj=object_name, ra=coord.ra.degree, dec=coord.dec.degree
    )
    file.write(header)
    file.write("VARLIST: MJD FLT FLUXCAL MAG MAGERR \n")

    form = "OBS: {date: <16} {instfilt: <20} {flux: <16} {fluxerr: <16} "
    form += "{mag: <16} {magerr: <6} \n"

    for row in phottable:
        inst = format_instrument_display_name(row["INSTRUMENT"])
        date = "%7.5f" % row["MJD"]
        mag = row["MAGNITUDE"]
        magerr = row["MAGNITUDE_ERROR"]
        flux = 10 ** (0.4 * (output_zpt - mag))
        fluxerr = 1.0 / 1.086 * magerr * flux

        datakeys = {
            "date": date,
            "instfilt": inst + "-" + row["FILTER"].upper(),
            "flux": flux,
            "fluxerr": fluxerr,
            "mag": mag,
            "magerr": magerr,
        }
        file.write(form.format(**datakeys))
    file.write("\n")


def show_photometry(
    final_photometry,
    latex=False,
    show=True,
    f=None,
    snana=False,
    coord=None,
    options=None,
    log=None,
    log_rows=None,
):
    """Display/write photometry as LaTeX, plain text, and/or SNANA (snana needs coord, options). Returns None."""
    if log is None:
        import logging
        log = logging.getLogger(__name__)

    if log_rows is None:
        log_rows = f is None

    keys = final_photometry.colnames
    if (
        "INSTRUMENT" not in keys
        or "FILTER" not in keys
        or "MJD" not in keys
        or "MAGNITUDE" not in keys
        or "MAGNITUDE_ERROR" not in keys
        or "EXPTIME" not in keys
    ):
        log.error("Photometry table has a key error.")
        return None

    avg_photometry = final_photometry[final_photometry["IS_AVG"] == 1]
    if len(avg_photometry) > 0:
        final_photometry = final_photometry[final_photometry["IS_AVG"] == 0]

    if snana and f is not None and coord is not None and options is not None:
        objname = "dummy"
        if options.get("args") and getattr(options["args"], "object", None):
            objname = options["args"].object
        zpt = options["global_defaults"]["output_zpt"]
        write_snana_photometry(avg_photometry, f, coord, objname, zpt)

    if latex:
        form = "{date: <10} & {inst: <10} & {filt: <10} "
        form += "{exp: <10} & {mag: <8} & {err: <8} \\\\"
        header = form.format(
            date="MJD",
            inst="Instrument",
            filt="Filter",
            exp="Exposure",
            mag="Magnitude",
            err="Uncertainty",
        )
        units = (
            form.format(
                date="(MJD)", inst="", filt="", exp="(s)", mag="", err=""
            )
            + "\\hline\\hline"
        )
        if len(final_photometry) > 0:
            show_photometry_data(
                final_photometry,
                form,
                header,
                units,
                log,
                file=f,
                log_rows=log_rows,
            )
        if len(avg_photometry) > 0:
            show_photometry_data(
                avg_photometry,
                form,
                header,
                units,
                log,
                file=f,
                avg=True,
                log_rows=log_rows,
            )

    if show:
        form = "{date: <12} {inst: <10} {filt: <8} "
        form += "{exp: <14} {mag: <9} {err: <11}"
        headkeys = {
            "date": "# MJD",
            "inst": "Instrument",
            "filt": "Filter",
            "exp": "Exposure",
            "mag": "Magnitude",
            "err": "Uncertainty",
        }
        if "LIMIT" in keys:
            form += " {lim: <10}"
            headkeys["lim"] = "Limit"
        header = form.format(**headkeys)
        if len(final_photometry) > 0:
            show_photometry_data(
                final_photometry,
                form,
                header,
                "",
                log,
                file=f,
                log_rows=log_rows,
            )
        if len(avg_photometry) > 0:
            show_photometry_data(
                avg_photometry,
                form,
                header,
                "",
                log,
                file=f,
                avg=True,
                log_rows=log_rows,
            )
    return None
