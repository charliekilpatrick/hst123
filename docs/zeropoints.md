# Zero points

hst123 reports magnitudes in the **AB system** by default. DOLPHOT uses the **Vega** system; hst123 converts using header keywords.

All HST images provide `PHOTFLAM` and `PHOTPLAM` per chip. The AB zero point used in hst123 is:

```text
ZP_AB = -2.5 * log10(PHOTFLAM) - 5 * log10(PHOTPLAM) - 2.408
```

References:

- [WFPC2 zeropoints](http://www.stsci.edu/instruments/wfpc2/Wfpc2_dhb/wfpc2_ch52.html)
- [ACS zeropoints](http://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints)
- [WFC3 photometric calibration](http://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration)
