#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for emission line class and database.

Joris Witstok, 18 May 2020
"""

import os

import numpy as np

from aux.roman import int_to_roman

import seaborn as sns

# Line wavelengths and names
lines = ["Lya", "CIII", "CIV", "OIII", "HeII", "SiIII", "Halpha", "Hbeta", "OII", "NII", "SII"]

# Lya line
Lyaline_UV = ["Lya"]
Lyaline_wl = {"Lya": 1215.67}
Lyaline_wls = {"Lya": (1215.67,)}
Lyaline_col = {"Lya": sns.color_palette("Set2", 8)[3]}
Lyaline_lab = {"Lya": r"Ly$\mathrm{\alpha}$"}
Lyaline_slab = {"Lya": r"Ly \alpha"}

# Lya line for stack of all objects (without redshift bins)
aLyaline = ["aLya"]
aLyaline_wl = {"aLya": 1215.67}
aLyaline_wls = {"aLya": (1215.67,)}
aLyaline_col = {"aLya": sns.color_palette("Set2", 8)[3]}
aLyaline_lab = {"aLya": r"Ly$\mathrm{\alpha}$"}
aLyaline_slab = {"aLya": r"Ly \alpha"}

# Other nebular lines
# Sources: Morton (1991), https://ui.adsabs.harvard.edu/abs/1991ApJS...77..119M/abstract), https://physics.nist.gov/PhysRefData/ASD/lines_form.html
nlines_UV = ["CIII", "CIV", "OIII", "HeII", "SiIII", "NV"]
nline_wl = {"CIII": 1907.7085, "CIV": 1549.5,
            "OIII": 1663.47, "HeII": 1640.4, "SiIII": 1887.0, "NV": 1240.146} # Ly-a/CIII/CIV/OIII/HeII/SiIII/NV central wavelength in Angstrom
nline_wls = {"CIII": (1906.683, 1908.734), "CIV": (1548.195, 1550.77), "OIII": (1660.809, 1666.150),
            "HeII": (1640.4,), "SiIII": (1881.96, 1892.03), "NV": (1240.146,)} # Ly-a/CIII/CIV/OIII/HeII/SiIII/NV specific wavelength(s) in Angstrom
nline_col = {"CIII": sns.color_palette("Set2", 8)[1], "CIV": sns.color_palette("Set2", 8)[5], "OIII": sns.color_palette("Set2", 8)[4],
            "HeII": sns.color_palette("Set2", 8)[0], "SiIII": sns.color_palette("Set2", 8)[2], "NV": sns.color_palette("Set1", 9)[0]}
nline_lab = {"CIII": r"$\mathrm{C \, III}$", "CIV": r"$\mathrm{C \, IV}$", "OIII": r"$\mathrm{O \, III]}$",
            "HeII": r"$\mathrm{He \, II}$", "SiIII": r"$\mathrm{Si \, III}$", "NV": r"$\mathrm{N \, V}$"}
nline_slab = {"CIII": r"C III", "CIV": r"C IV", "OIII": r"O III",
            "HeII": r"He II", "SiIII": r"Si III", "NV": r"N V"}

# Near-UV lines (source: https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
lines_NUV = ["MgII"]
line_wl_NUV = {"MgII": 2796.352}
line_wls_NUV = {"MgII": (2796.352,)}
line_col_NUV = {"MgII": sns.color_palette("Set1", 9)[3]}
line_lab_NUV = {"MgII": r"\mathrm{Mg \, II}"}
line_slab_NUV = {"MgII": r"Mg II"}

# Stellar lines
slines_UV = ["sCIII", "sOIV", "sSV"]
sline_wl = {"sCIII": 1175.71, "sOIV": 1343.25, "sSV": 1501.76}
sline_wls = {"sCIII": (1175.71,), "sOIV": (1342.99, 1343.51), "sSV": (1501.76,)}
sline_col = {"sCIII": sns.color_palette("Set3", 14)[3], "sOIV": sns.color_palette("Set3", 14)[9], "sSV": sns.color_palette("Set3", 14)[7]}
sline_lab = {"sCIII": r"$\mathrm{C \, III}$ (st. abs.)", "sOIV": r"$\mathrm{O \, IV}$ (st. abs.)", "sSV": r"$\mathrm{S \, V}$ (st. abs.)"}
sline_slab = {"sCIII": r"C III (st. abs.)", "sOIV": r"O IV (st. abs.)", "sSV": r"S V (st. abs.)"}

# Other absorption lines
alines_UV = ["sSiIIa", "sSiIIb", "sCII", "sSiIIc"]
aline_wl = {"sSiIIa": 1260.4221, "sSiIIb": 1303.27, "sCII": 1334.5323, "sSiIIc": 1526.7066}
aline_wls = {"sSiIIa": (1260.4221,), "sSiIIb": (1303.27,), "sCII": (1334.5323,), "sSiIIc": (1526.7066,)}
aline_col = {"sSiIIa": sns.color_palette("Set3", 14)[8], "sSiIIb": sns.color_palette("Set3", 14)[8],
            "sCII": sns.color_palette("Set3", 14)[11], "sSiIIc": sns.color_palette("Set3", 14)[8]}
aline_lab = {"sSiIIa": r"$\mathrm{Si \, II}$ (abs.)", "sSiIIb": r"$\mathrm{O \, I + Si \, II}$ (abs.)", "sCII": r"$\mathrm{C \, II}$ (abs.)", "sSiIIc": r"$\mathrm{Si \, II}$ (abs.)"}
aline_slab = {"sSiIIa": r"Si II (abs.)", "sSiIIb": r"O I + Si II (abs.)", "sCII": r"C II (abs.)", "sSiIIc": r"Si II (abs.)"}

UV_lines = Lyaline_UV + nlines_UV + slines_UV + alines_UV
UV_wl = {**Lyaline_wl, **aLyaline_wl, **nline_wl, **sline_wl, **aline_wl}
UV_wls = {**Lyaline_wls, **aLyaline_wls, **nline_wls, **sline_wls, **aline_wls}
UV_col = {**Lyaline_col, **aLyaline_col, **nline_col, **sline_col, **aline_col}

UV_lines_all = UV_lines + lines_NUV
UV_wl_all = {**UV_wl, **line_wl_NUV}
UV_wls_all = {**UV_wls, **line_wls_NUV}
UV_col_all = {**UV_col, **line_col_NUV}

# Optical lines

lines_il = ["OIII_opt", "OII", "Halpha", "Hbeta"]
lines_opt = ["OIII_opt", "OII", "Halpha", "Hbeta", "NII", "SII"]
line_opt_wl = {"OIII_opt": 5008.24, "OII": 3728.4835, "Halpha": 6564.5, "Hbeta": 4861.4,
            "NII": 6549.86, "SII": 6717.395} # OIII/Ha/Hb/OII/NII/SII central wavelength in Angstrom
line_opt_wls = {"OIII_opt": (4932.603, 4960.295, 5008.24,), "Halpha": (6564.5,), "Hbeta": (4861.4,),
            "OII": (3727.092, 3729.875), "NII": (6549.86, 6585.27), "SII": (6716.47, 6718.32)} # OIII/Ha/Hb/OII/NII/SII specific wavelength(s) in Angstrom
line_opt_col = {"OIII_opt": sns.color_palette("Set1", 9)[2], "OII": sns.color_palette("Set1", 9)[3],
                "Halpha": sns.color_palette("Set1", 9)[0], "Hbeta": sns.color_palette("Set1", 9)[1]}
line_opt_lab = {"OIII_opt": r"$\mathrm{[O \, III]}$", "OII": r"$\mathrm{[O \, II]}$",
                "NII": r"$\mathrm{[N \, II]}$", "SII": r"$\mathrm{[S \, II]}$",
                "Halpha": r"H$\mathrm{\alpha}$", "Hbeta": r"H$\mathrm{\beta}$"}
line_opt_slab = {"OIII_opt": r"[O III]", "OII": r"[O II]", "[NII]": r"[N II]", "SII": r"[S II]",
                    "Halpha": r"H \alpha", "Hbeta": r"H \beta"}

lines_all = UV_lines_all + lines_il
wl_all = {**UV_wl_all, **line_opt_wl}
wls_all = {**UV_wls_all, **line_opt_wls}
col_all = {**UV_col_all, **line_opt_col}

line_lab = {**Lyaline_lab, **aLyaline_lab, **nline_lab, **line_lab_NUV, **sline_lab, **aline_lab, **line_opt_lab}

line_slab = {**Lyaline_slab, **aLyaline_slab, **nline_slab, **line_slab_NUV, **sline_slab, **aline_slab, **line_opt_slab}



# Overview of lines
# 
# All wavelengths come from the CHIANTI database (https://www.chiantidatabase.org)
# 
#       Species     Wavelength(s)                   Origin       Kind            Alternative names   Color                               Forbidden line?                 Comments
#
#                                                                Emission/       E.g. Lya                                                Y: yes, N: no,
#                                                                 absorption/                                                            S: semi
#                                                                 both

lines_lib = [

        # Nebular lines: UV (NIII consists of one doublet ground transition and a triplet)

        ("HI",      (1215.6701,),                   "nebular",   "both",         "Lya",              sns.color_palette("Set2", 8)[3],    ('N',),                         ''),
        ("NV",      (1238.821, 1242.804),           "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[0],    ('N', 'N'),                     ''),
        ("NIV",     (1483.321, 1486.496),           "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[5],    ('Y', 'S'),                     ''),
        ("CIV",     (1548.187, 1550.772),           "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[5],    ('N', 'N'),                     ''),
        ("HeII",    (1640.42,),                     "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[0],    ('N',),                         "Mean wavelength of 7 transitions"),
        ("OIII",    (1660.809, 1666.150),           "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[4],    ('S', 'S'),                     ''),
        ("NIII",    (1746.822, 1748.646, 1749.674, \
                        1752.160, 1753.995),        "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[4],    ('S', 'S', 'S', 'S', 'S'),      ''),
        ("SiIII",   (1882.7070, 1892.029),          "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[2],    ('Y', 'S'),                     ''),
        ("CIII",    (1906.683, 1908.734),           "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[1],    ('Y', 'S'),                     ''),
        ("MgII",    (2796.352, 2803.531),           "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[3],    ('N', 'N'),                     ''),

        # Nebular lines: optical

        ("OII",     (3727.092, 3729.875),           "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[2],    ('Y', 'Y'),                     ''),
        ("NeIII",   (3869.849,),                    "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[2],    ('Y',),                         ''),
        ("NeIII",   (3968.59,),                     "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[2],    ('Y',),                         ''),
        ("HI",      (3971.198,),                    "nebular",   "emission",     "Hepsilon",         sns.color_palette("Set1", 9)[1],    ('N',),                         ''),
        ("HI",      (4102.892,),                    "nebular",   "emission",     "Hdelta",           sns.color_palette("Set1", 9)[1],    ('N',),                         ''),
        ("HeI",     (3889.751,),                    "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[3],    ('N',),                         ''),
        ("HI",      (4341.692,),                    "nebular",   "emission",     "Hgamma",           sns.color_palette("Set1", 9)[1],    ('N',),                         ''),
        ("OIII",    (4364.436,),                    "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[2],    ('Y',),                          ''),
        ("HeII",    (4687.115,),                    "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[0],    ('N',),                         ''),
        ("HI",      (4862.71,),                     "nebular",   "emission",     "Hbeta",            sns.color_palette("Set1", 9)[1],    ('N',),                         ''),
        ("OIII",    (4932.603, 4960.295, 5008.24,), "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[2],    ('Y', 'Y', 'Y'),                ''),
        ("OI",      (6302.046,),                    "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[1],    ('Y',),                         ''),
        ("NII",     (6549.861, 6585.273),           "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[0],    ('Y', 'Y'),                     ''),
        ("HI",      (6564.60,),                     "nebular",   "emission",     "Halpha",           sns.color_palette("Set1", 9)[1],    ('N',),                         ''),
        ("SII",     (6718.295, 6732.674),           "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[0],    ('Y', 'Y'),                     ''),

        # Stellar absorption lines

        ("CIII",    (1175.64,),                     "stellar",   "absorption",   None,               sns.color_palette("Set3", 14)[3],   ('N',),                         "Mean wavelength of 6 transitions"),
        ("OIV",     (1338.615, 1342.990, 1343.514), "stellar",   "absorption",   None,               sns.color_palette("Set3", 14)[9],   ('N', 'N', 'N'),                ''),
        ("SV",      (1501.763,),                    "stellar",   "absorption",   None,               sns.color_palette("Set3", 14)[7],   ('N',),                         ''),

        # IGM absorption lines

        ("SiII",    (1260.422, 1264.738, 1265.002), "IGM",       "absorption",   None,               sns.color_palette("Set3", 14)[8],   ('N', 'N', 'N'),                ''),
        ("OI",      (1302.168, 1304.858, 1306.029), "IGM",       "absorption",   None,               sns.color_palette("Set3", 14)[8],   ('N', 'N', 'N'),                ''),
        ("SiII",    (1304.370, 1309.276),           "IGM",       "absorption",   None,               sns.color_palette("Set3", 14)[8],   ('N', 'N'),                     ''),
        ("CII",     (1334.532, 1335.662, 1335.707), "IGM",       "absorption",   None,               sns.color_palette("Set3", 14)[11],  ('N', 'N', 'N'),                ''),
        ("SiII",    (1526.707,),                    "IGM",       "absorption",   None,               sns.color_palette("Set3", 14)[8],   ('N',),                         ''),

        # Far-infrared fine-structure coolant lines (numerical values in micron, equivalent to 10000 Angstrom)

        ("OIII",    (51.81454500e4,),               "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[3],    ('Y',),                         ''),
        ("NIII",    (57.33945000e4,),               "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[1],    ('Y',),                         ''),
        ("OI",      (63.18516400e4,),               "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[4],    ('Y',),                         ''),
        ("OIII",    (88.35639400e4,),               "nebular",   "emission",     None,               sns.color_palette("Set1", 9)[0],    ('Y',),                         ''),
        ("NII",     (121.8026800e4,),               "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[5],    ('Y',),                         ''),
        ("OI",      (145.53498700e4,),              "nebular",   "emission",     None,               sns.color_palette("Set3", 14)[8],   ('Y',),                         ''),
        ("CII",     (157.73617000e4,),              "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[2],    ('Y',),                         ''),
        ("NII",     (205.33880900e4,),              "nebular",   "emission",     None,               sns.color_palette("Set2", 8)[1],    ('Y',),                         ''),
        ("CI",      (370.41152700e4,),              "nebular",   "emission",     None,               sns.color_palette("Set3", 14)[3],   ('Y',),                         ''),
        ("CI",      (609.12468800e4,),              "nebular",   "emission",     None,               sns.color_palette("Set3", 14)[11],  ('Y',),                         ''),

        # Far-infrared molecular lines (numerical values in micron, equivalent to 10000 Angstrom)

        ("CO1-0",   (2601e4,),                      "molecular", "emission",     None,               sns.color_palette("Set3", 14)[0],   ('N',),                         ''),
        ("CO2-1",   (1300e4,),                      "molecular", "emission",     None,               sns.color_palette("Set3", 14)[1],   ('N',),                         ''),
        ("CO3-2",   (867e4,),                       "molecular", "emission",     None,               sns.color_palette("Set3", 14)[2],   ('N',),                         ''),
        ("CO4-3",   (650.3e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[3],   ('N',),                         ''),
        ("CO5-4",   (520.2e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[4],   ('N',),                         ''),
        ("CO6-5",   (433.6e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[5],   ('N',),                         ''),
        ("CO7-6",   (371.7e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[6],   ('N',),                         ''),
        ("CO8-7",   (325.2e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[7],   ('N',),                         ''),
        ("CO9-8",   (289.1e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[8],   ('N',),                         ''),
        ("CO10-9",  (260.2e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[9],   ('N',),                         ''),

        ("HCN1-0",  (3383e4,),                      "molecular", "emission",     None,               sns.color_palette("Set3", 14)[0],   ('N',),                         ''),
        ("HCN2-1",  (1691e4,),                      "molecular", "emission",     None,               sns.color_palette("Set3", 14)[1],   ('N',),                         ''),
        ("HCN3-2",  (1128e4,),                      "molecular", "emission",     None,               sns.color_palette("Set3", 14)[2],   ('N',),                         ''),
        ("HCN4-3",  (845.7e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[3],   ('N',),                         ''),
        ("HCN5-4",  (676.5e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[4],   ('N',),                         ''),
        ("HCN6-5",  (563.8e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[5],   ('N',),                         ''),
        ("HCN7-6",  (483.3e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[6],   ('N',),                         ''),
        ("HCN8-7",  (422.9e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[7],   ('N',),                         ''),
        ("HCN9-8",  (375.9e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[8],   ('N',),                         ''),
        ("HCN10-9", (338.4e4,),                     "molecular", "emission",     None,               sns.color_palette("Set3", 14)[9],   ('N',),                         ''),
        
        # Unknown/miscellaneous lines

        # 

]

# List of all lines (species/wl_A, e.g. ("CIII", 1908)) in library (use line(*args) for args in all_lines_lib to get all line objects),
# as well as a dictionary (with species as keyword) for all their properties
all_lines_lib = [(line_props[0], int(round(np.mean(line_props[1])))) for line_props in lines_lib]
lines_dict = {line_props[0]: [] for line_props in lines_lib}

for line_props in lines_lib:
    species, wls, origin, kind, alt_name, colour, forb_line, comment = line_props

    lines_dict[species].append((wls, origin, kind, alt_name, colour, forb_line, comment))

# Translation for emission/absorption strings
kind_dict = {"both": "emission/absorption"}

class line:
    """

    Class for holding information about emission/absorption lines. Possible arguments:
    - species name as a string (e.g. "CIII" or "OII")
    - wavelength (e.g. 1908 or 3728), with units set by wl_unit
    - wl_unit (optional): wavelengths units, Angstrom or micron (default: Angstrom)
    - line_i (optional): specify a specific line in a multiplet (e.g. 0 will result in only the 1907 Å line out of the CIII λ 1907, 1909 Å doublet)

    Examples:
    >>> CIII = line("CIII", 1908)
    >>> CIII
    <C III UV nebular emission doublet object at 1908 Angstrom>
    >>> CIII.wls
    (1906.683, 1908.734)

    >>> line("CII", 158, wl_unit="micron")
    <[C II] FIR nebular emission singlet object at 158 micron>

    >>> line("CO", "1-0", atomic=False, wl_unit="micron")
    <CO 1-0 microwave molecular emission singlet object at 2601 micron>

    """

    def __init__(self, *args, wl_unit="Angstrom", atomic=True, line_i=None):
        self.atomic = atomic
        if self.atomic:
            if len(args) == 2:
                assert isinstance(args[0], str)
                assert isinstance(args[1], int) or isinstance(args[1], float)
                ion_idx = np.where([ch.isupper() for ch in args[0]])[0][1]
                element = args[0][:ion_idx]
                ionisation = args[0][ion_idx:]
                if wl_unit == "Angstrom":
                    wl_A = args[1]
                elif wl_unit == "micron":
                    wl_A = args[1] * 1e4
            elif len(args) == 0:
                element = "N/A"
                ionisation = "N/A"
                wl_A = None
            else:
                raise ValueError('arguments should consist of the species and wavelength e.g. line("CIII", 1908)')

            if isinstance(ionisation, int) or isinstance(ionisation, float):
                ionisation = int_to_roman(int(ionisation))
            
            self.element = element[0].upper() + element[1:]
            self.ionisation = ionisation.upper()
            self.molecule = None
            self.transition = None
            self.species = self.element + self.ionisation
        else:
            if len(args) == 2:
                assert isinstance(args[0], str)
                assert isinstance(args[1], str)
                molecule = args[0]
                transition = args[1]
                wl_A = None
            elif len(args) == 0:
                molecule = "N/A"
                transition = "N/A"
                wl_A = None
            else:
                raise ValueError('arguments should consist of the molecule and transition e.g. line("CO", "1-0")')
            
            # For molecular transitions, 'element' is the molecule (e.g. "CO") and 'ionisation' is the transition (e.g. "1-0")
            self.element = None
            self.ionisation = None
            self.molecule = molecule
            self.transition = transition
            self.species = self.molecule + self.transition

        # Specify wavelength unit of line: Angstrom or micron
        self.wl_unit = wl_unit

        # Specify a specific line in a multiplet
        self.line_i = line_i

        self.get_line_info(wl_A)

        self.name = self.species + ("{:d}".format(self.wl_nat) if self.wl_nat != -1 else '')
        self.names = [self.species + ("{:d}".format(int(round(wl))) if not np.isnan(wl) else "NaN") for wl in self.wls_nat]
        
        self.set_labels()
    
    def set_line_i(self, line_i=None):
        # Specify a specific line in a multiplet (or reset by leaving line_i None)
        self.line_i = line_i
        self.wls = self.wls_orig

        self.get_line_info(np.mean(self.wls_orig))
        self.set_labels()

    def set_labels(self):
        # Label strings
        # r"$\mathrm{C \, III}$"
        alt_labels = {"Lya": (r"Ly$\mathrm{\alpha}$", r"Lyα", r"Ly \alpha"),
                        "Halpha": (r"H$\mathrm{\alpha}$", r"Hα", r"H \alpha"),
                        "Hbeta": (r"H$\mathrm{\beta}$", r"Hβ", r"H \beta"),
                        "Hgamma": (r"H$\mathrm{\gamma}$", r"Hγ", r"H \gamma"),
                        "Hdelta": (r"H$\mathrm{\delta}$", r"Hδ", r"H \delta"),
                        "Hepsilon": (r"H$\mathrm{\epsilon}$", r"Hε", r"H \epsilon"),
                        "OI+SiII": (r"$\mathrm{O \, I + Si \, II}$", r"O I + Si II", r"O \, I + Si \, II")}

        if self.atomic:
            particle = self.element
            descr = self.ionisation
        else:
            particle = self.molecule
            descr = self.transition

        if self.alt_name:
            self.flabel, self.stlabel, self.smlabel = alt_labels[self.alt_name]
            alt_str = "or {}, ".format(self.alt_name)
        else:
            # Fancy (LaTeX) label, simple text label, and simple label for use in \mathrm{}
            self.flabel = r"$\mathrm{{{} {} \, {} {}}}$".format(self.lpar, particle, descr, self.rpar)
            self.stlabel = r"{}{} {}{}".format(self.lpar, particle, descr, self.rpar)
            self.smlabel = r"{}{} \, {}{}".format(self.lpar, particle, descr, self.rpar)

            # Change alt_name to normal name if None
            self.alt_name = self.name
            alt_str = ''
        
        if self.wl_unit == "Angstrom":
            wl_val = self.wl
            wls_val = self.wls
            texsymb = r"\lambda \, "
            txtsymb = r"λ "
            texunit = r"\AA"
            txtunit = r"Å"
        elif self.wl_unit == "micron":
            wl_val = self.wl / 1e4
            wls_val = tuple(wl / 1e4 for wl in self.wls)
            texsymb = r""
            txtsymb = r""
            texunit = r"μm"
            txtunit = r"μm"
        
        if np.any(np.diff(np.round(self.wls_orig)) <= 1):
            wlf = ".1f"
        else:
            wlf = ".0f"

        # Quasi-extended label for all lines collectively, but only one (average) wavelength (fancy, simple text, and simple for use in \mathrm{})
        self.qflabel = self.flabel + r" ${}{:{wlf}} \, \mathrm{{ {} }}$".format(texsymb, wl_val, texunit, wlf=wlf)
        self.qstlabel = self.stlabel + r" {}{:{wlf}} {}".format(txtsymb, wl_val, txtunit, wlf=wlf)
        self.qsmlabel = self.smlabel + r"\, {}{}{:{wlf}} \, {}".format(txtsymb, r"\, " if txtsymb else r'', wl_val, texunit, wlf=wlf)
        
        # Extended label for all lines collectively (fancy, simple text, and simple for use in \mathrm{})
        self.eflabel = self.flabel + r" ${}".format(texsymb) + r", \, ".join(r"{:{wlf}}".format(wl, wlf=wlf) for wl in wls_val) + r" \, \mathrm{{ {} }}$".format(texunit)
        self.estlabel = self.stlabel + r" {}".format(txtsymb) + r", ".join(r"{:{wlf}}".format(wl, wlf=wlf) for wl in wls_val) + r" {}".format(txtunit)
        self.esmlabel = self.smlabel + r"\, {}{}".format(txtsymb, r"\, " if txtsymb else r'') + r", \, ".join(r"{:{wlf}}".format(wl, wlf=wlf) for wl in wls_val) + r" \, {}".format(texunit)
        
        # Extended labels for each line individually (fancy, simple text, and simple for use in \mathrm{})
        self.eflabels = [r"$\mathrm{{{} {} \, {} {}}}$".format('[' if self.forb_line[wli] == 'Y' else '',
                            particle, descr, '' if self.forb_line[wli] == 'N' else ']') + \
                            r"$\, {}{:{wlf}} \, \mathrm{{ {} }}$".format(texsymb, wl, texunit, wlf=wlf) for wli, wl in enumerate(wls_val)]
        self.estlabels = [r"{}{} {}{}".format('[' if self.forb_line[wli] == 'Y' else '',
                            particle, descr, '' if self.forb_line[wli] == 'N' else ']') + \
                            r" λ {:{wlf}} {}".format(wl, txtunit, wlf=wlf) for wli, wl in enumerate(wls_val)]
        self.esmlabels = [r"{}{} \, {}{}".format('[' if self.forb_line[wli] == 'Y' else '',
                            particle, descr, '' if self.forb_line[wli] == 'N' else ']') + \
                            r" \, λ \, {:{wlf}} \, {}".format(wl, texunit, wlf=wlf) for wli, wl in enumerate(wls_val)]

        self.name_str = self.lpar + particle + ' ' + descr + self.rpar + ' ' + \
                                ", ".join("{:.1f}".format(wl) for wl in wls_val) + " {}".format(txtunit) + \
                                " ({}{} {} {} line, {})".format(alt_str, self.wlr, self.origin, kind_dict.get(self.kind, self.kind), self.n_wls_str)
    
    def get_line_info(self, wl_A=None):
        # List of Species, Wavelength(s), Origin, Kind, Alternative names, Color, Forbidden line?
        species_list = lines_dict.get(self.species, [((np.nan,), "N/A", "N/A", None, 'k', ('N',))])
        n_lines = len(species_list)
        
        if n_lines == 1:
            # Make sure there is only one line known for this species
            line_idx = 0
        elif n_lines > 1:
            if wl_A is None:
                raise SystemError("multiple lines for " + self.species + " known (at wavelengths of " + \
                                    ', '.join("{}".format(lines_wls) for lines_wls in [slist[0] for slist in species_list]) + " Angstroms)")

            lines_wls = np.array([np.mean(slist[0]) for slist in species_list])
            line_idx = np.argmin(np.abs(lines_wls - wl_A))

        slist = species_list[line_idx]

        multiplicity_str = {1: "singlet", 2: "doublet", 3: "triplet"} #, 4: "quartet", 5: "quintet"

        # Wavelength properties
        self.wls = slist[0]
        self.wls_orig = self.wls
        
        if self.line_i is not None:
            self.wls = (self.wls[self.line_i],) if isinstance(self.line_i, int) else tuple(self.wls[line_i] for line_i in self.line_i)
        
        self.n_wls = len(self.wls)
        self.n_wls_str = multiplicity_str.get(self.n_wls, "multiplet")
        self.wl = np.mean(self.wls)
        self.wl_A = -1 if np.isnan(self.wl) else int(round(self.wl))

        # Frequency (in Hz: speed of light converted from m/s to Angstrom/s)
        self.nu = 299792458.0 * 1e10 / self.wl
        
        # Wavelength in alternative units
        if self.wl_unit == "Angstrom":
            self.wl_nat = self.wl_A
            self.wls_nat = self.wls
        elif self.wl_unit == "micron":
            self.wl_nat = -1 if np.isnan(self.wl) else int(round(self.wl/1e4))
            self.wls_nat = tuple(wl / 1e4 for wl in self.wls)

        # Wavelength range
        if self.wl >= 100.0 and self.wl < 4000.0:
            self.wlr = "UV"
        elif self.wl < 7500.0:
            self.wlr = "optical"
        elif self.wl < 2.5e4: # 25 000 Angstrom is 2.5 micron
            self.wlr = "NIR"
        elif self.wl < 10e4: # 10 micron
            self.wlr = "MIR"
        elif self.wl < 1e7: # 10 000 000 Angstrom is 1000 micron or 1 mm
            self.wlr = "FIR"
        elif self.wl < 1e9: # 1 000 000 000 Angstrom is 100 000 micron or 100 mm or 10 cm
            self.wlr = "microwave"
        elif self.wl >= 1e9:
            self.wlr = "radio"
        else:
            self.wlr = "N/A"

        # Other properties
        self.origin = slist[1]
        self.kind = slist[2]
        self.alt_name = slist[3]
        self.colour = slist[4]
        self.forb_line = np.array(slist[5])
        if self.line_i is not None:
            self.forb_line = np.array([self.forb_line[self.line_i]] if isinstance(self.line_i, int) else [self.forb_line[line_i] for line_i in self.line_i])

        if np.all(self.forb_line == 'Y'):
            self.lpar = '['
            self.rpar = ']'
        elif np.all(self.forb_line == 'S'):
            self.lpar = ''
            self.rpar = ']'
        else:
            self.lpar = ''
            self.rpar = ''

    def __repr__(self):
        if self.atomic:
            return "<{}{} {}{} {} {} {} {} object at {:d} {}>".format(self.lpar, self.element, self.ionisation, self.rpar,
                    self.wlr, self.origin, kind_dict.get(self.kind, self.kind), self.n_wls_str, self.wl_nat, self.wl_unit)
        else:
            return "<{}{} {}{} {} {} {} {} object at {:d} {}>".format(self.lpar, self.molecule, self.transition, self.rpar,
                    self.wlr, self.origin, kind_dict.get(self.kind, self.kind), self.n_wls_str, self.wl_nat, self.wl_unit)

    def __str__(self):
        return self.name_str

class line_list(line):

    def __init__(self, line_list):
        self.list = line_list
        if any([not isinstance(l, line) for l in self.list]):
            raise TypeError("one or more objects are not line objects")
        
        # Use the dictionary of a newly created line object, because self doesn't have line attributes yet
        for key in line().__dict__.keys():
            if key != "list":
                self.__dict__[key] = [l.__dict__[key] for l in self.list]



sct_ldict = {
                "Lya": ("HI", 1216),
                "aLya": ("HI", 1216),
                "CIII": ("CIII", 1909),
                "OIII": ("OIII", 1666),
                "HeII": ("HeII", 1640),
                "CIV": ("CIV", 1548),
                "SiIII": ("SiIII", 1887),
                "NV": ("NV", 1241),
                "sCIII": ("CIII", 1176),
                "sOIV": ("OIV", 1343),
                "sSV": ("SV", 1502),
                "sSiIIa": ("SiII", 1260),
                "sSiIIb": ("SiII", 1303),
                "sSiIIc": ("SiII", 1527),
                "sCII": ("CII", 1335),
                "OIII_opt": ("OIII", 4967),
                "OII": ("OII", 3728),
                "Halpha": ("HI", 6565),
                "Hbeta": ("HI", 4863),
                "NII": ("NII", 6568),
                "SII": ("SII", 6725)
            }

if __name__ == "__main__":
    ll = line_list([line(*args) for args in all_lines_lib])
    print("Lines in library (arguments and line info):\n", "Ion\tWavelength (Å)\tFrequency (GHz)\tInfo",
            *["{}\t{}\t\t{}\t\t{}".format(args[0], args[1], "{:.1f}".format(float(line(args[0], args[1]).nu/1e9)) if args[1] > 1e4 else '',
                                            line(args[0], args[1])) for args in all_lines_lib], sep='\n')
    print("\nLines in library (name and colour):\n", "Ion\tWavelength (Å)\tColour",
            *["{}\t{}\t\t#{:02x}{:02x}{:02x}".format(*args, *[int(c*255) for c in line(*args).colour]) for args in all_lines_lib], sep='\n')
    print("Examples of line objects:", line("CIII", 1908), line("OII", 3728), line("SiIII", 1887), line(), '', sep='\n')