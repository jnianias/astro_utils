import numpy as np

# Fundamental constants used extensively
c = 299792.458  # speed of light in km/s
rwlya = 1215.67 # The rest wavelength of Lyman alpha

# Some common absorption lines
abslines = [
    'SiII1260',
    'CII1334',
    'SiIV1394',
    'SiIV1403',
]

# Some common emission lines
emlines = [
    'CIV1548',
    'CIV1551',
    'HeII1640',
    'OIII1660',
    'OIII1666',
    'CIII1907',
    'CIII1909'
]

# Doublets
doublets = {
    'OVI1032': ('OVI1032', 'OVI1038'),
    'NV1238': ('NV1238','NV1243'),
    'SiIV1394': ('SiIV1394','SiIV1403'),
    'NIV1483': ('NIV1483','NIV1487'),
    'CIV1548': ('CIV1548','CIV1551'),
    'FeII1608': ('FeII1608','FeII1611'),
    'OIII1660': ('OIII1660','OIII1666'),
    'AlIII1854': ('AlIII1854','AlIII1862'),
    'SiIII1883': ('SiIII1883','SiIII1892'),
    'CIII1907': ('CIII1907','CIII1909')
}

# Compute first and second lines of each doublet
flines = np.array([x[0] for x in list(doublets.values())])
slines = np.array([x[1] for x in list(doublets.values())])

# Master wavelength dictionary
wavedict = {
    'AlII1671': 1670.79,
    'AlIII1854': 1854.1,
    'AlIII1862': 1862.17,
    'CII1334': 1334.53,
    'CII2324': 2324.21,
    'CII2326': 2326.11,
    'CII2328': 2327.64,
    'CII2329': 2328.84,
    'CIII1907': 1906.68,
    'CIII1909': 1908.73,
    'CIV1548': 1548.2,
    'CIV1551': 1550.77,
    'FeII1608': 1608.45,
    'FeII1611': 1611.2,
    'HeII1640': 1640.42,
    'NIII1750': 1749.67,
    'NIV1483': 1483.32,
    'NIV1487': 1486.5,
    'NV1238': 1238.82,
    'NV1243': 1242.8,
    'OI1302': 1302.17,
    'OIII1660': 1660.81,
    'OIII1666': 1666.15,
    'OVI1032': 1031.91,
    'OVI1038': 1037.61,
    'SiII1260': 1260.42,
    'SiII1304': 1304.37,
    'SiIII1883': 1882.71,
    'SiIII1892': 1892.03,
    'SiIV1394': 1393.76,
    'SiIV1403': 1402.77,
    'FeII2344': 2344.21,
    'FeII2374': 2374.46,
    'FeII2383': 2382.76,
    'LYALPHA': 1215.67,
    'SiII1527': 1526.71,
    'DUST': 2175.0
}

skylines = {
    'Raman OVI': 6825.4,
    'OI6300': 6300.304,
    'OI5577': 5577.339,
}