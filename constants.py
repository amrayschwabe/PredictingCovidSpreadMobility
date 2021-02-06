import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = f"{ROOT_DIR}/data/"
covidPath = f"{DATA_PATH}processed/covid_mobile_data_cleaned.csv"
movementPath = f"{DATA_PATH}processed/allCantonsInflowRelevantTime.csv"
mainColor = "#AB1A1A"
mainColorOpaque = "#AB1A1A26"
mainColorLight = "#dd2222BF"
colors = [
    "#BD0000",
    "#FF0000",
    "#FFA200",
    "#FFFF00",
    "#91FF00",
    "#00FF5E",
    "#00FFBC",
    "#00EFFF",
    "#009AFF",
    "#0022FF",
    "#0C1265",
    "#000000",
    "#504E4C",
    "#A4A4A4",
    "#95CCCE",
    "#B0DEFD",
    "#B0FDB0",
    "#FDB0DE",
    "#D9B0FD",
    "#F700FF",
    "#FF009A",
    "#CD00FF",
    "#9100FF",
    "#650C65",
    "#BF6E6E",
    "#FDC7B0",
]
breakDiff = 10 ^ -4
EMIter = 150
GLMIter = 300
maxKernelShape = 15
maxKernelScale = 15
maxR0 = 10
swissPopulation = 8000000
diseaseNameInInitialDF = "bag_falle"
regionNameInInitialDF = "canton_codes"
dateNameInInitialDF = "date"
nrRegions = 26
loggingLevel = 0
simulationRepetitions = 10
numberCores = 8
startDateSwitzerland = "2020-02-24"
regionOffsets = {
    "AG": 1,
    "AI": 18,
    "AR": 9,
    "BE": 3,
    "BL": 2,
    "BS": 2,
    "FR": 5,
    "GE": 2,
    "GL": 12,
    "GR": 2,
    "JU": 2,
    "LU": 8,
    "NE": 6,
    "NW": 13,
    "OW": 15,
    "SG": 8,
    "SH": 15,
    "SO": 10,
    "SZ": 7,
    "TG": 9,
    "TI": 0,
    "UR": 21,
    "VD": 2,
    "VS": 4,
    "ZG": 6,
    "ZH": 2,
}
startComparisonPeriod = "2020-02-10"
endComparisonPeriod = "2020-02-24"