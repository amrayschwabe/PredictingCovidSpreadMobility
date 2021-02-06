import json

import joblib
from constants import DATA_PATH


class EMConfig:
    """Class to hold configurations for the EMHawkes configuration"""

    def __init__(self, keyword):
        myConfig = joblib.load(f"{DATA_PATH}configs/{keyword}")
        self.keyword = keyword
        self.mobilityCovariatesNames = myConfig["mobilityCovariatesNames"]
        self.staticCovariatesNames = myConfig["staticCovariatesNames"]
        self.regionalDummy = myConfig["regionalDummy"]
        self.regionalDummyNames = myConfig["regionalDummyNames"]
        self.covariatesNames = myConfig["covariatesNames"]
        self.shiftMobility = myConfig["shiftMobility"]
        self.boundaryCorrect = myConfig["boundaryCorrect"]
        self.maxKernelShape = myConfig["maxKernelShape"]
        self.maxKernelScale = myConfig["maxKernelScale"]
        self.simulationRepetitions = myConfig["simulationRepetitions"]
        self.alphaFixed = myConfig["alphaFixed"]
        self.betaFixed = myConfig["betaFixed"]
        self.muSameForAllRegions = myConfig["muSameForAllRegions"]
        self.fitRegionsSeparately = myConfig["fitRegionsSeperately"]
        self.predMode = myConfig["predMode"]
        self.errorMetrics = myConfig["errorMetrics"]
        self.correctSickPeople = myConfig["correctSickPeople"]
        self.config = myConfig
        self.regressionMethod = myConfig.get("regressionMethod", "poisson")
        self.regularizationPenalty = myConfig.get("lassoPenalty", None)
        self.correctSickPeopleAlpha = myConfig.get("correctSickPeopleAlpha", 1)
        self.muRegression = myConfig.get("muRegression", False)
        self.muZero = myConfig.get("muZero", False)
        self.incubationDistribution = myConfig.get("incubationDistribution", "normal")

    def print(self):
        print(json.dumps(self.config, sort_keys=True, indent=4))


def createConfig(
        future_past: str = "future", incubationDistribution: str = "incubationDistribution"
) -> None:
    configDict = {}
    mobilityCovariatesNames = [
        # place names of mobility covariate columns in your df here as a list of strings. For mobility covariates,
        # reduction wrt to baseline period will be calculated
    ]
    staticCovariatesNames = [
        # place names of static covariate columns in your df here as a list of strings.
    ]
    # needed if every region is estiamted as having seperate spreading dynamics
    regionalDummy = False
    regionalDummyNames = []
    if regionalDummy:
        regionalDummyNames = [
            #list dummy names here as a list of strings
        ]
    covariatesNames = (
            mobilityCovariatesNames + staticCovariatesNames + regionalDummyNames
    )
    configDict["shiftMobility"] = 0
    configDict["boundaryCorrect"] = 14
    configDict["maxKernelShape"] = 15
    configDict["maxKernelScale"] = 15
    configDict["mobilityCovariatesNames"] = mobilityCovariatesNames
    configDict["staticCovariatesNames"] = staticCovariatesNames
    configDict["regionalDummy"] = regionalDummy
    configDict["covariatesNames"] = covariatesNames
    configDict["simulationRepetitions"] = 100
    configDict["alphaFixed"] = 1.2
    configDict["betaFixed"] = 5.7
    configDict["muSameForAllRegions"] = True
    configDict["fitRegionsSeperately"] = False
    configDict["predMode"] = future_past
    configDict["errorMetrics"] = [
        "RMSE",
        "RRMSE",
        "R2",
        "MPE",
        "MAE",
        "MAPE",
        "MDAE",
        "ME",
    ]
    configDict["lassoPenalty"] = 1
    configDict["correctSickPeople"] = True
    configDict["regionalDummyNames"] = regionalDummyNames
    configDict["regressionMethod"] = "poissonLasso"
    configDict["correctSickPeopleAlpha"] = 0.25
    configDict["muRegression"] = False
    configDict["muZero"] = True
    configDict["incubationDistribution"] = incubationDistribution
    configKeyword = "googleMobilityAbsoluteMaximumTemperatureDemo_correct_0_25_Lasso"
    if future_past == "past":
        configKeyword = configKeyword + "_past"
    joblib.dump(configDict, DATA_PATH + "configs/config_" + configKeyword)
    print(f'"config_{configKeyword}",')


if __name__ == "__main__":
    createConfig("future", "gamma")
    createConfig("past", "gamma")
