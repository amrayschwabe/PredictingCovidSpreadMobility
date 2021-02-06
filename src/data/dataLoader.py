import copy
from dataclasses import dataclass
from typing import Tuple, Optional

import pandas as pd
from pandas import Index

from constants import (
    nrRegions,
    covidPath,
    diseaseNameInInitialDF,
    regionNameInInitialDF,
    dateNameInInitialDF,
    movementPath,
    DATA_PATH, startComparisonPeriod, endComparisonPeriod,
)
import numpy as np

from src.createConfigFile import EMConfig
from src.utility import normalize, extractAndPivot, extractDataForStartDates

RECALCULATE_CORRECTION = False


@dataclass
class HawkesData:
    """Class to hold all data needed for the EM Algorithm"""

    casesTrain: pd.DataFrame
    casesTrainSpreading: pd.DataFrame
    casesTest: pd.DataFrame
    casesTestSpreading: pd.DataFrame
    covariatesTrain: pd.DataFrame
    covariatesTest: pd.DataFrame
    dataAll: pd.DataFrame


def splitTrainTestRegion(
        data: pd.DataFrame,
        nrRegions: int,
        indexLeaveOutRegion: int,
        columnsDataFrame: Index,
) -> [pd.DataFrame, pd.DataFrame]:
    dataTrain = pd.DataFrame(columns=columnsDataFrame, dtype=int)
    dataTest = pd.DataFrame(columns=columnsDataFrame, dtype=int)
    for index, (_, caseRow) in enumerate(data.iterrows()):
        if index % int(nrRegions) == indexLeaveOutRegion:
            dataTest = dataTest.append(caseRow)
        else:
            dataTrain = dataTrain.append(caseRow)
    return dataTrain, dataTest


def loadDataLeaveOneCantonOut(
        nrDays: int, stateToLeaveOut: int, config: EMConfig
) -> HawkesData:
    assert (
            stateToLeaveOut < nrRegions
    ), "can't exclude state that is larger than number of regions"
    data = loadData(nrDays, 0, config)
    casesTrain, casesTest = splitTrainTestRegion(
        data.casesTrain, nrRegions, stateToLeaveOut, data.casesTrain.columns
    )
    covariatesTrain, covariatesTest = splitTrainTestRegion(
        data.covariatesTrain, nrRegions, stateToLeaveOut, data.covariatesTrain.columns
    )
    if config.correctSickPeople:
        casesTrainSpreading, casesTestSpreading = splitTrainTestRegion(
            data.casesTrainSpreading,
            nrRegions,
            stateToLeaveOut,
            data.casesTrain.columns,
        )
    else:
        casesTrainSpreading = casesTrain
        casesTestSpreading = casesTest
    return HawkesData(
        covariatesTrain=covariatesTrain,
        covariatesTest=covariatesTest,
        casesTrain=casesTrain,
        casesTest=casesTest,
        casesTrainSpreading=casesTrainSpreading,
        casesTestSpreading=casesTestSpreading,
        dataAll=data.dataAll,
    )


def loadData(nrTrainingDays: int, nrTestDays: int, config: EMConfig) -> HawkesData:
    # read in data
    dataAll = pd.read_csv(covidPath)
    covariatesTrain, covariatesTest = loadCovariatesData(
        dataAll, nrTrainingDays, nrTestDays, config
    )
    casesTrain, casesTest, casesTrainSpreading, casesTestSpreading = loadCaseData(
        dataAll, nrTrainingDays, nrTestDays, config
    )
    return HawkesData(
        covariatesTrain=covariatesTrain,
        covariatesTest=covariatesTest,
        casesTrain=casesTrain,
        casesTest=casesTest,
        casesTrainSpreading=casesTrainSpreading,
        casesTestSpreading=casesTestSpreading,
        dataAll=dataAll,
    )


def correctOneCanton(
        cantonName: str,
        caseData: pd.DataFrame,
        movement: pd.DataFrame,
        cantonPop: pd.DataFrame,
        config: EMConfig,
) -> pd.DataFrame:
    inflowCanton = movement[movement["canton_name"] == cantonName]
    for index, row in inflowCanton.iterrows():
        if index != cantonName:
            for date in caseData.columns:
                value = row[date]
                casesInfectedInOriginCanton = caseData.loc[index, date]
                percentageInfectedOfCantonPopulation = (
                        casesInfectedInOriginCanton / cantonPop.loc[index, "canton_pop"]
                )
                additionalCases = config.correctSickPeopleAlpha * np.round(
                    percentageInfectedOfCantonPopulation * value
                )
                caseData.loc[cantonName, date] += additionalCases
    return caseData


def correctForMovement(
        caseData: pd.DataFrame,
        movement: pd.DataFrame,
        cantonPop: Optional[pd.DataFrame],
        config: EMConfig,
        cantonName: str = None,
) -> pd.DataFrame:
    caseData = copy.deepcopy(caseData)
    if cantonPop is None:
        cantonPop = (
            pd.read_csv(covidPath)
                .loc[:, ["canton_codes", "canton_pop"]]
                .drop_duplicates()
        )
        cantonPop = cantonPop.set_index("canton_codes")
    if cantonName is not None:
        caseData = correctOneCanton(cantonName, caseData, movement, cantonPop, config)
        return caseData.loc[[cantonName], :]
    for cantonName in movement["canton_name"].unique():
        caseData = correctOneCanton(cantonName, caseData, movement, cantonPop, config)
    return caseData


def recalculateCorrection(
        caseDataCut: pd.DataFrame, nrTrainingDays: int, nrTestDays: int, config: EMConfig
) -> [pd.DataFrame, pd.DataFrame]:
    movement = pd.read_csv(movementPath, index_col="Unnamed: 0")
    caseDataSpreading = correctForMovement(
        caseData=caseDataCut, movement=movement, cantonPop=None, config=config
    )
    caseDataSpreading.to_csv(
        DATA_PATH
        + f"processed/covid_mobile_data_cleaned_corrected_alpha_{config.correctSickPeopleAlpha}.csv"
    )
    covidTrainSpreading = caseDataSpreading.iloc[:, 0:nrTrainingDays]
    covidTestSpreading = caseDataSpreading.iloc[
                         :, nrTrainingDays: nrTrainingDays + nrTestDays
                         ]
    return covidTrainSpreading, covidTestSpreading


def loadCaseData(
        dataAll: pd.DataFrame, nrTrainingDays: int, nrTestDays: int, config: EMConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    caseData = extractAndPivot(
        dataAll,
        (diseaseNameInInitialDF, regionNameInInitialDF, dateNameInInitialDF),
        regionNameInInitialDF,
        dateNameInInitialDF,
    )
    caseDataCut = extractDataForStartDates(caseData, nrTrainingDays, nrTestDays)
    covidTrain = caseDataCut.iloc[:, 0:nrTrainingDays]
    covidTest = caseDataCut.iloc[:, nrTrainingDays: nrTrainingDays + nrTestDays]
    if config.correctSickPeople:
        if RECALCULATE_CORRECTION:
            covidTrainSpreading, covidTestSpreading = recalculateCorrection(
                caseDataCut, nrTrainingDays, nrTestDays, config
            )
        else:
            try:
                caseDataSpreading = pd.read_csv(
                    DATA_PATH
                    + f"processed/covid_mobile_data_cleaned_corrected_alpha_{config.correctSickPeopleAlpha}.csv",
                    index_col="Unnamed: 0",
                )
                covidTrainSpreading = caseDataSpreading.iloc[:, 0:nrTrainingDays]
                covidTestSpreading = caseDataSpreading.iloc[
                                     :, nrTrainingDays: nrTrainingDays + nrTestDays
                                     ]
            except OSError:
                covidTrainSpreading, covidTestSpreading = recalculateCorrection(
                    caseDataCut, nrTrainingDays, nrTestDays, config
                )
        return covidTrain, covidTest, covidTrainSpreading, covidTestSpreading
    return covidTrain, covidTest, covidTrain, covidTest


def loadCovariatesData(
        dataAll: pd.DataFrame, nrTrainingDays: int, nrTestDays: int, config: EMConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    covariatesCut = loadCovariatesNotSplit(dataAll, nrTrainingDays, nrTestDays, config)

    (
        covariatesTrain,
        covariatesTrainMean,
        covariatesTrainStd,
    ) = extractCovariatesTrainData(covariatesCut, nrTrainingDays)
    covariatesTest = extractCovariatesTestData(
        covariatesCut,
        covariatesTrainMean,
        covariatesTrainStd,
        nrTrainingDays,
        nrTestDays,
    )
    return covariatesTrain, covariatesTest


def extractCovariatesTrainData(
        covariates: pd.DataFrame, nrTrainingDays: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    covariatesTrain = covariates.iloc[: nrRegions * nrTrainingDays, :]
    covariatesTrain, covariatesTrainStd, covariatesTrainMean = normalize(
        covariatesTrain
    )
    return covariatesTrain, covariatesTrainMean, covariatesTrainStd


def extractCovariatesTestData(
        covariates: pd.DataFrame,
        covariatesTrainMean: np.ndarray,
        covariatesTrainStd: np.ndarray,
        nrTrainingDays: int,
        nrTestDays: int,
) -> pd.DataFrame:
    covariatesTest = covariates.iloc[
                     nrRegions * nrTrainingDays: nrRegions * (nrTestDays + nrTrainingDays), :
                     ]
    covariatesTest, _, _ = normalize(
        covariatesTest, covariatesTrainMean, covariatesTrainStd
    )
    return covariatesTest


def prepareMobilityData(
        dataAll: pd.DataFrame,
        dateNameInInitialDF: str,
        regionNameInInitialDF: str,
        covType: str,
        nrTrainingDays: int,
        nrTestDays: int,
        config: EMConfig,
) -> np.ndarray:
    covariatesData = extractAndPivot(
        dataAll,
        (dateNameInInitialDF, regionNameInInitialDF, covType),
        regionNameInInitialDF,
        dateNameInInitialDF,
    )

    # calculate the mean of the two weeks before the first case
    meanJanuary = np.mean(covariatesData.loc[:, startComparisonPeriod:endComparisonPeriod], axis=1)
    covariatesData = covariatesData.divide(meanJanuary, axis=0)
    # smooth and calculate percentage of change
    covariatesData = covariatesData.rolling(7, 1, axis=1).mean()
    covariatesData = pd.DataFrame(-(1 - covariatesData))
    # select only time frame we have full data for, and shift mobility to account for incubation time etc

    covariatesShiftedStart = extractDataForStartDates(
        covariatesData, nrTrainingDays, nrTestDays, config.shiftMobility
    )
    # for EM we need it in nrCantons x nrDays, nrCovariateTypes form
    covariatesShiftedStart = np.reshape(
        [covariatesShiftedStart],
        (np.prod(covariatesShiftedStart.shape), 1),
        order="F",
    )
    return covariatesShiftedStart.flatten()


def prepareStaticCovariatesData(
        dataAll: pd.DataFrame, covType: str, nrTrainingDays: int, nrTestDays: int
) -> np.ndarray:
    covariatesData = extractAndPivot(
        dataAll,
        (dateNameInInitialDF, regionNameInInitialDF, covType),
        regionNameInInitialDF,
        dateNameInInitialDF,
    )

    covariatesShiftedStart = extractDataForStartDates(
        covariatesData, nrTrainingDays, nrTestDays
    )
    # for EM we need it in nrCantons x nrDays, nrCovariateTypes form

    covariatesShiftedStart = np.reshape(
        [covariatesShiftedStart],
        (np.prod(covariatesShiftedStart.shape), 1),
        order="F",
    )
    return covariatesShiftedStart.flatten()


def loadCovariatesNotSplit(
        dataAll: pd.DataFrame, nrTrainingDays: int, nrTestDays: int, config: EMConfig
) -> pd.DataFrame:
    covarAll = pd.DataFrame()
    for covType in config.mobilityCovariatesNames:
        covarAll[covType] = prepareMobilityData(
            dataAll,
            dateNameInInitialDF,
            regionNameInInitialDF,
            covType,
            nrTrainingDays,
            nrTestDays,
            config,
        )
    for covType in config.staticCovariatesNames:
        covarAll[covType] = prepareStaticCovariatesData(
            dataAll, covType, nrTrainingDays, nrTestDays
        )
    if config.regionalDummy:
        for index in range(nrRegions - 1):
            dummyVector = np.zeros((nrRegions, nrTestDays + nrTrainingDays))
            dummyVector[index] = 1
            dummyVector = np.reshape(
                [dummyVector], (np.prod(dummyVector.shape), 1), order="F"
            )
            covarAll[f"dummy{index}"] = dummyVector.flatten()
    return covarAll


if __name__ == "__main__":
    myConfig = EMConfig("XXX")
    loadDataLeaveOneCantonOut(40, 1, myConfig)
