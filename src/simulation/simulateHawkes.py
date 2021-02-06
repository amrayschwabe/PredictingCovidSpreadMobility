from typing import Optional, List

from scipy.stats import norm, lognorm, gamma, weibull_min
import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from scipy.optimize import curve_fit
from constants import swissPopulation, nrRegions, covidPath
from src.createConfigFile import EMConfig
from src.data.dataLoader import correctForMovement


def simHawkes(
    mu: np.ndarray,
    alpha: float,
    beta: float,
    R0: np.ndarray,
    nrTrainingDays: int,
    nrTestDays: int,
    covidCounty: np.ndarray,
    config: EMConfig,
    nrRepetitions: int = 1,
    seed: int = 0,
) -> np.ndarray:
    assert (
        covidCounty.shape[0] == nrTrainingDays
    ), "We need as many days with covid cases as training days"
    np.random.seed(seed)
    # first simulate background events, this is done by picking p points where p is poisson with mu*totalDays
    # expected occurences. These events are distributed uniformly in the interval [0, nrTrainingDays + nrTestDays].
    # Additionally, we take the events triggered by known past events into account.
    simResults = np.zeros((nrRepetitions, nrTestDays))
    for r in range(nrRepetitions):
        cases = np.zeros(nrTestDays)
        times = []
        for day in range(nrTestDays):
            # timestamps for calculating the triggered events from know events in the training period.
            # For this, calculate kernel influence from each of the past known days onto current period

            timestamps = day + nrTrainingDays - np.array(range(nrTrainingDays))
            if config.incubationDistribution == "lognormal":
                sigma = 0.418
                mu = 1.621
                intensity = lognorm.cdf(
                    timestamps, s=sigma, scale=np.exp(mu)
                ) - lognorm.cdf(timestamps - 1, s=sigma, scale=np.exp(mu))
            else:
                intensity = norm.cdf(timestamps, scale=alpha, loc=beta) - norm.cdf(
                    timestamps - 1, scale=alpha, loc=beta
                )
            intensity = intensity.reshape(-1, 1) * np.array(
                R0[list(range(0, nrTrainingDays))] * covidCounty
            ).reshape(-1, 1)
            intensity = np.sum(intensity) + mu
            nrTriggeredCases = np.random.poisson(intensity, size=1)[0]
            cases[day] = nrTriggeredCases

            # uniformly sample random times for the cases
            times.extend(day + np.random.random((nrTriggeredCases, 1)))
        count = len(times)
        # now simulate offspring from each of those simulated cases
        i = 0
        while i < count:
            # for every event sample offspring
            nrChildren = int(np.random.poisson(R0[int(times[i]) + nrTrainingDays]))
            # find day when offspring is sick
            if nrChildren > 0:
                infectionTimes = (
                    norm.rvs(size=nrChildren, scale=alpha, loc=beta) + times[i]
                )
                infectionTimes = np.array(
                    list(filter(lambda x: x < nrTestDays, infectionTimes))
                )
                count += len(infectionTimes)
                times.extend(infectionTimes)
                for infectionTime in infectionTimes:
                    cases[int(infectionTime)] += 1
            i += 1
        simResults[r] = cases
    return simResults


def simHawkesOneDay(
    mu: float,
    alpha: float,
    beta: float,
    R0: np.ndarray,
    nrTrainingDays: int,
    day: int,
    cases: np.ndarray,
    config: EMConfig,
    threshold: int = 1e-5,
) -> np.ndarray:
    assert (
        cases.shape[0] >= nrTrainingDays
    ), "The number of cases does not match the number of training days"
    timestamps = nrTrainingDays + day - np.array(range(nrTrainingDays + day))
    if config.incubationDistribution == "weibull":
        intensity = weibull_min.cdf(
            timestamps + 0.5, c=2.453, scale=6.258
        ) - weibull_min.cdf(timestamps - 0.5, c=2.453, scale=6.258)
        intensity[len(intensity) - 1] += weibull_min.cdf(0.5, c=2.453, scale=6.258)
    elif config.incubationDistribution == "gamma":
        intensity = gamma.cdf(timestamps + 0.5, a=5.807, scale=0.948) - gamma.cdf(
            timestamps - 0.5, a=5.807, scale=0.948
        )
        intensity[len(intensity) - 1] += gamma.cdf(0.5, a=5.807, scale=0.948)
    elif config.incubationDistribution == "lognormal":
        sigma = 0.5
        mu = 1.63
        intensity = lognorm.cdf(
            timestamps + 0.5, s=sigma, scale=np.exp(mu)
        ) - lognorm.cdf(timestamps - 0.5, s=sigma, scale=np.exp(mu))
        intensity[len(intensity) - 1] += lognorm.cdf(0.5, scale=np.exp(mu), s=sigma)
    elif config.incubationDistribution == "normal":
        intensity = norm.cdf(timestamps + 0.5, scale=alpha, loc=beta) - norm.cdf(
            timestamps - 0.5, scale=alpha, loc=beta
        )
        intensity[len(intensity) - 1] += norm.cdf(0.5, scale=alpha, loc=beta)
    else:
        raise NotImplementedError
    intensity = intensity[intensity > threshold].reshape(-1, 1)
    kernelRange = list(
        range(nrTrainingDays + day - intensity.shape[0], nrTrainingDays + day)
    )
    intensityDay = intensity * np.array(R0[kernelRange].T * cases[kernelRange]).reshape(
        -1, 1
    )
    intensityDay = np.round(np.sum(intensityDay) + mu)
    # TODO: why here poisson distribution instead of just taking expectation? misschien voor confidence interval
    nrTriggeredCases = np.random.poisson(intensityDay)
    nrTriggeredCases = min(nrTriggeredCases, swissPopulation)
    return nrTriggeredCases


def simHawkesAllCantons(
    covidTrain: pd.DataFrame,
    covidOtherCantons: Optional[pd.DataFrame],
    mus: np.ndarray,
    alphas: list,
    betas: list,
    R0pred: np.ndarray,
    nrTrainingDays: int,
    nrTestDays: int,
    config: EMConfig,
    movement: Optional[pd.DataFrame],
    cantonPop: Optional[pd.DataFrame],
):
    assert (
        nrTrainingDays + nrTestDays < movement.shape[1]
    ), "Our testing time exceeds the number of days for which movement data is available"

    allCasesSpreading = covidTrain
    allCasesInfected = pd.DataFrame()
    for day in range(nrTestDays):
        allCasesDayInfected = []
        for c, (_, row) in enumerate(covidTrain.iterrows()):
            infected = simHawkesOneDay(
                mus[c][nrTrainingDays + day],
                alphas[c],
                betas[c],
                R0pred[c],
                nrTrainingDays,
                day,
                np.asarray(allCasesSpreading)[c],
                config,
            )
            allCasesDayInfected.append(infected)
        # consolidate travel data
        date = movement.columns[allCasesSpreading.shape[1]]
        allCasesDayInfected = pd.DataFrame(
            allCasesDayInfected, index=covidTrain.index.unique(), columns=[str(date)]
        )
        if config.correctSickPeople:
            if covidOtherCantons is not None:
                allCantonsCasesDayInfected = pd.concat(
                    [covidOtherCantons.loc[:, [str(date)]], allCasesDayInfected]
                )
                allCasesDaySpreading = correctForMovement(
                    caseData=allCantonsCasesDayInfected,
                    movement=movement,
                    cantonPop=cantonPop,
                    cantonName=allCasesDayInfected.index[0],
                    config=config,
                )
            else:
                allCasesDaySpreading = correctForMovement(
                    caseData=allCasesDayInfected,
                    movement=movement,
                    cantonPop=cantonPop,
                    cantonName=None,
                    config=config,
                )
        else:
            allCasesDaySpreading = allCasesDayInfected
        allCasesSpreading = pd.concat([allCasesSpreading, allCasesDaySpreading], axis=1)
        allCasesInfected = pd.concat([allCasesInfected, allCasesDayInfected], axis=1)
    return allCasesInfected, allCasesSpreading.iloc[:, covidTrain.shape[1] :]


def simHawkesAllRepetitions(
    covidTrain: pd.DataFrame,
    covidOtherCantons: Optional[pd.DataFrame],
    mus: np.ndarray,
    alphas: list,
    betas: list,
    R0pred: np.ndarray,
    nrTrainingDays: int,
    nrTestDays: int,
    simulationRepetitions: int,
    config: EMConfig,
    movement: Optional[pd.DataFrame],
):
    numberRegions = covidTrain.shape[0]
    simResultsSpreading = np.zeros((numberRegions, simulationRepetitions, nrTestDays))
    simResultsInfected = np.zeros((numberRegions, simulationRepetitions, nrTestDays))
    cantonPop = (
        pd.read_csv(covidPath).loc[:, ["canton_codes", "canton_pop"]].drop_duplicates()
    )
    cantonPop = cantonPop.set_index("canton_codes")
    for r in range(simulationRepetitions):
        simResultsInfected[:, r, :], simResultsSpreading[:, r, :] = simHawkesAllCantons(
            covidTrain,
            covidOtherCantons,
            mus,
            alphas,
            betas,
            R0pred,
            nrTrainingDays,
            nrTestDays,
            config,
            movement,
            cantonPop,
        )
    return simResultsInfected, simResultsSpreading


def extrapolateCovar(
    covariatesTrain: pd.DataFrame, nrExtrapolateDays: int
) -> pd.DataFrame:
    # 3rd polynomial for this data
    def func(myX, a, b):
        return a * myX + b

    # Initial parameter guess, just to kick off the optimization
    guess = (3, 1)
    extrapolatedCovar = pd.DataFrame()
    for i, covarType in covariatesTrain.T.iterrows():
        reshaped = pd.DataFrame(
            np.reshape(
                np.asarray(covarType),
                (nrRegions, int(covariatesTrain.shape[0] / nrRegions)),
                order="F",
            )
        )
        newReshaped = pd.DataFrame()
        # Curve fit each column
        for index, row in reshaped.iterrows():
            # Get x & y
            x = reshaped.columns.astype(float).values
            y = row
            # Curve fit column and get curve parameters
            try:
                params = curve_fit(func, x, y, guess)
                # extend y with as many nan values as we want to extrapolate
                y = pd.concat(
                    [y, pd.DataFrame([np.nan for _ in range(nrExtrapolateDays)])],
                    axis=0,
                )
                y.reset_index(drop=True, inplace=True)
                # Get the index values for NaNs in the column
                x = y[pd.isnull(y)[0]].index.astype(float).values
                # Extrapolate those points with the fitted function
                newReshaped = pd.concat(
                    [newReshaped, pd.DataFrame(func(x, *params[0])).T], axis=0
                )
            except Exception:
                newReshaped = pd.concat(
                    [
                        newReshaped,
                        pd.DataFrame(
                            [row.iloc[-1] for _ in range(nrExtrapolateDays)]
                        ).T,
                    ],
                    axis=0,
                )

        reshaped = pd.DataFrame(
            np.reshape(
                np.asarray(newReshaped), (np.prod(newReshaped.shape), 1), order="F"
            )
        )
        extrapolatedCovar = pd.concat([extrapolatedCovar, reshaped], axis=1)
    return extrapolatedCovar


def predR0(
    covariatesTrain: pd.DataFrame,
    covariatesTest: pd.DataFrame,
    covariatesNames: list,
    poisson_results: list,
    predMode: str = "future",
) -> (np.ndarray, pd.DataFrame):
    covarAll = getAllCovariates(
        predMode=predMode,
        covariatesTrain=covariatesTrain,
        covariatesTest=covariatesTest,
        covariatesNames=covariatesNames,
    )
    R0pred = predR0givenCovar(covarAll, poisson_results)
    return R0pred, covariatesTest


def getAllCovariates(
    predMode: str,
    covariatesTrain: pd.DataFrame,
    covariatesTest: pd.DataFrame,
    covariatesNames: List[str],
) -> pd.DataFrame:
    assert predMode == "future" or predMode == "past" or predMode == "extrapolate", (
        "Pred mode can either be" " future, past or extrapolate"
    )
    if predMode == "future":
        covarAll = np.concatenate((covariatesTrain, covariatesTest))
    elif predMode == "past":
        nrTestDays = int(covariatesTest.shape[0] / nrRegions)
        nrTrainingDays = int(covariatesTrain.shape[0] / nrRegions)
        lastKnownCovar = covariatesTrain[(nrTrainingDays - 1) * nrRegions :]
        covariatesTest = pd.concat([lastKnownCovar] * nrTestDays)
        covariatesTest.columns = covariatesTrain.columns
        covarAll = np.concatenate((covariatesTrain, covariatesTest))
    else:
        covariatesTest = extrapolateCovar(
            covariatesTrain, int(covariatesTest.shape[0] / nrRegions)
        )
        covariatesTest.columns = covariatesTrain.columns
        covarAll = np.concatenate((covariatesTrain, covariatesTest))
    covarAll = pd.DataFrame(covarAll, columns=covariatesNames)
    return covarAll


def predMu(
    covariatesTrain: pd.DataFrame,
    covariatesTest: pd.DataFrame,
    poisson_results: list,
    config: EMConfig,
    daysToPredict: int,
) -> (np.ndarray, pd.DataFrame):
    if config.muRegression:
        covarAll = getAllCovariates(
            predMode=config.predMode,
            covariatesTrain=covariatesTrain,
            covariatesTest=covariatesTest,
            covariatesNames=config.covariatesNames,
        )
        muPred = predXgivenCovar(covarAll, poisson_results)
    else:
        muPred = np.concatenate([poisson_results] * daysToPredict).reshape(
            -1, daysToPredict
        )
    return muPred, covariatesTest


def predXgivenCovar(covarAll: pd.DataFrame, poisson_results: list) -> np.ndarray:
    # predict reproduction rate in the future based on mobility data and learned poisson function
    ypreds = []
    for i in range(nrRegions):
        covarCanton = covarAll[covarAll.index % nrRegions == i]
        poisson_result = poisson_results[i]
        ypred = poisson_result.predict(covarCanton)
        ypreds.append(ypred)
    Xpred = np.asarray(ypreds)
    return Xpred


def predR0givenCovar(covarAll: pd.DataFrame, poisson_results: list) -> np.ndarray:
    R0pred = predXgivenCovar(covarAll=covarAll, poisson_results=poisson_results)
    R0pred = np.clip(R0pred, a_max=5, a_min=0)
    return R0pred


def predictR0givenCovarOneRegion(
    covarAll: pd.DataFrame, poisson_results: GLMResultsWrapper, nrTrainingDays: int
) -> np.ndarray:
    R0pred = predictXgivenCovarOneRegion(covarAll, poisson_results, nrTrainingDays)
    R0pred = np.clip(R0pred, a_max=5, a_min=0)
    return R0pred


def predictXgivenCovarOneRegion(
    covarAll: pd.DataFrame, poisson_results: GLMResultsWrapper, nrTrainingDays: int
) -> np.ndarray:
    ypred = poisson_results.predict(covarAll)
    X = np.reshape([ypred], (1, nrTrainingDays), order="F")
    return X
