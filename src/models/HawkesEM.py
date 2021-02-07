import copy
from typing import List

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from tqdm.auto import tqdm
from scipy.stats import norm, lognorm, gamma, weibull_min
from itertools import combinations
import pandas as pd
import numpy as np
import sys
import statsmodels.formula.api as smf
import statsmodels.api as sm
import time
from constants import loggingLevel, maxR0, nrRegions, regionOffsets

from src.createConfigFile import EMConfig
from src.data.dataLoader import HawkesData

sys.path.append("..")


def updateProbabilities(
        covidValues: list, R0: list, kernelPDF: np.ndarray, mu: List[float]
) -> [np.array, np.array]:
    p = np.asarray(R0) * kernelPDF * np.asarray(covidValues)

    # an event triggering itself is background rate
    muDiag = np.diag(
        np.asarray(mu).reshape(
            -1,
        )
    )
    p = p + muDiag

    # intensity is sum of probabilities of this event happening
    normHelper = np.sum(p, axis=1)

    # normalize probabilities
    # probability that ONE case on day i is caused by ANY case on day j
    p = np.asarray(p) / np.asarray(normHelper)[:, np.newaxis]
    # probability that ONE case on day i is caused by ONE case on day j
    q = np.divide(
        np.asarray(p),
        np.asarray(covidValues),
        out=p.copy(),
        where=np.asarray(covidValues) != 0,
    )
    return p, q


def EMHawkes(
        EMIter: int,
        nrTrainingDays: int,
        data: HawkesData,
        config: EMConfig,
        alpha: float = 2,
        beta: float = 5,
        breakDiff: float = 10e-5,
) -> object:
    # do we want to model the relationship between mobility and cases as the same or differently for each region
    if not config.fitRegionsSeparately:
        alpha, beta, mu, R0, poisson_result = EMHawkesHelper(
            EMIter,
            nrTrainingDays,
            data,
            config,
            alpha,
            beta,
            breakDiff,
        )
        alphas = [alpha for _ in range(nrRegions)]
        betas = [beta for _ in range(nrRegions)]
        mus = mu
        R0s = R0
        poisson_results = [poisson_result for _ in range(nrRegions)]
    else:
        alphas = []
        betas = []
        mus = []
        R0s = []
        poisson_results = []
        for i in range(nrRegions):
            regionData = copy.deepcopy(data)
            regionData.casesTrainSpreading = data.casesTrainSpreading.iloc[[i]]
            regionData.casesTrain = data.casesTrainSpreading.iloc[[i]]
            regionData.covariatesTrain = data.covariatesTrain[
                data.covariatesTrain.index % nrRegions == i
                ]
            alpha, beta, mu, R0, poisson_result = EMHawkesHelper(
                EMIter,
                nrTrainingDays,
                regionData,
                config,
                alpha,
                beta,
                breakDiff,
            )
            alphas.append(alpha)
            betas.append(beta)
            mus.append(mu[0])
            R0s.append(R0[0])
            poisson_results.append(poisson_result)

    return alphas, betas, mus, R0s, poisson_results


def precomputeKernelPDF(
        alpha: float, beta: float, nrTrainingDays: int, config: EMConfig
) -> np.ndarray:
    kernelPDF = np.zeros((nrTrainingDays, nrTrainingDays))
    if config.incubationDistribution == "weibull":
        for i in range(nrTrainingDays):
            for j in range(i):
                if i - j == 1:
                    kernelPDF[i, j] = weibull_min.cdf(
                        i - j + 0.5, c=alpha, scale=beta
                    ) - weibull_min.cdf(i - j - 1, c=alpha, scale=beta)
                else:
                    kernelPDF[i, j] = weibull_min.cdf(
                        i - j + 0.5, c=alpha, scale=beta
                    ) - weibull_min.cdf(i - j - 0.5, c=alpha, scale=beta)
    elif config.incubationDistribution == "gamma":
        for i in range(nrTrainingDays):
            for j in range(i):
                if i - j == 1:
                    kernelPDF[i, j] = gamma.cdf(
                        i - j + 0.5, a=alpha, scale=beta
                    ) - gamma.cdf(i - j - 1, a=alpha, scale=beta)
                else:
                    kernelPDF[i, j] = gamma.cdf(
                        i - j + 0.5, a=alpha, scale=beta
                    ) - gamma.cdf(i - j - 0.5, a=alpha, scale=beta)
    elif config.incubationDistribution == "lognormal":
        for i in range(nrTrainingDays):
            for j in range(i):
                if i - j == 1:
                    kernelPDF[i, j] = lognorm.cdf(
                        i - j + 0.5, s=alpha, scale=beta
                    ) - lognorm.cdf(i - j - 1, s=alpha, scale=beta)
                else:
                    kernelPDF[i, j] = lognorm.cdf(
                        i - j + 0.5, s=alpha, scale=beta
                    ) - lognorm.cdf(i - j - 0.5, s=alpha, scale=beta)
    elif config.incubationDistribution == "normal":
        for i in range(nrTrainingDays):
            for j in range(i):
                if i - j == 1:
                    kernelPDF[i, j] = norm.cdf(
                        i - j + 0.5, scale=alpha, loc=beta
                    ) - norm.cdf(i - j - 1, scale=alpha, loc=beta)
                else:
                    kernelPDF[i, j] = norm.cdf(
                        i - j + 0.5, scale=alpha, loc=beta
                    ) - norm.cdf(i - j - 0.5, scale=alpha, loc=beta)
    else:
        raise NotImplementedError
    return kernelPDF


def EMHawkesHelper(
        EMIter: int,
        nrTrainingDays: int,
        data: HawkesData,
        config: EMConfig,
        alpha: float = 2,
        beta: float = 5,
        breakDiff: float = 10e-3,
):
    assert EMIter > 0, "There must be a non-negative amount of EM iterations"
    assert nrTrainingDays > 0, "There must be a non-negative amount of training days"
    assert data.casesTrain is not None, "There must be covid values"
    assert config.boundaryCorrect >= 0, "The boundary correction cannot be negative"
    assert (
            nrTrainingDays - config.boundaryCorrect >= 0
    ), "There must be days after boundary correction"
    assert (
            config.maxKernelScale >= beta
    ), "The maximum scale (beta) must be larger than the initial value"
    assert alpha >= 0, "Alpha has to be positive"
    assert beta >= 0, "Beta has to be positive"
    assert (
            len(config.covariatesNames) == data.covariatesTrain.shape[1]
    ), "There must be a name for each of the covariates columns"
    assert data.covariatesTrain.shape[0] == np.prod(
        data.casesTrain.shape
    ), "Every covariate must be present for every day and for every region"

    currentRegions = data.casesTrain.shape[0]
    boundaryCorrectedDays = nrTrainingDays - config.boundaryCorrect
    # R0 reproduction number, init to 1
    R0 = np.ones((currentRegions, nrTrainingDays))

    # init p matrix, which denotes the probability that ONE event on day i was triggered by ANY event on day j
    p = []
    for i in range(currentRegions):
        p.append(np.zeros((nrTrainingDays, nrTrainingDays)))

    # init q matrix, which denotes the probability that ONE event on day i was triggered by ONE event on day j
    q = []
    for i in range(currentRegions):
        q.append(np.zeros((nrTrainingDays, nrTrainingDays)))

    # init background rates mus
    mus = np.zeros(currentRegions) + 1

    # get all the possible infections pairs and sort by day the infection breaks out (later day)
    combos = np.asarray(list(combinations(range(0, nrTrainingDays), 2)))
    combos = combos[combos[:, 1].argsort()]
    diffCombos = combos[:, 1] - combos[:, 0]

    differenceDays = np.tile(diffCombos, currentRegions)
    # how many people got covid for the matching frequency
    frequencyDays = np.zeros(combos.shape[0] * currentRegions)

    # formula for the poisson regression, we just regress Y (reproduction number) against all covariates
    glmString = "Y ~"
    for el in config.covariatesNames:
        glmString = glmString + " + " + el

    # init values
    alphaDelta = []
    betaDelta = []
    R0Delta = []

    alphaPrev = None
    betaPrev = None
    R0Prev = None

    pbar = tqdm(total=EMIter)

    # do we know our kernel shape, if not we estimate
    if config.alphaFixed and config.betaFixed:
        kernelPDF = precomputeKernelPDF(
            config.alphaFixed, config.betaFixed, nrTrainingDays, config
        )
    else:
        kernelPDF = np.zeros((nrTrainingDays, nrTrainingDays))
    for niter in range(EMIter):
        timeStart = time.time()
        # precompute kernelpdf
        if not (config.alphaFixed and config.betaFixed):
            kernelPDF = precomputeKernelPDF(alpha, beta, nrTrainingDays, config)

        # E-Step, estimate the branching tree of covid spread
        for c in range(currentRegions):
            if sum(data.casesTrainSpreading.iloc[c]) != 0:
                if isinstance(mus[c], float):
                    mu = pd.DataFrame(
                        [
                            mus[c]
                            for _ in range(data.casesTrainSpreading.iloc[c].shape[0])
                        ]
                    )
                else:
                    dataCanton = data.covariatesTrain[
                        np.arange(data.covariatesTrain.shape[0]) % currentRegions == c
                        ]
                    mu = mus[c].predict(dataCanton)
                p[c], q[c] = updateProbabilities(
                    data.casesTrainSpreading.iloc[c], R0[c], kernelPDF, mu
                )

        # M-Step, estimate background rate, kernel shape and reproduction number (R0, alpha, beta, mu)
        Q = []
        for c in range(currentRegions):
            # q without the background rate
            QWithoutBackground = q[c] - np.diag(np.diag(q[c]))

            # the average number of observed children generated by j is the \sum_i q(i,j)*t(i)
            casesCounty = np.array((data.casesTrain.iloc[c]).T)
            averageChildrenPerDay = QWithoutBackground * casesCounty[:, np.newaxis]
            averageObservedChildren = np.sum(averageChildrenPerDay, axis=0)
            Q.append(averageObservedChildren.astype(float))
        Q = pd.DataFrame(Q)

        # weights for observations, which is the number of events at day j
        weights = np.reshape(
            [data.casesTrain.iloc[:, :boundaryCorrectedDays]],
            np.prod(data.casesTrain.iloc[:, :boundaryCorrectedDays].shape),
            order="F",
        )
        weights = weights.astype(int)
        # estimate R0 and the coefficients in Poisson regression (alpha, beta)
        # boundary correct
        glmX = np.array(data.covariatesTrain[: currentRegions * boundaryCorrectedDays])[
            weights > 0
            ]
        glmY = np.array(Q.iloc[:, :boundaryCorrectedDays])
        glmY = np.reshape([glmY], (np.prod(glmY.shape), -1), order="F")[weights > 0]
        weights = weights[weights > 0]
        dataGLM = pd.concat(
            [
                pd.DataFrame(glmX, columns=config.covariatesNames),
                pd.DataFrame(glmY, columns=["Y"]),
            ],
            axis=1,
        )

        dataGLMPred = pd.DataFrame(data.covariatesTrain, columns=config.covariatesNames)
        dataGLMPred.reset_index(inplace=True, drop=True)
        # fit model

        # what regression do we want
        # unregularized poisson regression
        if config.regressionMethod.lower() == "poisson".lower():
            poisson_model = smf.glm(
                formula=glmString,
                data=dataGLM,
                family=sm.families.Poisson(),
                freq_weights=weights,
            )
            regression_results = poisson_model.fit()
            ypred = regression_results.predict(dataGLMPred)
        # lasso regularized poisson regression
        elif config.regressionMethod.lower() == "poissonLasso".lower():
            poisson_model = smf.glm(
                formula=glmString,
                data=dataGLM,
                family=sm.families.Poisson(),
                freq_weights=weights,
            )
            regression_results = poisson_model.fit_regularized(
                maxiter=200, alpha=config.regularizationPenalty
            )
            ypred = regression_results.predict(dataGLMPred)
        # ridge regularized poisson regression
        elif config.regressionMethod.lower() == "poissonRidge".lower():
            poisson_model = smf.glm(
                formula=glmString,
                data=dataGLM,
                family=sm.families.Poisson(),
                freq_weights=weights,
            )
            regression_results = poisson_model.fit_regularized(
                maxiter=200, alpha=config.regularizationPenalty, L1_wt=0
            )
            ypred = regression_results.predict(dataGLMPred)
        # gaussian process regression
        elif config.regressionMethod.lower() == "gaussianProcess".lower():
            kernel = WhiteKernel() + Matern(length_scale=0.5)
            regression_results = GaussianProcessRegressor(kernel=kernel).fit(glmX, glmY)
            ypred = regression_results.predict(dataGLMPred)
        else:
            raise NotImplementedError

        R0 = np.reshape([ypred], (currentRegions, nrTrainingDays), order="F")
        R0 = np.clip(R0, a_min=0, a_max=maxR0)

        # estimate the background rate mu
        mus = []
        # if we do not estimate mu to be stable, we can also regress it from mobility covariates
        if config.muRegression:
            backgroundCases = []
            for c in range(currentRegions):
                backgroundCases.append(np.diag(p[c]).T * data.casesTrain.iloc[c, :])
            backgroundCases = np.reshape(
                [np.asarray(backgroundCases)],
                (data.covariatesTrain.shape[0], 1),
                order="F",
            )
            dataGLMMu = pd.concat(
                [
                    pd.DataFrame(data.covariatesTrain, columns=config.covariatesNames),
                    pd.DataFrame(backgroundCases, columns=["Y"]),
                ],
                axis=1,
            )
            poisson_model = smf.glm(
                formula=glmString,
                data=dataGLMMu,
                family=sm.families.Poisson(),
            )
            muRegression = poisson_model.fit_regularized(
                maxiter=100, alpha=config.regularizationPenalty
            )
            for c in range(currentRegions):
                mus.append(muRegression)

        # if we model mu as zero, we put it to a very small value (to not have division by zero)
        elif config.muZero:
            mus = [0.0001 for _ in range(currentRegions)]
        # here we assume mu is static over time
        else:
            for c in range(currentRegions):
                nrCovidDays = max(1, nrTrainingDays - list(regionOffsets.values())[c])
                mus.append(
                    np.sum(np.diag(p[c]).T * data.casesTrain.iloc[c, :]) / (nrCovidDays)
                )
            if config.muSameForAllRegions:
                meanMu = np.mean(mus)
                mus = np.array([meanMu for _ in mus])

        if config.alphaFixed is not None and config.betaFixed is not None:
            alpha = config.alphaFixed
            beta = config.betaFixed
        # estimate kernel shape
        else:
            # for every city, add frequency of corona cases with x days difference to overall param estim
            for c in range(currentRegions):
                covidCasesCounty = pd.DataFrame(data.casesTrain.iloc[c])
                prob = p[c]
                freq = (prob[combos[:, 1], combos[:, 0]]).reshape(-1, 1)
                freq = freq * covidCasesCounty.iloc[combos[:, 1]]
                frequencyDays[
                c * combos.shape[0]: (c + 1) * combos.shape[0]
                ] = freq.iloc[:, 0]
            # fit kernel
            frequency = np.bincount(differenceDays, weights=frequencyDays)[1:]
            # expand model according to frequency of observations
            expandedObservations = np.repeat(
                list(range(1, nrTrainingDays)), np.round(frequency).astype(int)
            )
            if len(expandedObservations) > 0:
                beta, alpha = norm.fit(expandedObservations)
                beta = min(beta, config.maxKernelShape)
                alpha = min(alpha, config.maxKernelScale)
                if config.alphaFixed is not None:
                    alpha = config.alphaFixed
                if config.betaFixed is not None:
                    beta = config.betaFixed

        # check for convergence
        if niter == 0:
            alphaPrev = alpha
            betaPrev = beta
            R0Prev = R0
        else:
            alphaDelta.append(np.sqrt(np.square(alpha - alphaPrev)))
            betaDelta.append(np.sqrt(np.square(beta - betaPrev)))
            R0Delta.append(np.sqrt(np.square(R0 - R0Prev)))
            alphaPrev = alpha
            betaPrev = beta
            R0Prev = R0

        # Early Stop
        if niter > 5:
            if (
                    (np.asarray(alphaDelta[-4:]) < breakDiff).all()
                    and (np.asarray(betaDelta[-4:]) < breakDiff).all()
                    and (np.asarray(R0Delta[-4:]) < breakDiff).all()
            ):
                if loggingLevel > 0:
                    print("Convergence criterion met. Break out of EM")
                pbar.update(EMIter - niter)
                pbar.close()
                break
        if loggingLevel > 0:
            print(f"Iteration: {niter}, time used: {time.time() - timeStart}")
            print(f"Alpha: {alpha}, Beta: {beta}, R0: {np.average(np.average(R0))}")
        pbar.update(1)
    return alpha, beta, mus, R0, regression_results
