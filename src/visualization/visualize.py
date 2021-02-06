from typing import Iterable, Optional, List
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from scipy.stats import norm
from constants import (
    nrRegions,
    colors,
    ROOT_DIR,
    mainColorLight,
    mainColorOpaque,
)
from src.utility import convertAllToDataFrames


def plotMu(mus, cantonPop):
    sns.reset_defaults()
    sns.set(rc={"figure.figsize": (7, 5)}, style="white")  # nicer layout
    ax = sns.histplot(mus, kde=False)
    ax.set(xlabel="mu", ylabel="count", title="Mus for all Cantons")
    sns.despine()

    ax = sns.scatterplot(x=mus, y=cantonPop)
    ax.set(xlabel="mu", ylabel="canton population", title="Canton Population vs Mu")
    sns.despine()
    plt.show()


def plotR0(R0: np.ndarray, cantonNames: list, ax=None):
    sns.reset_defaults()
    sns.set(rc={"figure.figsize": (7, 5)}, style="white")  # nicer layout
    if ax is not None:
        ax.set(ylim=(0, 5))
        sns.lineplot(data=R0.T, ax=ax, legend=None, dashes=False, palette=colors)
    else:
        ax = sns.lineplot(data=R0.T, palette=colors, dashes=False)
        ax.set(ylim=(0, 5))
        ax.legend(
            cantonNames,
            frameon=False,
            bbox_to_anchor=(1.0, 1),
            loc="upper left",
            fontsize="xx-small",
        )
    ax.set(xlabel="day", ylabel="R0", title="R0 over time per Canton")
    ax.axhline(1, color="grey", dashes=[6, 2])
    sns.despine()


def plotTrueCasesSmooth(
        casesKnown: np.ndarray, casesTruth: Optional[np.ndarray], ax, color: str
) -> None:
    if casesTruth is not None:
        casesTruth = np.concatenate(([casesKnown[len(casesKnown) - 1]], casesTruth))
        sns.lineplot(
            x=list(
                range(
                    casesKnown.shape[0] - 1,
                    len(casesTruth) + casesKnown.shape[0] - 1,
                )
            ),
            y=casesTruth,
            color=color,
            ax=ax,
            linewidth=4,
        )


def plotSimulationRuns(
        casesProjected: np.ndarray, casesKnown: np.ndarray, ax, color: str
) -> None:
    for row in casesProjected:
        row = np.concatenate(([casesKnown[len(casesKnown) - 1]], row))
        sns.lineplot(
            x=list(
                range(
                    casesKnown.shape[0] - 1,
                    row.shape[0] + casesKnown.shape[0] - 1,
                )
            ),
            y=row,
            ax=ax,
            color=color,
            linewidth=4,
        )


def addSimErrorTextBox(simError: dict, ax) -> None:
    if simError is not None:
        errorString = ""
        for key, value in simError.items():
            errorString = errorString + f"{key}: {value[0]:.2f}\n"
        t = ax.text(
            0.05,
            0.95,
            errorString,
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="top",
        )
        t.set_bbox(dict(facecolor="whitesmoke", alpha=0.8, edgecolor="white"))


def fittedUntilHereLine(xPosition: int, yPosition: int, ax) -> None:
    ax.text(
        xPosition - 1,
        yPosition,
        "Start",
        color="darkgrey",
        horizontalalignment="right",
        size=20,
    )
    ax.axvline(xPosition, 0, 1, color="darkgrey")


def plotProjected(
        casesProjected: np.ndarray,
        casesKnown: np.ndarray,
        xLabel: str,
        yLabel: str,
        title: str,
        casesTruth: Optional[np.ndarray] = None,
        ax=None,
        simError: dict = None,
        plotTrue=True,
):
    if plotTrue:
        if ax is None:
            ax = sns.lineplot(
                x=list(range(casesKnown.shape[0])),
                y=casesKnown,
                color=sns.color_palette("Greys", 4)[1],
                linewidth=4,
            )
            ax.lines[0].set_label(s=None)
        else:
            sns.lineplot(
                x=list(range(casesKnown.shape[0])),
                y=casesKnown,
                color=sns.color_palette("Greys", 4)[1],
                ax=ax,
                linewidth=4,
            )
            ax.lines[0].set_label(s=None)
        plotTrueCasesSmooth(
            casesKnown, casesTruth, ax, sns.color_palette("Greys", 4)[1]
        )
    if len(casesProjected.shape) < 2:
        casesProjected = np.asarray(casesProjected).reshape(
            (1, casesProjected.shape[0])
        )
    plotSimulationRuns(casesProjected, casesKnown, ax, mainColorOpaque)
    ax.set(xlabel=xLabel, ylabel=yLabel)
    ax.set_title(label=title, fontsize=22)
    ax.set_xlabel(ax.get_xlabel(), fontsize=22)
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)
    ax.tick_params(labelsize=22)
    addSimErrorTextBox(simError, ax)
    sns.despine()


def ilocOrNone(myObject: Optional[pd.DataFrame], index) -> Optional[Series]:
    if myObject is None:
        return None
    else:
        return myObject.iloc[index]


def plotAllCantons(
        simResults: np.ndarray,
        covidTrain: Iterable,
        covidTest: Iterable,
        colorPalette: str = "OrRd",
        simErrors: Optional[pd.DataFrame] = None,
        plotKeyword: str = "",
):
    sns.set_palette(sns.color_palette(colorPalette))
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(int(np.ceil(nrRegions / 4)), 4, figsize=(20, 20))
    if int(np.ceil(nrRegions / 4)) * 4 != nrRegions:
        for i in range(int(np.ceil(nrRegions / 4)) * 4 - nrRegions):
            fig.delaxes(axs[int(np.ceil(nrRegions / 4)) - 1, 4 - i - 1])
    covidTrain, covidTest = convertAllToDataFrames(covidTrain, covidTest)
    if len(simResults.shape) == 2:
        simResults = simResults.reshape((simResults.shape[0], 1, simResults.shape[1]))
    for ax, i in zip(axs.flat, range(nrRegions)):
        plotProjected(
            simResults[i],
            covidTrain.iloc[i],
            "days",
            covidTrain.index[i],
            "",
            covidTest.iloc[i],
            ax=ax,
            simError=ilocOrNone(simErrors, [i]),
        )
        legend = fig.legend(
            labels=[
                "real cases",
                "predicted cases",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=4,
            fontsize=22,
        )
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(
        f"{ROOT_DIR}/notebooks/figures/plotAllCantons{plotKeyword}",
        dpi=300,
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )
    plt.show()


def plotKernel(alpha, beta):
    xpoints = np.linspace(0, 20, 500)
    p = norm.pdf(xpoints, loc=beta, scale=alpha)

    plt.plot(xpoints, p, "b", linewidth=1)
    plt.fill_between(xpoints, xpoints * 0, p, facecolor="b", alpha=0.1)
    plt.ylabel("Probability Density")
    sns.despine()
    plt.show()


def plotCovariates(
        covariatesTrain,
        covariatesTest,
        nrTrainingDays,
        nrTestDays,
        cantonNames,
        nrRegions=nrRegions,
):
    covarAll = pd.concat([covariatesTrain, covariatesTest])
    for col in covarAll.iteritems():
        ax = sns.lineplot(
            data=pd.DataFrame(
                np.reshape(
                    [col[1]], (nrRegions, nrTrainingDays + nrTestDays), order="F"
                )
            ).T,
            palette=colors[:nrRegions],
            dashes=False,
        )
        ax.set(title=col[0])
        ax.legend(
            cantonNames,
            frameon=False,
            bbox_to_anchor=(1.0, 1),
            loc="upper left",
            fontsize="xx-small",
        )
        fittedUntilHereLine(nrTrainingDays - 1, 0.8 * max(col[1]), plt)
        plt.show()
