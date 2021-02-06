from typing import Tuple, List
from constants import startDateSwitzerland
import pandas as pd
import numpy as np


def extractDataForStartDates(
        data: pd.DataFrame, nrTrainingDays: int, nrTestDays: int, shift: int = 0
) -> pd.DataFrame:
    dataCut = []
    index = []
    for region in data.index:
        tmp = data.iloc[
              data.index.get_loc(region),
              data.columns.get_loc(startDateSwitzerland)
              - shift: data.columns.get_loc(startDateSwitzerland)
                       + nrTrainingDays
                       + nrTestDays
                       - shift,
              ]
        index.append(region)
        dataCut.append(tmp)
    dataCut = pd.DataFrame(dataCut, index=index)
    assert nrTrainingDays + nrTestDays == dataCut.shape[1]
    return dataCut


def extractAndPivot(
        data: pd.DataFrame, extracedTypes: tuple, index: str, columns: str
) -> pd.DataFrame:
    transformedData = data.loc[:, extracedTypes]
    transformedData = transformedData.pivot(index=index, columns=columns)
    transformedData = transformedData.T.reset_index(level=0, drop=True).T
    return transformedData


def normalize(
        data: pd.DataFrame, dataMean: np.ndarray = None, dataStd: np.ndarray = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    assert (
            dataMean is not None
            and dataStd is not None
            or dataMean is None
            and dataStd is None
    ), (
        "Either mean and "
        "standard deviation "
        "need to be given or"
        " neither of them"
    )
    if dataMean is not None and dataStd is not None:
        data = (data - dataMean) / dataStd
    else:
        dataStd = data.std(axis=0)
        dataMean = data.mean(axis=0)
        data = (data - dataMean) / dataStd
    return data, dataStd, dataMean


def convertAllToDataFrames(*args) -> List[pd.DataFrame]:
    returnargs = []
    for obj in args:
        if not isinstance(obj, pd.DataFrame):
            try:
                returnargs.append(pd.DataFrame(obj))
            except Exception:
                print(f"object {obj} could not be converted to a pandas DataFrame")
        else:
            returnargs.append(obj)
    return returnargs
