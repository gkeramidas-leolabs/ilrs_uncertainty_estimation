import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

symbols = ["X", "Y", "Z", "Vx", "Vy", "Vz"]


def produce_init_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Function that returns a summary dataframe containing all symbols and 2 out of the 3 summary values for each."""
    dfc = copy.deepcopy(df)
    dfc["date"] = pd.to_datetime(dfc["date"])
    dfc.set_index("date", inplace=True)
    summary_df = pd.DataFrame(
        columns=[f"{symbol} - E(std)" for symbol in symbols]
        + [f"{symbol} - STD(rms)" for symbol in symbols]
        + [f"{symbol} - RMS(std)" for symbol in symbols]
    )
    for symbol in symbols:
        summary_df[f"{symbol} - E(std)"] = dfc[f"{symbol}std"].resample("30D").mean()
        summary_df[f"{symbol} - STD(rms)"] = dfc[f"{symbol}std"].resample("30D").std()
        summary_df[f"{symbol} - RMS(std)"] = None

    return summary_df.reset_index()


def calculate_rms_of_stds(df: pd.DataFrame, summary_df: pd.DataFrame, symbol: str):
    """Function which modifies the summary dataframe and adds the 3rd summary value."""
    for j in range(len(summary_df[f"{symbol} - E(std)"])):
        istart = 30 * j
        if istart + 30 < len(df[f"{symbol}std"]):
            iend = istart + 30
        else:
            iend = len(df[f"{symbol}std"])
        Sum = 0
        for i in range(istart, iend):
            Sum += (
                df[f"{symbol}std"].iloc[i] - summary_df[f"{symbol} - E(std)"].iloc[j]
            ) ** 2
        RMS = np.sqrt(Sum)
        summary_df.at[j, f"{symbol} - RMS(std)"] = RMS


def produce_final_summary_df(df: pd.DataFrame, summary_df: pd.DataFrame):
    for symbol in symbols:
        calculate_rms_of_stds(df, summary_df, symbol)


def plot_raw_data(df: pd.DataFrame):
    """Produces plots of the raw data with only 4 dates just to guide the eye."""
    dfc = copy.deepcopy(df)
    dfc["date"] = pd.to_datetime(dfc["date"])

    symbol_pairs = list(zip(symbols[0:3], symbols[3:]))

    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    for i, pair in enumerate(symbol_pairs):
        axs[i][0].plot(dfc["date"], dfc[f"{pair[0]}rms"], label="rms")
        axs[i][0].plot(dfc["date"], dfc[f"{pair[0]}std"], label="std")
        axs[i][0].set_title(f"{pair[0]}")
        axs[i][0].legend()
        axs[i][1].plot(dfc["date"], dfc[f"{pair[1]}rms"], label="rms")
        axs[i][1].plot(dfc["date"], dfc[f"{pair[1]}std"], label="std")
        axs[i][1].set_title(f"{pair[1]}")
        axs[i][1].legend()
    for ax in axs.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.tight_layout()
    plt.show()


def plot_summary(df: pd.DataFrame):

    symbol_pairs = list(zip(symbols[0:3], symbols[3:]))

    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    for i, pair in enumerate(symbol_pairs):
        axs[i][0].plot(df.index, df[f"{pair[0]} - E(std)"], label="E(std)")
        axs[i][0].plot(df.index, df[f"{pair[0]} - STD(rms)"], label="STD(rms)")
        axs[i][0].plot(df.index, df[f"{pair[0]} - RMS(std)"], label="RMS(std)")
        axs[i][0].set_title(f"{pair[0]}")
        axs[i][0].legend()
        axs[i][1].plot(df.index, df[f"{pair[1]} - E(std)"], label="E(std)")
        axs[i][1].plot(df.index, df[f"{pair[1]} - STD(rms)"], label="STD(rms)")
        axs[i][1].plot(df.index, df[f"{pair[1]} - RMS(std)"], label="RMS(std)")
        axs[i][1].set_title(f"{pair[1]}")
        axs[i][1].legend()
    for ax in axs.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.tight_layout()
    plt.show()


def main(filepath: str):
    df = pd.read_csv(filepath)
    plot_raw_data(df)
    summary_df = produce_init_summary_df(df)
    produce_final_summary_df(df, summary_df)
    plot_summary(summary_df)
    print(summary_df.tail(3))
