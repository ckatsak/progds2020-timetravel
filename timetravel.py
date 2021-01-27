#!/usr/bin/env python3
"""
Solution of the project in Python for ProgDS course at ECE NTUA, Fall 2020

Python 3.8.5 64-bit

Author: Christos Katsakioris <ckatsak [at] cslab [dot] ece [dot] ntua [dot] gr>

Last update: Tue 26 Jan 2021 05:19:07 PM EET

"""

import argparse
import datetime as dt
import glob
import itertools
import logging
import os
import os.path
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

#
# Constants
#
DEFAULT_STOCKS_PATH = "./Stocks"
"""The directory path of the stock data for all companies."""
DEFAULT_RESULTS_FILE_SMALL = "./small.txt"
"""The file path where the small result sequence will be flushed into."""
DEFAULT_RESULTS_FILE_LARGE = "./large.txt"
"""The file path where the large result sequence will be flushed into."""
DEFAULT_PLOT_FILE_SMALL = "./small.png"
"""The file path where the plot for the small sequence will be flushed into."""
DEFAULT_PLOT_FILE_LARGE = "./large.png"
"""The file path where the plot for the large sequence will be flushed into."""

COMMISSION = 0.01
"""Set to 1% per transaction, be it a `BUY-*` or a `SELL-*` action."""

FIRST_DATE = dt.datetime.strptime("1962-01-02", "%Y-%m-%d").date()
LAST_DATE = dt.datetime.strptime("2017-11-10", "%Y-%m-%d").date()
ONE_DAY = dt.timedelta(days=1)

# Small sequence constants
SMALL_TRANSACTION_LIMIT = 1000
WINDOW = dt.timedelta(
    days=int((LAST_DATE - FIRST_DATE).days / (SMALL_TRANSACTION_LIMIT / 2))
)
"""The sliding window's size (whole days) depends on the transaction limit."""

# Large sequence constants
LARGE_TRANSACTION_LIMIT = 1000000
DAILY_TXS = (LARGE_TRANSACTION_LIMIT // 2) // int(
    (LAST_DATE - FIRST_DATE).days * (5 / 7)
)
"""
Daily transactions are roughly `(LARGE_TRANSACTION_LIMIT / 2)` (to account for
one "sell" action per each "buy" action) divided by the number of days that we
are allowed to transact (i.e., total duration * (5/7), to account for the
weekends, the holidays, possibly missing days in the dataset, but also possibly
allow a slight oversubscription of the transactions on the days).

"""


SELECTED_COMPANIES = [
    "aapl",  # Apple
    "amd",  # AMD
    "amzn",  # Amazon
    "bp",  # BP
    "cmcsa",  # Comcast
    "csco",  # Cisco
    "dis",  # Disney
    "fb",  # Facebook
    "ge",  # General Electric
    "googl",  # Alphabet
    "ibm",  # IBM
    "intc",  # Intel
    "jpm",  # JP Morgan
    "ko",  # Coca Cola
    "msft",  # Microsoft
    "ntgr",  # Netgear
    "nvda",  # Nvidia
    "orcl",  # Oracle
    "pfe",  # Pfizer
    "pg",  # P&G
    "t",  # AT&T
    "mo*",
    "oi*",
    "x*",
    "y*",
]


#
# Global variables
#
balance = 1.0
valuation = {}
portfolio = {}
result_sequence = []
results_filepath: str
plot_filepath: str


def process_company(df: pd.DataFrame) -> Tuple[float, int, float, int]:
    """
    Calculate ((min_price, date), (max_diff, date)) for the given df.

    Used in the case of the SMALL sequence.

    """
    min_price, max_diff = float("inf"), float("-inf")
    i_min, i_max = None, None
    for i, row in df.iterrows():
        price = df.loc[i]["Low"]
        diff = df.loc[i]["High"] - min_price
        if diff > max_diff:
            max_diff, i_max = diff, i
        if price < min_price:
            min_price, i_min = price, i
    return (min_price, i_min, max_diff, i_max)


Company = Tuple[str, Tuple[float, int], Tuple[float, int]]
"""Simple type alias, convenient in declarations."""


def great_filter(companies: List[Company]) -> List[Company]:
    """
    Filter out any companies that do not match the criteria (i.e., it should be
    currently affordable and its high price must be preceding its low price),
    returning the rest of them sorted (desc) by max_diff.

    Used in the case of the SMALL sequence.

    """
    try:
        return sorted(
            filter(
                # MUST be able to afford it; i.e., min_price * 1.01 <= balance
                lambda c: c[1][0] * (1 + COMMISSION) < balance
                # low MUST be earlier than high; i.e., min_index < max_index
                and c[1][1] < c[2][1]
                # must hold: `sell_price * .99 > buy_price * 1.1`
                and (c[1][0] + c[2][0]) * (1.0 - COMMISSION)
                > (c[1][0]) * (1.0 + COMMISSION),
                companies,
            ),
            key=lambda c: c[2][0],
            reverse=True,
        )
    except TypeError as te:
        # One of the Timestamps at c[2][1] around 8/9-2009 is found to be int
        logging.debug(te)
        return []


def process_window(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Given a dataframe for a time period, return the best choice of a company
    to buy and sell stocks of (both actions within the same time period).

    Used in the case of the SMALL sequence.

    """
    logging.debug("{} companies in total".format(len(df.Company.unique())))
    logging.debug("Current balance = ${}".format(balance))
    companies = []
    for company in df.Company.unique():
        mnp, imn, mxd, imx = process_company(df[df.Company == company])
        companies.append((company, (mnp, imn), (mxd, imx)))
    companies = great_filter(companies)
    logging.debug(companies)

    # Skip this window if there is nothing interesting or actionable
    if not companies:
        return None

    # Our best choice of a company for this window
    best = companies[0]
    bdf = df[df.Company == best[0]]
    # Calculate the maximum volume we can buy
    vol = min(
        # The maximum volume we can afford at the lowest price (+ COMMISSION)
        int(balance / (bdf.loc[best[1][1]]["Low"] * (1.0 + COMMISSION))),
        # The 10% of the total volume exchanged on min_price day
        bdf.loc[best[1][1]]["Volume"] // 10,
        # The 10% of the total volume exchanged on max_diff day
        bdf.loc[best[2][1]]["Volume"] // 10,
    )
    # Skip this window if we cannot buy any stocks of the best company
    if not vol:
        return None
    # logging.debug(
    #     "Buy {} stocks of {} on {} at ${} and sell them on {} at ${}".format(
    #         vol,
    #         best[0],
    #         best[1][1],
    #         best[1][0],
    #         best[2][1],
    #         best[1][0] + best[2][0],
    #     )
    # )
    return {
        "df": bdf,
        "company": best[0],
        "min_price": best[1][0],
        "date_min": best[1][1],
        "max_diff": best[2][0],
        "date_max": best[2][1],
        "volume": vol,
    }


def buy_low(date: dt.datetime, row: pd.Series, vol: int) -> None:
    """
    Update the balance and the portfolio of a new 'buy-low' action described by
    the given arguments.

    """
    _buy("Low", date, row, vol)


def buy_open(date: dt.datetime, row: pd.Series, vol: int) -> None:
    """
    Update the balance and the portfolio of a new 'buy-open' action described
    by the given arguments.

    """
    _buy("Open", date, row, vol)


def sell_high(date: dt.datetime, row: pd.Series, vol: int) -> None:
    """
    Update the balance and the portfolio of a new 'sell-high' action described
    by the given arguments.

    """
    _sell("High", date, row, vol)


def sell_close(date: dt.datetime, row: pd.Series, vol: int) -> None:
    """
    Update the balance and the portfolio of a new 'sell-close' action described
    by the given arguments.

    """
    _sell("Close", date, row, vol)


def _buy(when: str, date: dt.datetime, row: pd.Series, vol: int) -> None:
    global balance
    charge = (row[when] * vol) * (1.0 + COMMISSION)
    assert balance >= charge, "{}=balance < charge={}".format(balance, charge)
    logging.debug(
        "BUY-{} {}: {} stocks of '{}' for a total of ${}".format(
            when, date.strftime("%Y-%m-%d"), vol, row["Company"], charge
        )
    )
    balance -= charge
    try:
        portfolio[row["Company"]] += vol
    except KeyError:
        portfolio[row["Company"]] = vol
    result_sequence.append(
        {
            "date": date.strftime("%Y-%m-%d"),
            "action": "buy-" + when.lower(),
            "company": row["Company"],
            "vol": vol,
        }
    )


def _sell(when: str, date: dt.datetime, row: pd.Series, vol: int) -> None:
    global balance
    profit = (row[when] * vol) * (1.0 - COMMISSION)
    logging.debug(
        "SELL-{} {}: {} stocks of '{}' for a total of ${}".format(
            when, date.strftime("%Y-%m-%d"), vol, row["Company"], profit
        )
    )
    assert portfolio[row["Company"]] == vol, "inconsistent portfolio"
    portfolio[row["Company"]] -= vol
    balance += profit
    result_sequence.append(
        {
            "date": date.strftime("%Y-%m-%d"),
            "action": "sell-" + when.lower(),
            "company": row["Company"],
            "vol": vol,
        }
    )


def solve_small(df: pd.DataFrame) -> None:
    """Solution for the sequence of N <= 1000 total transactions."""
    win_start = FIRST_DATE
    win_end = FIRST_DATE + WINDOW

    # Loop through the sliding window until the end of the duration
    while win_end <= LAST_DATE:
        logging.info("--> {} to {}:".format(win_start, win_end))

        best = process_window(df.loc[win_start:win_end])

        if best:
            # logging.debug("BDF:\n{}".format(best["df"]))
            old_balance = balance
            buy_low(
                best["date_min"],
                best["df"].loc[best["date_min"]],
                best["volume"],
            )
            date = win_start
            while date < best["date_min"]:
                valuation[date] = [old_balance, 0]
                date += ONE_DAY
            assert best["date_min"] == date
            # logging.debug("VALUATION:\n{}".format(valuation))
            # logging.debug("MONOTON:{}".format(best["df"].index.is_monotonic))
            # logging.debug("best['df']:\n{}".format(best["df"]))
            # logging.debug("ROW:\n{}".format(best["df"].loc[date]))
            while date < best["date_max"]:
                try:
                    valuation[date] = [
                        balance,
                        best["df"].loc[pd.Timestamp(date)]["Close"]
                        * portfolio[best["company"]],
                    ]
                except KeyError as ke:
                    # Weekends, holidays, perhaps days missing from the dataset
                    logging.debug("KeyError @ solve_small(): {}".format(ke))
                date += ONE_DAY
            sell_high(
                best["date_max"],
                best["df"].loc[best["date_max"]],
                best["volume"],
            )
            valuation[best["date_max"]] = [balance, 0]
            while date < win_end:
                valuation[date] = [balance, 0]
                date += ONE_DAY

        # Slide the window
        win_start += WINDOW + ONE_DAY
        win_end = win_start + WINDOW

    # Announce the results to the log and to stdout, and export them to a file
    export_results(results_filepath)
    plot_valuation(plot_filepath)


DeterminedAction = Tuple[str, float, float]
"""Simple type alias, convenient in declarations."""


def determine_action(df: pd.DataFrame) -> Optional[DeterminedAction]:
    """
    For the given company, return ('action', buy_price, price_diff) or None if
    any of the interesting columns is zero'd.

    Used in the case of the LARGE sequence.

    """
    row = next(df.iterrows())[1]
    for col in ["Open", "High", "Low", "Close"]:
        # Skip any company reporting weird zero values
        if not row[col]:
            return None
    op, hi = row["Open"], row["High"]
    lo, cl = row["Low"], row["Close"]
    return ("OPEN", op, hi - op) if hi - op > cl - lo else ("LOW", lo, cl - lo)


def process_day(today: dt.datetime, df: pd.DataFrame, txs: int) -> int:
    """
    Greedily buy and sell stocks of `0` to `DAILY_TXS // 2` companies within
    the given single day, leveraging either the `BUY-OPEN - SELL-HIGH` or the
    `BUY-LOW - SELL-CLOSE` pattern of intra-day trading.

    Used in the case of the LARGE sequence.

    """
    daily_cap = min(DAILY_TXS, LARGE_TRANSACTION_LIMIT - txs)
    assert not daily_cap % 2, "daily_cap cannot be an even number!"

    companies = (
        pd.Series(df.Company)
        if not isinstance(df.Company, pd.Series)
        else df.Company.unique()
    )
    op_hi, lo_cl = [], []
    for company in companies:
        # logging.debug("Processing company '{}'".format(company))
        best_action = determine_action(df[df.Company == company])
        # logging.debug("...best_action = {}".format(best_action))
        if not best_action:
            # logging.debug("...skipping it")
            continue
        # logging.debug("...appending {}".format(best_action))
        lst = op_hi if best_action[0] == "OPEN" else lo_cl
        lst.append((company, best_action[1], best_action[2]))
    # logging.debug("LISTS:\nop_hi = {}\nlo_cl = {}\n".format(op_hi, lo_cl))
    # op_hi = list(filter(lambda c: c[1] > balance, op_hi))
    # lo_cl = list(filter(lambda c: c[1] > balance, lo_cl))

    def proc(lst: List[DeterminedAction]) -> List[DeterminedAction]:
        """Filter, sort and select top candidate companies."""
        return sorted(
            filter(
                # must be profitable
                lambda c: c[2] > 0.0
                # must be affordable
                and c[1] * (1.0 + COMMISSION) < balance
                # must hold: `sell_price * .99 > buy_price * 1.1`
                and (c[1] + c[2]) * (1 - COMMISSION) > c[1] * (1 + COMMISSION),
                lst,
            ),
            key=lambda c: c[2],
            reverse=True,
        )[: daily_cap // 2]

    op_hi = proc(op_hi)
    lo_cl = proc(lo_cl)
    # logging.debug(
    #     "\nop_hi = {};\nlo_cl = {}".format(pformat(op_hi), pformat(lo_cl))
    # )

    def simulate(lst: List[DeterminedAction]) -> float:
        """Simulate a sequence of actions to project the potential profit."""
        assert len(lst) <= daily_cap
        profit = 0.0
        virt_balance = balance
        i = 0
        while virt_balance > 0.0 and i < len(lst):
            vol = min(
                int(virt_balance / (lst[i][1] * (1.0 + COMMISSION))),
                next(df[df.Company == lst[i][0]].iterrows())[1]["Volume"]
                // 10,
            )
            virt_balance -= (vol * lst[i][1]) * (1.0 + COMMISSION)
            profit += (vol * lst[i][2]) * (1.0 - COMMISSION)
            i += 1
        # logging.debug("virt_balance = {}".format(virt_balance))
        return profit

    logging.debug("OPEN simulated profit = {}".format(simulate(op_hi)))
    logging.debug("LOW simulated profit = {}".format(simulate(lo_cl)))

    global balance
    intra_type, companies = (
        ("OPEN-HIGH", op_hi)
        if simulate(op_hi) > simulate(lo_cl)
        else ("LOW-CLOSE", lo_cl)
    )
    assert len(companies) <= daily_cap // 2
    portf = {}
    i = 0
    while balance > 0.0 and i < len(companies):
        logging.debug(
            "intra_type = {}; tx = {}; companies = {}".format(
                intra_type, i * 2, companies
            )
        )
        vol = min(
            int(balance / (companies[i][1] * (1.0 + COMMISSION))),
            next(df[df.Company == companies[i][0]].iterrows())[1]["Volume"]
            // 10,
        )
        if vol:
            if intra_type == "OPEN-HIGH":
                buy_open(
                    today,
                    next(df[df.Company == companies[i][0]].iterrows())[1],
                    vol,
                )
                portf[companies[i][0]] = (
                    today,
                    next(df[df.Company == companies[i][0]].iterrows())[1],
                    vol,
                )
            else:
                buy_low(
                    today,
                    next(df[df.Company == companies[i][0]].iterrows())[1],
                    vol,
                )
                portf[companies[i][0]] = (
                    today,
                    next(df[df.Company == companies[i][0]].iterrows())[1],
                    vol,
                )
        i += 1

    for company, pz in portf.items():
        if intra_type == "OPEN-HIGH":
            sell_high(pz[0], pz[1], pz[2])
        else:
            sell_close(pz[0], pz[1], pz[2])

    return i * 2


def solve_large(df: pd.DataFrame) -> None:
    """Solution for the sequence of N <= 1000000 total transactions."""
    date = FIRST_DATE
    txs = 0
    while date <= LAST_DATE and txs < LARGE_TRANSACTION_LIMIT:
        logging.info("--> {}:".format(date))
        logging.info("Balance = ${}; # transactions = {}".format(balance, txs))
        try:
            txs += process_day(date, df.loc[pd.Timestamp(date)], txs)
        except KeyError as ke:
            # Silently skip weekends, holidays, missing data, etc
            logging.debug("KeyError @ solve_large(): {}".format(ke))
        valuation[date] = [balance, 0]
        date += ONE_DAY
    # Announce the results to the log and to stdout, and export them to a file
    export_results(results_filepath)
    plot_valuation(plot_filepath)


def export_results(results_file: str) -> None:
    """
    Announce the results (to the log and to stdout) and also export them to the
    given file properly formatted.

    """
    # Announce
    res = "Made a profit of ${} after {} transactions.".format(
        balance - 1.0, len(result_sequence)
    )
    logging.info(res)
    print(res)
    # Export
    with open(results_file, "w") as fout:
        print(len(result_sequence), file=fout)
        for entry in result_sequence:
            print("{date} {action} {company} {vol}".format(**entry), file=fout)


def plot_valuation(plot_filepath: str) -> None:
    """
    Plot the valuation (balance & portfolio) over time.

    """
    df = pd.DataFrame.from_dict(
        valuation, orient="index", columns=["Balance", "Portfolio"]
    )
    # logging.debug("PLOT_DF:\n{}".format(df))
    # logging.debug("PORTFOLIO:\n{}".format(portfolio))  # empty
    # logging.debug("VALUATION(len={}):\n{}".format(len(valuation), valuation))

    plt.figure(figsize=(20, 10))
    x = pd.to_datetime(df.index.values)
    plt.semilogy(x, df["Balance"])
    plt.semilogy(x, df["Portfolio"])

    plt.fill_between(df.index.values, df["Balance"])
    plt.gcf().autofmt_xdate()
    plt.gcf().axes[0].xaxis_date()
    plt.gcf().axes[0].xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title("Valuation")
    plt.savefig(plot_filepath)


def load_dataset(filepaths: Iterator[str]) -> pd.DataFrame:
    """
    Given a sequence of file paths that constintute the dataset, read the whole
    dataset into a single dataframe and return it.

    FIXME(ckatsak): Parameter's type annotation might be inaccurate here.

    """

    def load_company(filepath: str) -> Optional[pd.DataFrame]:
        """
        Given a path of a company's CSV file, return None if the file is empty,
        otherwise read the data into a new dataframe, create an additional
        column for the company's name, and return the new dataframe.

        """
        COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]
        try:
            df = pd.read_csv(filepath, usecols=COLS, parse_dates=["Date"])
        except Exception as e:
            logging.info("pandas.read_csv('%s'): %s", filepath, e)
            return None
        df["Company"] = os.path.basename(filepath).split(".")[0].upper()
        return df

    return (
        pd.concat(
            filter(lambda df: df is not None, map(load_company, filepaths)),
            ignore_index=True,
        )
        .sort_values("Date")
        .set_index(["Date"])
    )


def _parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=("ProgDS @ ECE NTUA, Fall 2020; by Christos Katsakioris")
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        dest="verbosity",
        help="set the verbosity of the log (max: `-vv`)",
    )
    parser.add_argument(
        "-i", "--stocks", help="path of the input directory (e.g., `Stocks/`)"
    )
    parser.add_argument(
        "-o", "--result", help="path to a file to flush the output sequence"
    )
    parser.add_argument(
        "-p", "--plot", help="path to a file to flush the output plot"
    )
    size_parser = parser.add_mutually_exclusive_group(required=True)
    size_parser.add_argument(
        "--small",
        dest="small",
        action="store_true",
        help="produce a small sequence",
    )
    size_parser.add_argument(
        "--large",
        dest="small",
        action="store_false",
        help="produce a large sequence",
    )
    return parser.parse_args()


def main(argv):
    global results_filepath, plot_filepath

    lfmt = "%(asctime)s %(name)-8s %(module)s %(levelname)-8s    %(message)s"
    try:
        args = _parse_args()

        # Setup log verbosity
        if not args.verbosity:
            llvl = logging.WARNING
        elif args.verbosity == 1:
            llvl = logging.INFO
        elif args.verbosity >= 2:
            llvl = logging.DEBUG
        logging.basicConfig(format=lfmt, level=llvl)

        # Solve for small or large
        stocks_path = args.stocks or DEFAULT_STOCKS_PATH
        if args.small:
            results_filepath = args.result or DEFAULT_RESULTS_FILE_SMALL
            plot_filepath = args.plot or DEFAULT_PLOT_FILE_SMALL
            solve_small(
                load_dataset(
                    itertools.chain.from_iterable(
                        map(
                            lambda f: glob.glob(
                                os.path.join(
                                    stocks_path, ".".join([f, "us.txt"])
                                )
                            ),
                            SELECTED_COMPANIES,
                        )
                    )
                )
            )
        else:
            results_filepath = args.result or DEFAULT_RESULTS_FILE_LARGE
            plot_filepath = args.plot or DEFAULT_PLOT_FILE_LARGE
            solve_large(
                load_dataset(
                    map(lambda dirent: dirent.path, os.scandir(stocks_path))
                )
            )
        return 0
    except Exception as e:
        logging.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
