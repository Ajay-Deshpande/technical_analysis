"""
.. module:: wrapper
   :synopsis: Wrapper of Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)
"""

import pandas as pd

from ta.momentum import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.others import (
    CumulativeReturnIndicator,
    DailyLogReturnIndicator,
    DailyReturnIndicator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)


def add_volume_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
    volume_cols: set = set()
) -> pd.DataFrame:
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    if 'volume_adi' in volume_cols:
        # Accumulation Distribution Index
        df[f"{colprefix}volume_adi"] = AccDistIndexIndicator(
            high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
        ).acc_dist_index()

    if 'volume_obv' in volume_cols:
        # On Balance Volume
        df[f"{colprefix}volume_obv"] = OnBalanceVolumeIndicator(
            close=df[close], volume=df[volume], fillna=fillna
        ).on_balance_volume()

    if 'volume_cmf' in volume_cols:
        # Chaikin Money Flow
        df[f"{colprefix}volume_cmf"] = ChaikinMoneyFlowIndicator(
            high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
        ).chaikin_money_flow()

    if 'volume_fi' in volume_cols:
        # Force Index
        df[f"{colprefix}volume_fi"] = ForceIndexIndicator(
            close=df[close], volume=df[volume], window=13, fillna=fillna
        ).force_index()

    if 'volume_em' in volume_cols or 'volume_sma_em' in volume_cols:
        # Ease of Movement
        indicator_eom = EaseOfMovementIndicator(
            high=df[high], low=df[low], volume=df[volume], window=14, fillna=fillna
        )
        df[f"{colprefix}volume_em"] = indicator_eom.ease_of_movement()
        df[f"{colprefix}volume_sma_em"] = indicator_eom.sma_ease_of_movement()

    if 'volume_vpt' in volume_cols:
        # Volume Price Trend
        df[f"{colprefix}volume_vpt"] = VolumePriceTrendIndicator(
            close=df[close], volume=df[volume], fillna=fillna
        ).volume_price_trend()

    if 'volume_vwap' in volume_cols:
        # Volume Weighted Average Price
        df[f"{colprefix}volume_vwap"] = VolumeWeightedAveragePrice(
            high=df[high],
            low=df[low],
            close=df[close],
            volume=df[volume],
            window=14,
            fillna=fillna,
        ).volume_weighted_average_price()

    if not vectorized:
        if 'volume_mfi' in volume_cols:
            # Money Flow Indicator
            df[f"{colprefix}volume_mfi"] = MFIIndicator(
                high=df[high],
                low=df[low],
                close=df[close],
                volume=df[volume],
                window=14,
                fillna=fillna,
            ).money_flow_index()

        if 'volume_nvi' in volume_cols:
            # Negative Volume Index
            df[f"{colprefix}volume_nvi"] = NegativeVolumeIndexIndicator(
                close=df[close], volume=df[volume], fillna=fillna
            ).negative_volume_index()

    return df


def add_volatility_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
    volatility_cols: set = set()
) -> pd.DataFrame:
    """Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    if 'volatility_bbm' in volatility_cols or \
        'volatility_bbh' in volatility_cols or \
        'volatility_bbl' in volatility_cols or \
        'volatility_bbw' in volatility_cols or \
        'volatility_bbp' in volatility_cols or \
        'volatility_bbhi' in volatility_cols or \
        'volatility_bbli' in volatility_cols:
        # Bollinger Bands
        indicator_bb = BollingerBands(
            close=df[close], window=20, window_dev=2, fillna=fillna
        )
        df[f"{colprefix}volatility_bbm"] = indicator_bb.bollinger_mavg()
        df[f"{colprefix}volatility_bbh"] = indicator_bb.bollinger_hband()
        df[f"{colprefix}volatility_bbl"] = indicator_bb.bollinger_lband()
        df[f"{colprefix}volatility_bbw"] = indicator_bb.bollinger_wband()
        df[f"{colprefix}volatility_bbp"] = indicator_bb.bollinger_pband()
        df[f"{colprefix}volatility_bbhi"] = indicator_bb.bollinger_hband_indicator()
        df[f"{colprefix}volatility_bbli"] = indicator_bb.bollinger_lband_indicator()

    if "volatility_kcc" in volatility_cols or \
        "volatility_kch" in volatility_cols or \
        "volatility_kcl" in volatility_cols or \
        "volatility_kcw" in volatility_cols or \
        "volatility_kcp" in volatility_cols or \
        "volatility_kchi" in volatility_cols or \
        "volatility_kcli" in volatility_cols:
        # Keltner Channel
        indicator_kc = KeltnerChannel(
            close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
        )
        df[f"{colprefix}volatility_kcc"] = indicator_kc.keltner_channel_mband()
        df[f"{colprefix}volatility_kch"] = indicator_kc.keltner_channel_hband()
        df[f"{colprefix}volatility_kcl"] = indicator_kc.keltner_channel_lband()
        df[f"{colprefix}volatility_kcw"] = indicator_kc.keltner_channel_wband()
        df[f"{colprefix}volatility_kcp"] = indicator_kc.keltner_channel_pband()
        df[f"{colprefix}volatility_kchi"] = indicator_kc.keltner_channel_hband_indicator()
        df[f"{colprefix}volatility_kcli"] = indicator_kc.keltner_channel_lband_indicator()


    if "volatility_dcl" in volatility_cols or \
        "volatility_dch" in volatility_cols or \
        "volatility_dcm" in volatility_cols or \
        "volatility_dcw" in volatility_cols or \
        "volatility_dcp" in volatility_cols:
        # Donchian Channel
        indicator_dc = DonchianChannel(
            high=df[high], low=df[low], close=df[close], window=20, offset=0, fillna=fillna
        )
        df[f"{colprefix}volatility_dcl"] = indicator_dc.donchian_channel_lband()
        df[f"{colprefix}volatility_dch"] = indicator_dc.donchian_channel_hband()
        df[f"{colprefix}volatility_dcm"] = indicator_dc.donchian_channel_mband()
        df[f"{colprefix}volatility_dcw"] = indicator_dc.donchian_channel_wband()
        df[f"{colprefix}volatility_dcp"] = indicator_dc.donchian_channel_pband()

    if not vectorized:
        if 'volatility_atr' in volatility_cols:
            # Average True Range
            df[f"{colprefix}volatility_atr"] = AverageTrueRange(
                close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
            ).average_true_range()

        if 'volatility_ui' in volatility_cols:
            # Ulcer Index
            df[f"{colprefix}volatility_ui"] = UlcerIndex(
                close=df[close], window=14, fillna=fillna
            ).ulcer_index()

    return df


def add_trend_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
    trend_cols: set = set()
) -> pd.DataFrame:
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    if 'trend_macd' in trend_cols or \
        'trend_macd_signal' in trend_cols or \
        'trend_macd_diff' in trend_cols:
        # MACD
        indicator_macd = MACD(
            close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
        )
        df[f"{colprefix}trend_macd"] = indicator_macd.macd()
        df[f"{colprefix}trend_macd_signal"] = indicator_macd.macd_signal()
        df[f"{colprefix}trend_macd_diff"] = indicator_macd.macd_diff()

    if 'trend_sma_fast' in trend_cols or 'trend_sma_slow' in trend_cols:
        # SMAs
        df[f"{colprefix}trend_sma_fast"] = SMAIndicator(
            close=df[close], window=12, fillna=fillna
        ).sma_indicator()
        df[f"{colprefix}trend_sma_slow"] = SMAIndicator(
            close=df[close], window=26, fillna=fillna
        ).sma_indicator()

    if 'trend_ema_fast' in trend_cols or 'trend_ema_slow' in trend_cols:
        # EMAs
        df[f"{colprefix}trend_ema_fast"] = EMAIndicator(
            close=df[close], window=12, fillna=fillna
        ).ema_indicator()
        df[f"{colprefix}trend_ema_slow"] = EMAIndicator(
            close=df[close], window=26, fillna=fillna
        ).ema_indicator()

    if 'trend_vortex_ind_pos' in trend_cols or \
        "trend_vortex_ind_neg" in trend_cols or \
        "trend_vortex_ind_diff" in trend_cols:
        # Vortex Indicator
        indicator_vortex = VortexIndicator(
            high=df[high], low=df[low], close=df[close], window=14, fillna=fillna
        )
        df[f"{colprefix}trend_vortex_ind_pos"] = indicator_vortex.vortex_indicator_pos()
        df[f"{colprefix}trend_vortex_ind_neg"] = indicator_vortex.vortex_indicator_neg()
        df[f"{colprefix}trend_vortex_ind_diff"] = indicator_vortex.vortex_indicator_diff()

    if 'trend_trix' in trend_cols:
        # TRIX Indicator
        df[f"{colprefix}trend_trix"] = TRIXIndicator(
            close=df[close], window=15, fillna=fillna
        ).trix()

    if 'trend_mass_index' in trend_cols:
        # Mass Index
        df[f"{colprefix}trend_mass_index"] = MassIndex(
            high=df[high], low=df[low], window_fast=9, window_slow=25, fillna=fillna
        ).mass_index()

    if "trend_dpo" in trend_cols:
        # DPO Indicator
        df[f"{colprefix}trend_dpo"] = DPOIndicator(
            close=df[close], window=20, fillna=fillna
        ).dpo()

    if "trend_kst" in trend_cols or \
        "trend_kst_sig" in trend_cols or \
        "trend_kst_diff" in trend_cols:
        # KST Indicator
        indicator_kst = KSTIndicator(
            close=df[close],
            roc1=10,
            roc2=15,
            roc3=20,
            roc4=30,
            window1=10,
            window2=10,
            window3=10,
            window4=15,
            nsig=9,
            fillna=fillna,
        )
        df[f"{colprefix}trend_kst"] = indicator_kst.kst()
        df[f"{colprefix}trend_kst_sig"] = indicator_kst.kst_sig()
        df[f"{colprefix}trend_kst_diff"] = indicator_kst.kst_diff()

    if 'trend_ichimoku_conv' in trend_cols or \
        "trend_ichimoku_base" in trend_cols or \
        "trend_ichimoku_a" in trend_cols or \
        "trend_ichimoku_b" in trend_cols:
        # Ichimoku Indicator
        indicator_ichi = IchimokuIndicator(
            high=df[high],
            low=df[low],
            window1=9,
            window2=26,
            window3=52,
            visual=False,
            fillna=fillna,
        )
        df[f"{colprefix}trend_ichimoku_conv"] = indicator_ichi.ichimoku_conversion_line()
        df[f"{colprefix}trend_ichimoku_base"] = indicator_ichi.ichimoku_base_line()
        df[f"{colprefix}trend_ichimoku_a"] = indicator_ichi.ichimoku_a()
        df[f"{colprefix}trend_ichimoku_b"] = indicator_ichi.ichimoku_b()

    if 'trend_stc' in trend_cols:
        # Schaff Trend Cycle (STC)
        df[f"{colprefix}trend_stc"] = STCIndicator(
            close=df[close],
            window_slow=50,
            window_fast=23,
            cycle=10,
            smooth1=3,
            smooth2=3,
            fillna=fillna,
        ).stc()

    if not vectorized:
        if "trend_adx" in trend_cols or \
            "trend_adx_pos" in trend_cols or \
            "trend_adx_neg" in trend_cols:
            # Average Directional Movement Index (ADX)
            indicator_adx = ADXIndicator(
                high=df[high], low=df[low], close=df[close], window=14, fillna=fillna
            )
            df[f"{colprefix}trend_adx"] = indicator_adx.adx()
            df[f"{colprefix}trend_adx_pos"] = indicator_adx.adx_pos()
            df[f"{colprefix}trend_adx_neg"] = indicator_adx.adx_neg()

        if "trend_cci" in trend_cols:
            # CCI Indicator
            df[f"{colprefix}trend_cci"] = CCIIndicator(
                high=df[high],
                low=df[low],
                close=df[close],
                window=20,
                constant=0.015,
                fillna=fillna,
            ).cci()

        if "trend_visual_ichimoku_a" in trend_cols or "trend_visual_ichimoku_b" in trend_cols:
            # Ichimoku Visual Indicator
            indicator_ichi_visual = IchimokuIndicator(
                high=df[high],
                low=df[low],
                window1=9,
                window2=26,
                window3=52,
                visual=True,
                fillna=fillna,
            )
            df[f"{colprefix}trend_visual_ichimoku_a"] = indicator_ichi_visual.ichimoku_a()
            df[f"{colprefix}trend_visual_ichimoku_b"] = indicator_ichi_visual.ichimoku_b()

        if "trend_aroon_up" in trend_cols or \
            "trend_aroon_down" in trend_cols or \
             "trend_aroon_ind" in trend_cols:
            # Aroon Indicator
            indicator_aroon = AroonIndicator(
                high=df[high], low=df[low], window=25, fillna=fillna
            )
            df[f"{colprefix}trend_aroon_up"] = indicator_aroon.aroon_up()
            df[f"{colprefix}trend_aroon_down"] = indicator_aroon.aroon_down()
            df[f"{colprefix}trend_aroon_ind"] = indicator_aroon.aroon_indicator()

        if "trend_psar_up" in trend_cols or \
            "trend_psar_down" in trend_cols or \
            "trend_psar_up_indicator" in trend_cols or \
            "trend_psar_down_indicator" in trend_cols:
            # PSAR Indicator
            indicator_psar = PSARIndicator(
                high=df[high],
                low=df[low],
                close=df[close],
                step=0.02,
                max_step=0.20,
                fillna=fillna,
            )
            # df[f'{colprefix}trend_psar'] = indicator.psar()
            df[f"{colprefix}trend_psar_up"] = indicator_psar.psar_up()
            df[f"{colprefix}trend_psar_down"] = indicator_psar.psar_down()
            df[f"{colprefix}trend_psar_up_indicator"] = indicator_psar.psar_up_indicator()
            df[
                f"{colprefix}trend_psar_down_indicator"
            ] = indicator_psar.psar_down_indicator()

    return df


def add_momentum_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
    momentum_cols: set = set()
) -> pd.DataFrame:
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    if 'momentum_rsi' in momentum_cols:
        # Relative Strength Index (RSI)
        df[f"{colprefix}momentum_rsi"] = RSIIndicator(
            close=df[close], window=14, fillna=fillna
        ).rsi()

    if 'momentum_stoch_rsi' in momentum_cols or \
        "momentum_stoch_rsi_k" in momentum_cols or \
        "momentum_stoch_rsi_d" in momentum_cols:
        # Stoch RSI (StochRSI)
        indicator_srsi = StochRSIIndicator(
            close=df[close], window=14, smooth1=3, smooth2=3, fillna=fillna
        )
        df[f"{colprefix}momentum_stoch_rsi"] = indicator_srsi.stochrsi()
        df[f"{colprefix}momentum_stoch_rsi_k"] = indicator_srsi.stochrsi_k()
        df[f"{colprefix}momentum_stoch_rsi_d"] = indicator_srsi.stochrsi_d()

    if 'momentum_tsi' in momentum_cols:
        # TSI Indicator
        df[f"{colprefix}momentum_tsi"] = TSIIndicator(
            close=df[close], window_slow=25, window_fast=13, fillna=fillna
        ).tsi()

    if 'momentum_uo' in momentum_cols:
        # Ultimate Oscillator
        df[f"{colprefix}momentum_uo"] = UltimateOscillator(
            high=df[high],
            low=df[low],
            close=df[close],
            window1=7,
            window2=14,
            window3=28,
            weight1=4.0,
            weight2=2.0,
            weight3=1.0,
            fillna=fillna,
        ).ultimate_oscillator()

    if 'momentum_stoch' in momentum_cols or 'momentum_stoch_signal' in momentum_cols:
        # Stoch Indicator
        indicator_so = StochasticOscillator(
            high=df[high],
            low=df[low],
            close=df[close],
            window=14,
            smooth_window=3,
            fillna=fillna,
        )
        df[f"{colprefix}momentum_stoch"] = indicator_so.stoch()
        df[f"{colprefix}momentum_stoch_signal"] = indicator_so.stoch_signal()

    if 'momentum_wr' in momentum_cols:
        # Williams R Indicator
        df[f"{colprefix}momentum_wr"] = WilliamsRIndicator(
            high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna
        ).williams_r()

    if 'momentum_ao' in momentum_cols:
        # Awesome Oscillator
        df[f"{colprefix}momentum_ao"] = AwesomeOscillatorIndicator(
            high=df[high], low=df[low], window1=5, window2=34, fillna=fillna
        ).awesome_oscillator()

    if 'momentum_roc' in momentum_cols:
        # Rate Of Change
        df[f"{colprefix}momentum_roc"] = ROCIndicator(
            close=df[close], window=12, fillna=fillna
        ).roc()

    if "momentum_ppo" in momentum_cols or \
        "momentum_ppo_signal" in momentum_cols or \
        "momentum_ppo_hist" in momentum_cols:
        # Percentage Price Oscillator
        indicator_ppo = PercentagePriceOscillator(
            close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
        )
        df[f"{colprefix}momentum_ppo"] = indicator_ppo.ppo()
        df[f"{colprefix}momentum_ppo_signal"] = indicator_ppo.ppo_signal()
        df[f"{colprefix}momentum_ppo_hist"] = indicator_ppo.ppo_hist()

    if "momentum_pvo" in momentum_cols or \
        "momentum_pvo_signal" in momentum_cols or \
        "momentum_pvo_hist" in momentum_cols:
        # Percentage Volume Oscillator
        indicator_pvo = PercentageVolumeOscillator(
            volume=df[volume], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
        )
        df[f"{colprefix}momentum_pvo"] = indicator_pvo.pvo()
        df[f"{colprefix}momentum_pvo_signal"] = indicator_pvo.pvo_signal()
        df[f"{colprefix}momentum_pvo_hist"] = indicator_pvo.pvo_hist()

    if not vectorized:
        if 'momentum_kama' in momentum_cols:
            # KAMA
            df[f"{colprefix}momentum_kama"] = KAMAIndicator(
                close=df[close], window=10, pow1=2, pow2=30, fillna=fillna
            ).kama()

    return df


def add_others_ta(
    df: pd.DataFrame,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
    other_ind_cols: set = set()
) -> pd.DataFrame:
    """Add others analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    if 'others_dr' in other_ind_cols:
        # Daily Return
        df[f"{colprefix}others_dr"] = DailyReturnIndicator(
            close=df[close], fillna=fillna
        ).daily_return()

    if 'others_dlr' in other_ind_cols:
        # Daily Log Return
        df[f"{colprefix}others_dlr"] = DailyLogReturnIndicator(
            close=df[close], fillna=fillna
        ).daily_log_return()

    if 'others_cr' in other_ind_cols:
        # Cumulative Return
        df[f"{colprefix}others_cr"] = CumulativeReturnIndicator(
            close=df[close], fillna=fillna
        ).cumulative_return()

    return df


def add_all_ta_features(
    df: pd.DataFrame,
    open: str,  # noqa
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
    momentum_cols: set = set(),
    trend_cols: set = set(),
    volume_cols: set = set(),
    volatility_cols: set = set(),
    other_ind_cols: set = set(),
) -> pd.DataFrame:
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        volume=volume,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
        volume_cols=volume_cols
    )
    df = add_volatility_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
        volatility_cols=volatility_cols
    )
    df = add_trend_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
        trend_cols=trend_cols
    )
    df = add_momentum_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        volume=volume,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
        momentum_cols=momentum_cols
    )
    df = add_others_ta(df=df, close=close, fillna=fillna, colprefix=colprefix, other_ind_cols=other_ind_cols)
    return df
