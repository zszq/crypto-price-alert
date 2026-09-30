from price_alert.universe import select_liquid_contracts


def test_selects_only_usdt_contracts_strictly_above_quote_volume_threshold():
    tickers = [
        {"contract": "BTC_USDT", "last": "70000", "volume_24h_quote": "10000001"},
        {"contract": "ETH_USDT", "last": "2000", "volume_24h_quote": "10000000"},
        {"contract": "BTC_USD", "last": "70000", "volume_24h_quote": "999999999"},
        {"contract": "BAD_USDT", "last": "0", "volume_24h_quote": "999999999"},
    ]
    contracts = [
        {"name": "BTC_USDT", "contract_type": "", "status": "trading"},
        {"name": "ETH_USDT", "contract_type": "", "status": "trading"},
        {"name": "BTC_USD", "contract_type": "", "status": "trading"},
        {"name": "BAD_USDT", "contract_type": "", "status": "trading"},
    ]

    selected = select_liquid_contracts(tickers, contracts, 10_000_000)

    assert [item.symbol for item in selected] == ["BTC_USDT"]


def test_uses_deprecated_usd_volume_only_as_fallback():
    selected = select_liquid_contracts(
        [{"contract": "SOL_USDT", "last": "100", "volume_24h_usd": "20000000"}],
        [{"name": "SOL_USDT", "contract_type": "", "status": "trading"}],
        10_000_000,
    )
    assert selected[0].volume_24h_quote == 20_000_000


def test_excludes_non_crypto_and_non_trading_contracts():
    tickers = [
        {"contract": symbol, "last": "100", "volume_24h_quote": "20000000"}
        for symbol in ("BTC_USDT", "OPENAI_USDT", "XAU_USDT", "EUR_USDT", "OLD_USDT")
    ]
    contracts = [
        {"name": "BTC_USDT", "contract_type": "", "status": "trading"},
        {"name": "OPENAI_USDT", "contract_type": "stocks", "status": "trading"},
        {"name": "XAU_USDT", "contract_type": "metals", "status": "trading"},
        {"name": "EUR_USDT", "contract_type": "forex", "status": "trading"},
        {"name": "OLD_USDT", "contract_type": "", "status": "delisting"},
    ]

    selected = select_liquid_contracts(tickers, contracts, 10_000_000)

    assert [item.symbol for item in selected] == ["BTC_USDT"]


def test_excludes_contract_when_classification_is_missing():
    selected = select_liquid_contracts(
        [{"contract": "UNKNOWN_USDT", "last": "1", "volume_24h_quote": "20000000"}],
        [{"name": "UNKNOWN_USDT", "status": "trading"}],
        10_000_000,
    )

    assert selected == []


def test_non_finite_exchange_numbers_are_treated_as_invalid():
    tickers = [
        {"contract": "INF_USDT", "last": "inf", "volume_24h_quote": "20000000"},
        {"contract": "INFVOL_USDT", "last": "1", "volume_24h_quote": "inf"},
        {"contract": "NAN_USDT", "last": "1", "volume_24h_quote": "nan"},
    ]
    contracts = [{"name": item["contract"], "contract_type": "", "status": "trading"} for item in tickers]

    assert select_liquid_contracts(tickers, contracts, 10_000_000) == []


def test_excludes_contracts_in_delisting():
    selected = select_liquid_contracts(
        [{"contract": "OLD_USDT", "last": "1", "volume_24h_quote": "20000000"}],
        [{"name": "OLD_USDT", "contract_type": "", "status": "trading", "in_delisting": True}],
        10_000_000,
    )

    assert selected == []


def test_retained_contracts_use_lower_exit_threshold():
    tickers = [
        {"contract": "KEEP_USDT", "last": "1", "volume_24h_quote": "9000000"},
        {"contract": "NEW_USDT", "last": "1", "volume_24h_quote": "9000000"},
        {"contract": "DROP_USDT", "last": "1", "volume_24h_quote": "7000000"},
    ]
    contracts = [{"name": item["contract"], "contract_type": "", "status": "trading"} for item in tickers]

    selected = select_liquid_contracts(
        tickers,
        contracts,
        10_000_000,
        retained_symbols=["keep_usdt", "DROP_USDT"],
        exit_volume_ratio=0.8,
    )

    assert [item.symbol for item in selected] == ["KEEP_USDT"]


def test_selected_contracts_carry_price_precision_from_order_price_round():
    selected = select_liquid_contracts(
        [
            {"contract": "BTC_USDT", "last": "112345.6", "volume_24h_quote": "20000000"},
            {"contract": "ODD_USDT", "last": "1", "volume_24h_quote": "20000000"},
        ],
        [
            {"name": "BTC_USDT", "contract_type": "", "status": "trading", "order_price_round": "0.1"},
            {"name": "ODD_USDT", "contract_type": "", "status": "trading"},
        ],
        10_000_000,
    )

    assert {item.symbol: item.price_decimals for item in selected} == {"BTC_USDT": 1, "ODD_USDT": None}
