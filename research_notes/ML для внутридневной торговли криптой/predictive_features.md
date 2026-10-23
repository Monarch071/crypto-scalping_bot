# Predictive features for short-horizon (5 min to 1 day) crypto returns, beyond a coin's own price history

Method note: these notes come from about 16 web searches. Direct page fetches failed with DNS errors for arxiv.org, link.springer.com and prestolabs.io, so most numbers below come from search-engine summaries of abstracts and snippets, not from full-text reading. Treat exact figures as "reported by the source, not independently checked in the full text". Status labels used throughout: **IS** = in-sample only; **OOS** = holdout/out-of-sample; **NET** = after transaction costs.

## Q1. Intraday and daily cross-sectional momentum/reversal in crypto (2018-2026): is it profitable after costs?

### Takeaway
Weekly-to-monthly cross-sectional momentum is an established crypto factor (Liu, Tsyvinski & Wu). At intraday horizons (15 min to 1 day) the main effect is short-term **reversal**. It is statistically very widespread, but the gross edge per trade is around 1 bp, below a ~5 bp round-trip spot cost. Recent cost-aware studies conclude it is **not exploitable by public-fee takers/makers**. Most studies that look "significant" do not survive realistic costs.

### Cited Findings
- Liu, Tsyvinski & Wu ("Common Risk Factors in Cryptocurrency", Journal of Finance 2022): three factors (crypto market, size, momentum) capture the cross-section of expected returns. The momentum is weekly-horizon and the baseline model has no short-horizon reversal factor. — [Yale Economics](https://economics.yale.edu/research/common-risk-factors-cryptocurrency); [BFI slides](https://bfi.uchicago.edu/wp-content/uploads/8_Yukun-Liu_presentation_final.pdf)
- A later four-factor model replaces momentum with market, size, **reversal** and illiquidity factors. It attributes short-term reversals to retail speculation, overreaction to news, market manipulation and forced liquidations of leveraged positions (IS, asset-pricing test, not a trading test). — [Emerald, China Accounting and Finance Review](https://www.emerald.com/cafr/article/27/4/493/1271913/Unravelling-cross-sectional-patterns-in)
- "Cryptoasset Factor Models" (2018): short-horizon crypto markets are strongly mean-reverting cross-sectionally (early-sample, IS). — [arXiv 1811.07860](https://arxiv.org/pdf/1811.07860)
- Kitron & Wengrowicz (arXiv 2608.21888, 2026 preprint): at a **15-minute** horizon, about 90% of 183 Binance pairs show significant directional reversal, against 2.7% of 187 US stocks/ETFs. The result survives an artifact battery, a permutation null and a **six-month holdout (OOS)**. But the gross edge peaks near **1.3 bp per trade against a 5 bp round-trip cost**, so it is "not exploitable under benchmark spot-cost assumptions" for a public-schedule taker or maker. 81% of the reversal signal remains after projecting out an equal-weight crypto factor. Under the most conservative accounting the crypto-vs-equity AUC gap shrinks to +0.011. The authors disclose they run an automated crypto trading system. — [arXiv abs](https://arxiv.org/abs/2608.21888); [HTML](https://arxiv.org/html/2608.21888v1)
- A secondary, auto-generated critique argues that about half of that paper's crypto-vs-equity gap is measurement artifact. It also says the confidence interval prices sampling variation but not estimation uncertainty. — [GitHub jjakimoto/research-issues #1451](https://github.com/jjakimoto/research-issues/issues/1451)
- Wen et al. (2022, North American Journal of Economics and Finance) find both intraday momentum and reversal in crypto, conditioned on jumps, FOMC announcements and liquidity levels (IS; no robust net-of-cost result was found in the snippet). — [ResearchGate](https://www.researchgate.net/publication/361202793_Intraday_return_predictability_in_the_cryptocurrency_markets_momentum_reversal_or_both)
- Shen, Urquhart & Wang (2022, Financial Review): Bitcoin **intraday time-series momentum** (earlier intraday returns predict the last half-hour). They report breakeven transaction costs rather than assuming a fixed cost, noting the commonly used 50 bp estimate varies widely. — [Univ. of Reading repository](https://centaur.reading.ac.uk/100181/3/21Sep2021Bitcoin%20Intraday%20Time-Series%20Momentum.R2.pdf)
- JRFM 2026 (MDPI) tests lagged momentum/reversal for BTC and ETH across daytime and overnight sessions with 0-2 bp cost sensitivity. It finds no significant advantage after costs. — [MDPI JRFM 19(9):692](https://www.mdpi.com/1911-8074/19/9/692)
- Dobrynskaya: at daily frequency, cross-sectional momentum lasts about 2-4 weeks, with stronger reversal at longer horizons. — [HSE conference PDF](https://conference.hse.ru/files/download_file_ex?hash=FAE0AB2DC7A67656E89A0B1CB27D8C7D&id=3B5EE9A5-0B18-458A-9458-B4ED0F6C6664)
- AUT study (time-series and cross-sectional momentum): many momentum portfolios that look significant are wiped out once realistic transaction costs and slippage are included. — [AUT ACFR PDF](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)
- "Up or down? Short-term reversal, momentum, and liquidity effects in cryptocurrency markets" documents short-term reversal interacting with liquidity (snippet only; details not verified). — [ResearchGate](https://www.researchgate.net/publication/354881251_Up_or_down_Short-term_reversal_momentum_and_liquidity_effects_in_cryptocurrency_markets)
- Informal backtest (not peer-reviewed): a daily cross-sectional reversal strategy shows strong OOS alpha, but costs eliminate it at about 138% daily turnover. — [GitHub prams2104/crypto-momentum-backtest](https://github.com/prams2104/crypto-momentum-backtest)

### Inferences
- For a 5-min to 1-day bot, cross-sectional reversal is a real, robust **feature** but a weak **standalone strategy**. It is better used as one input that improves entry timing or maker-quote placement than as a taker strategy. A ~1 bp gross edge needs maker rebates/VIP fees or high-tier fee schedules to be positive.
- Reversal is likely concentrated in less liquid alts, where costs are highest, so capacity is small.
- Decay: the reversal looks persistent in frequency (2018 paper, 2026 paper), but the profitable portion appears to have always been below costs at retail fee levels. No evidence of a once-profitable edge that later decayed was found.

### Gaps
- No peer-reviewed study was found of **intraday cross-sectional** momentum/reversal across many coins with explicit, realistic maker/taker costs on perps (perp fees are lower than spot).
- Full tables (Sharpe, turnover, net returns by year) of Kitron & Wengrowicz and of Wen et al. were not verified because full-text fetches failed.

## Q2. Do funding rate, open interest and liquidation data predict short-term returns or volatility?

### Takeaway
Funding rates are themselves **predictable** (persistent, mean-reverting), and they drive carry returns. Evidence that funding predicts **price direction** at short horizons is weak for single assets. A practitioner note suggests it explains a modest share of 7-day price changes and may work better cross-sectionally, with high turnover. No peer-reviewed hourly test of open interest or liquidations as return predictors was found. Liquidation research concerns leverage, margin and cascade risk, which bears more on volatility/tail risk than on direction.

### Cited Findings
- Inan (SSRN 5576424, Oct 2025): **OOS** forecasts of BTC perp funding rates on Binance and Bybit. Double-autoregressive (DAR) models beat a no-change benchmark on both forecast error and directional accuracy. This shows funding is forecastable; it does not show that funding predicts returns. — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424)
- Presto Labs research note (industry; date uncertain, roughly 2023-2024): funding-rate changes account for about **12.5% of the variation in 7-day price changes** for single assets. Funding data "may be more useful cross-sectionally", and a funding-based stat-arb alpha looks favorable but has **high turnover**. — [Presto Labs](https://www.prestolabs.io/research/can-funding-rate-predict-price-change) (page not fetchable; figures from search snippet)
- Theory: a linear funding rule induces an endogenous **mean-reverting basis**. — [ResearchGate, "Funding Rate Mechanism in Perpetual Futures"](https://www.researchgate.net/publication/401892113_Funding_Rate_Mechanism_in_Perpetual_Futures); no-arbitrage perp pricing in [Ackerer, Hugonnier & Jermann](https://finance.wharton.upenn.edu/~jermann/AHJ-main-10.pdf); [Fundamentals of Perpetual Futures (arXiv 2212.06888)](https://arxiv.org/html/2212.06888v7)
- He, Wang & Zhang (SSRN 6185958): across 200 Binance USDT-margined perps, the nominal funding rule does not by itself determine effective price feedback. Funding mechanics differ across contracts. — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6185958)
- "Liquidation, Leverage and Optimal Margin in Bitcoin Futures Markets" (arXiv 2102.04591): estimates long/short liquidation intensity as liquidations / open interest and infers leverage from daily high/low prices. This is daily frequency and margin-design oriented. — [arXiv](https://arxiv.org/pdf/2102.04591)
- Gornall, Rinaldi & Xiao (AEA 2026, preliminary): trader liquidations shifted from quarterly to perpetual futures, and perps show lower risk than quarterlies in part of the sample. — [AEA 2026 program](https://www.aeaweb.org/conference/2026/program/paper/ByyFEfr4)
- The 10 Oct 2025 cascade on Hyperliquid exceeded $19B and is described as the largest single-day liquidation event in crypto history. It is an example of the tail-risk channel. — [MDPI IJFS 14(7):178](https://www.mdpi.com/2227-7072/14/7/178)
- Microstructure of perps across CEX and DEX (temporal dynamics; details not verified). — [MDPI IJFS 14(5):103](https://www.mdpi.com/2227-7072/14/5/103)
- Sentiment combined with leverage: a paper links social-media sentiment and leverage to BTC volatility (abstract only). — [TPREF](https://journals.aserspublishing.eu/tpref/article/view/9522)

### Inferences
- Treat funding, OI changes and liquidation flow mainly as **regime/volatility and crowding features**: extreme positive funding plus rising OI signals crowded longs and cascade risk. Do not expect them to work as stand-alone directional alpha at 5-min horizons. They update slowly (funding every 8h or 1h; OI at seconds-to-minutes on exchange APIs), which limits value at the shortest horizons.
- Cross-sectional use (ranking coins by funding) is the most promising direction per Presto Labs, but turnover eats the edge. Evidence is industry-only.

### Gaps
- No peer-reviewed, OOS, net-of-cost study was found showing that funding rate, ΔOI or liquidation volume predicts 5-min to 1-day returns. No Coinglass/Glassnode/Kaiko quantitative backtest was found either.
- No evidence was found on whether these signals decayed after 2021.

## Q3. How strong is order-flow imbalance (OFI) and trade-sign at minute horizons, and how fast does it decay?

### Takeaway
OFI's **contemporaneous** relationship with price changes is strong (the classic Cont-Kukanov-Stoikov equity result). In crypto, the explanatory power is regime-dependent: on Binance BTC/USDT, R² for price changes jumped from 5% to 21.55% after fees were introduced in March 2023. **Predictive** power for future returns is small, with R² of 0.17%-1.00% at 1-10 second horizons, and unstable: null on about half the days in one replication. The signal lives at seconds and decays within seconds to minutes. Its gross edge is usually below taker round-trip costs, so it is useful mainly for execution and market-making.

### Cited Findings
- Cont, Kukanov & Stoikov (J. Financial Econometrics; NYSE TAQ, 50 stocks): price changes are linear in OFI, with slope inversely proportional to depth. Secondary sources cite R² of about 70% (contemporaneous; figure not confirmed from the paper). — [quantmemo summary](https://quantmemo.com/writing/paper-cont-kukanov-stoikov-order-flow-imbalance); [Medium / TDS](https://medium.com/data-science/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6)
- Binance BTC/USDT natural experiment (Springer chapter, "Fee Structure and Order Flow Informativeness"): after Binance's March 2023 move from zero fees to commissions, the explanatory power of OFI for price changes rose from **5% to 21.55% R²**. Zero-fee wash and noise flow had diluted OFI's informativeness. — [Springer](https://link.springer.com/chapter/10.1007/978-3-032-18109-1_13)
- Schmalz (SSRN 7227998), "Order Flow Imbalance and Short-Horizon BTC/USDT Returns: A Signal That Kept Needing More Scrutiny": tests Cont-style OFI on Binance.US diff-depth data for **1-10 second** mid-price returns. On about 7 days, OFI had significant positive predictive R² of **0.17%-1.00%**; on about 9 days, the pooled result was null or negative at every horizon. — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7227998)
- Multivariate Hawkes model on BTC LOB data: order-flow and LOB imbalances contain information about future price changes at intraday and **sub-second** horizons. — [Springer, Decisions in Economics and Finance (2026)](https://link.springer.com/article/10.1007/s10203-026-00570-z)
- "Explainable Patterns in Cryptocurrency Microstructure" (arXiv 2602.00776): the predictive effect of imbalance is stronger for instruments with **larger ticks**, relative to price. — [arXiv](https://arxiv.org/html/2602.00776v1)
- Bitcoin fragmentation and cross-impact (arXiv 2108.09750): classic top-of-book imbalance is extremely noisy because of tiny top-of-book sizes. Multi-level, distance-weighted imbalance and cross-venue OFI (cross-impact) are needed. — [arXiv](https://arxiv.org/pdf/2108.09750)
- Generalized OFI price impact (arXiv 2112.02947). — [arXiv](https://arxiv.org/pdf/2112.02947)
- Decay: the signal is strongest over seconds and decays quickly. One practitioner illustration shows the predicted move falling from about 0.6 to 0.1 cents as the horizon lengthens, always below round-trip cost (illustrative, not BTC data). — [Halley Tech Substack](https://halleytech.substack.com/p/order-book-imbalance-short-horizon-prediction-math-for-quants); [LuxAlgo](https://www.luxalgo.com/library/concept/order-book-imbalance/)
- At daily scale, trade-flow imbalance (B-S)/(B+S) nowcasts BTC crash risk and stays significant at the 5% level up to a 7-day lag. — [PMC 10040314](https://pmc.ncbi.nlm.nih.gov/articles/PMC10040314/)
- Order-flow image representations predict short-term **volatility** (arXiv 2304.02472). — [arXiv](https://arxiv.org/pdf/2304.02472)
- Imbalance patterns and depth structure differ across Binance, Kraken, Huobi and OKEx, and by quote currency. — [Towards Data Science](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/)

### Inferences
- At 5-min horizons, raw OFI direction is likely mostly decayed. Aggregated OFI over the last 1-5 minutes and multi-level/cross-venue OFI may remain useful as ML features, especially for volatility and for choosing passive vs aggressive execution.
- Fee-regime changes such as Binance zero-fee periods change the informativeness of OFI. Models must be retrained or regime-aware.

### Gaps
- No crypto study was found that reports a clean predictive-R² decay curve at 1, 5 and 15 minutes, nor a net-of-cost OFI taker strategy that is profitable at retail fees.

## Q4. BTC→altcoin and cross-exchange lead-lag: horizon and exploitability at about 100 ms latency

### Takeaway
BTC leads altcoins (one-way Granger causality), with slower response in small, illiquid coins. Cross-exchange BTC lags were up to about **15 seconds in 2018** tick data. Leadership has since concentrated on Binance (by volume) and on CME futures for institutional flow. Arbitrage is limited by fees, slippage and thin books. Lags between major venues are likely now sub-second (no crypto ms-level study was found), so at ~100 ms latency a retail trader cannot compete on major pairs. What remains are slower responses in small-cap alts and on DEXs at the minute scale, where costs are higher.

### Cited Findings
- 2018 tick-by-tick study across exchanges: strongly asymmetric cross-correlations, with some BTC prices following others with a lag of **up to 15 seconds**. Lags relate to infrastructure, fees and investor type. Fees, slippage and thin liquidity limit arbitrage profits. — [businessperspectives.org (IMFI 2023)](https://businessperspectives.org/images/pdf/applications/publishing/templates/article/assets/17735/IMFI_2023_01_Anderson.pdf); [CBS thesis, 2018 data](https://research.cbs.dk/en/studentProjects/high-frequency-lead-lag-relationships-in-the-bitcoin-market-an-em/)
- "Price Transmission from Bitcoin to Altcoins: High-Frequency Evidence and Implications for Trading Strategy" (Asia-Pacific Financial Markets, 2026): unidirectional Granger causality BTC → ALTs, and small caps show significant **delayed** responses. An ML lag strategy beats buy-and-hold in the authors' tests (cost treatment not verified). — [Springer](https://link.springer.com/article/10.1007/s10690-026-09589-z)
- "Altcoin-Bitcoin Arbitrage" (arXiv 1903.06033). — [arXiv](https://arxiv.org/pdf/1903.06033)
- CEX vs DEX (preliminary Zenodo preprint, 1-min OHLC): BTC shows CEX leadership with about 5-minute leads, while ETH and SOL show DEX leadership (SOL DEX about 40-minute lead vs Binance). Treat as weak evidence. — [Zenodo 17084252](https://zenodo.org/records/17084252)
- Spot vs futures: CME BTC futures lead price formation, with transaction size a key determinant (J. International Money and Finance 2025). — [IDEAS/RePEc](https://ideas.repec.org/a/eee/jimfin/v159y2025ics0261560625001500.html). Hayashi-Yoshida study: futures lead spot for BTC except under stress. — [arXiv 2506.08718](https://arxiv.org/abs/2506.08718)
- Kaiko: Binance's average daily volume was about $19B, 4x the next platform (Coinbase about $2.8B). Binance and Coinbase dominate price discovery, and Binance is described as a "single point of failure" for depth and price discovery. — [Kaiko PDF](https://resources.kaiko.com/hubfs/Liquidity%20Flywheel%20Drives%20Binance%20to%20300%20Million%20Users.pdf); [Kaiko Q1 2025 report](https://marketing.kaiko.com/hubfs/Kaiko%20Research%20-%20Quarterly%20Report,%20Q1%202025.pdf); [CryptoRank news summary](https://cryptorank.io/news/feed/0ae01-binance-liquidity-concentration-risk)
- Earlier price-discovery evidence submitted to the SEC (2021) on venue leadership in BTC. — [SEC exhibit 3A](https://www.sec.gov/files/rules/sro/nysearca/2021/34-93445-ex3a.pdf)
- Cross-impact across fragmented BTC venues: OFI on one venue moves prices on others. — [arXiv 2108.09750](https://arxiv.org/pdf/2108.09750)
- The one ms-scale lead-lag figure found (about 8 ms) is from equities (Chicago-New Jersey) and does not apply to crypto. — [arXiv 1408.3650](https://arxiv.org/pdf/1408.3650)

### Inferences
- The 15-second lag in 2018 has almost certainly compressed as market makers co-located on Binance/OKX/Bybit (inferred; no direct measurement found). With ~100 ms latency, a retail bot should not try latency arbitrage on BTC/ETH between top venues.
- More realistic uses: (a) BTC and ETH returns over the last 1-60 seconds/minutes on Binance perps as features for predicting small-cap alt moves; (b) Binance perp price as the "fair value" leader for slower venues or DEXs. The edge exists mostly in illiquid names, where spread and impact are largest.

### Gaps
- No 2023-2026 crypto study was found measuring lead-lag in milliseconds between Binance and other venues, or the net PnL of BTC→alt lead-lag after costs at a given latency.

## Q5. Funding-rate arbitrage (long spot / short perp) realized returns 2020-2026 and risks

### Takeaway
Crypto carry (futures/perp basis) has been large and volatile. It averaged about 7-8% p.a. and at times exceeded 40% p.a. (BIS). It is mostly driven by funding and reflects demand from trend-chasing retail plus limited arbitrage capital. Sharpe was very high in 2020-2023 but has **decayed**: about 4 from 2024 and reportedly negative in 2025, and yields fell below ~4% by mid-2024 as capital (e.g., Ethena) crowded in. High "up to 115.9%" or Sharpe-38 figures are peak or simulated numbers. Risks: funding flipping negative, exchange/counterparty risk, liquidation of the short leg in spikes, ADL and cascades, fees and spread. Only about 40% of top cross-venue opportunities are profitable after costs.

### Cited Findings
- BIS WP 1087 "Crypto carry" (Schmeling, Schrimpf & Todorov; April 2023, revised Oct 2025): carry (futures-spot basis) "can exceed 40% p.a." at times and is highly time-varying. It is driven by demand from smaller trend-chasing investors and limited arbitrage capital, and it predicts crash risk. The trade is long spot / short futures. — [BIS](https://bis.org/publ/work1087.htm); [PDF](https://www.bis.org/publ/work1087.pdf)
- VoxEU column by the same authors: carry is persistent with large spikes and averages **about 7-8% p.a.** depending on asset and contract. The "risk-free arbitrage" framing is misleading, and fundamentals cannot explain the magnitude. — [VoxEU/CEPR](https://cepr.org/voxeu/columns/crypto-carry-market-segmentation-and-price-distortions-digital-asset-markets); [CEPR DP20719](https://cepr.org/publications/dp20719)
- arXiv 2510.14435 ("Cryptocurrency as an Investable Asset Class: Coming of Age"): annualized Sharpe of crypto carry was **6.45 over 2020-2025**, **4.06 from 2024**, and **negative in 2025**. Profit is mostly driven by funding. — [arXiv](https://arxiv.org/pdf/2510.14435)
- BitMEX research, as summarized in news outlets: yields fell **below 4% by mid-2024** as automated capital and Ethena-style products saturated funding. — [KuCoin news](https://www.kucoin.com/news/flash/bitmex-reports-collapse-of-crypto-arbitrage-strategy-amid-market-saturation); [BitcoinWorld](https://bitcoinworld.co.in/crypto-arbitrage-strategy-collapses-bitmex/)
- BitMEX Q2 2026 derivatives report: Hyperliquid paid on average **+7.17% annualized more funding than Binance** on BTC from 2023 to 2026, positive 95% of the time. This is a cross-venue funding spread. — [BitMEX blog](https://www.bitmex.com/blog/2026q2-derivatives-report)
- "Exploring risk and return profiles of funding rate arbitrage on CEX and DEX" (Blockchain Research and Applications, 2025): covers Binance, BitMEX, ApolloX and Drift; BTC, ETH, XRP, BNB and SOL; 60 scenarios. Secondary sources cite "up to 115.9% over six months with 1.92% max drawdown". These are **peak** figures, not verified from the paper. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- CEX-DEX funding arbitrage as a basis trade (Digital Finance, 2026): **simulated** on Binance funding 2021-2024, annualized vol 2.58% and Sharpe about 38.8. The authors note this reflects the simulated carry-like nature and is not realized. — [Springer](https://link.springer.com/article/10.1007/s42521-026-00213-3)
- "Two-Tiered Structure of Cryptocurrency Funding Rate Markets" (Mathematics 2026; 749 symbols): only **about 40% of top funding-arb opportunities produce positive returns after transaction costs and spread**. — [MDPI](https://www.mdpi.com/2227-7390/14/2/346)
- Vendor blog claims average annual return of 19.26% in 2025 vs 14.39% in 2024, with drawdowns under 2%. Unsourced; conflicts with the arXiv/BitMEX decay evidence. — [Zipmex](https://zipmex.com/blog/how-to-analyze-funding-rates-in-crypto/)
- Narrative history, from the "golden age" of 2021 to the 2024-2025 period. — [BlockBeats](https://en.theblockbeats.news/news/56190)
- Paper on funding payments keeping BTC perps stable through crises. — [ResearchGate](https://www.researchgate.net/publication/388019725_Funding_Payments_Crisis-Proofed_Bitcoin's_Perpetual_Futures)

### Inferences
- The realized yield of a BTC/ETH cash-and-carry in 2024-2026 is likely low single digits to around 10% p.a., highly regime-dependent: high in bull markets, near zero or negative in 2025 per the arXiv Sharpe. The higher yields are in alts and on DEXs, where tail, venue and liquidity risk are higher.
- Capacity on BTC/ETH majors is large, but crowding has compressed returns. This is a clear example of a **decayed** edge.

### Gaps
- No primary, audited series of realized net returns by year (2020-2026) for a standard BTC cash-and-carry was found. Figures conflict across sources.

## Other features in scope: spot-perp basis, volatility regimes, seasonality, on-chain flows, social sentiment

### Takeaway
Seasonality is robust for **volatility and volume** (peak at ~16 UTC, low at ~5 UTC; bursts at the top of the hour and at 5-minute marks; quieter weekends), not for returns. On-chain exchange net inflows predict **volatility** better than returns intraday; the exceptions are a USDT inflow → BTC/ETH up effect and ETH inflows → ETH down. Twitter sentiment is a weak-to-moderate hourly predictor in in-sample studies. The basis is mean-reverting, and the carry level predicts crash risk.

### Cited Findings
- Hansen, Kim & Kimbrough, "Periodicity in Cryptocurrency Volatility and Liquidity" (J. Financial Econometrics 2024; Binance and Coinbase Pro): volatility peaks around **16 UTC** and is lowest near **5 UTC**. Volatility bursts at the **top of the hour**, with smaller bursts at :15, :30 and :45 and every 5 minutes (BTC/USDT Binance, Oct 2020-May 2021). Weekends are quieter, Saturday most of all. The patterns **intensified over time**, likely linked to algorithmic trading and **funding times**. Similar within-hour patterns appear in the S&P 500. — [arXiv 2109.12142](https://arxiv.org/pdf/2109.12142); [Quantpedia summary](https://quantpedia.com/periodicity-in-cryptocurrencies-recurrent-patterns-in-volatility-and-volume/)
- Intraday on-chain flows (arXiv 2411.06327; 2017-2023): BTC net inflows **lack predictive power for BTC returns** except at 4h, but are **negatively associated with volatility** at 1, 2, 3, 4 and 6h. **USDT net inflows to exchanges positively predict BTC and ETH returns** at multiple intervals. ETH net inflows negatively predict ETH returns and volatility at all intraday intervals (IS regressions; OOS not verified). — [arXiv](https://arxiv.org/abs/2411.06327)
- Daily net BTC exchange flow has a Pearson correlation of 0.47 with daily BTC volatility. A Synthesizer Transformer using whale-alert and CryptoQuant data forecasts volatility spikes. — [arXiv 2211.08281](https://arxiv.org/pdf/2211.08281); related Q-learning work in [ResearchGate](https://www.researchgate.net/publication/374099900_Forecasting_Bitcoin_Volatility_through_On-Chain_and_Whale-Alert_Tweet_Analysis_using_the_Q-Learning_Algorithm)
- Gao, Huang & Wang (2021), "Financial Twitter Sentiment on Bitcoin Return and High-Frequency Volatility" (hourly): **bullish** finance-tweet sentiment foreshadows higher BTC return and volatility, while bearish/neutral sentiment has no consistent effect. Sources disagree on horizon: the abstract says over 24 hours, a secondary summary says 2/6/12h but not 24h. — [Virtual Economics](https://virtual-economics.eu/index.php/VE/article/view/101); [ResearchGate](https://www.researchgate.net/publication/353401947_Financial_Twitter_Sentiment_on_Bitcoin_Return_and_High-Frequency_Volatility)
- Twitter sentiment predicts returns of BTC, BCH and LTC among the nine largest coins. — [ScienceDirect (J. Behav. Exp. Finance)](https://www.sciencedirect.com/science/article/abs/pii/S104244312030072X); tweet sentiment and attention predictive power for BTC — [ScienceDirect (IREF 2022)](https://www.sciencedirect.com/science/article/abs/pii/S1059056022000375)
- BTC volatility forecasting with 30M+ tweets and 15-min prices; "who" posts matters more than "what". — [arXiv 2110.14317](https://arxiv.org/pdf/2110.14317)
- Bull vs bear on-chain and Twitter effects on BTC volatility. — [PMC 10773860](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10773860/)
- Volatility regimes: intraday momentum/reversal depends on jumps, FOMC days and liquidity (Wen et al.). — [ResearchGate](https://www.researchgate.net/publication/361202793_Intraday_return_predictability_in_the_cryptocurrency_markets_momentum_reversal_or_both); realized-semivariance-based TS momentum/reversal — [ScienceDirect, J. Empirical Finance 2023](https://www.sciencedirect.com/science/article/abs/pii/S0927539823000245)
- Basis: perp basis is mean-reverting under linear funding (theory), and the carry level is linked to crash risk (BIS). — [ResearchGate](https://www.researchgate.net/publication/401892113_Funding_Rate_Mechanism_in_Perpetual_Futures); [BIS WP 1087](https://bis.org/publ/work1087.htm)

### Inferences
- Time-of-day/day-of-week and minute-of-hour features are cheap and robust for **volatility/spread/volume** prediction (position sizing, execution timing, stop sizing), and are linked to funding timestamps (00/08/16 UTC on Binance). They are not directional alpha.
- On-chain flows: their latency (block confirmation plus labeling) makes them unsuitable below 1h. Stablecoin exchange inflows are the most promising directional on-chain feature at 1-6h, but only IS evidence was found.
- Sentiment evidence is mostly 2015-2021 IS. Twitter/X API access and the data-generating process have changed since 2023, so decay is likely but unmeasured.

### Gaps
- No OOS, net-of-cost test of sentiment, on-chain or seasonality-driven *return* strategies at 5-min to 1-day horizons was found.
- No quantitative evidence on whether the hourly volatility patterns kept intensifying after 2021.
