# ML trading backtest pitfalls and realistic performance benchmarks (crypto / short-horizon)

Research date: 2026-10-21. Many primary-source domains (pwc.com, cxoadvisory, quantpedia, carmamaths, bnnbloomberg, mql5) could not be fetched from this environment, so several figures come from search-result snippets of the primary page. Each one is flagged where that applies.

## 1. Methodological pitfalls: leakage, overlapping labels, purged/embargoed CV, CPCV, deflated Sharpe, PBO, multiple-testing haircuts

### Takeaway
The best-documented ways a backtest fakes profitability are (1) information leaking across train/test splits, made worse by overlapping, path-dependent labels such as triple-barrier labels, and (2) selection bias from trying many configurations. Lopez de Prado's purging and embargo deal with the first. Deflated Sharpe, PBO/CSCV, CPCV and Harvey-Liu haircuts deal with the second. With only a few dozen trials on a few years of data, a Sharpe of about 1 can appear when the true Sharpe is 0.

### Cited Findings
- **Purged CV, how it works.** Purged k-fold CV was developed by Lopez de Prado to prevent look-ahead bias when labels depend on future events. Purging drops training samples whose label window overlaps a test sample's window. The embargo "additionally drops a buffer of training observations immediately after each test block to absorb serial correlation that outlives the labels." The primary reference is *Advances in Financial Machine Learning* (2018), ch. 7, "Cross-Validation in Finance". — [Wikipedia: Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation); [LuxAlgo explainer](https://www.luxalgo.com/library/concept/purged-cross-validation/); [TradingInterview lesson](https://www.tradinginterview.com/courses/machine-learning/lessons/cross-validation-for-financial-data-purging-and-embargoing/)
- **Triple-barrier labels and leakage.** The triple-barrier method produces variable-length label intervals that end at the first barrier touch or at the vertical (time) barrier. This is exactly why purging is needed: a training label that ends inside the test window leaks test information into training. — [LuxAlgo explainer](https://www.luxalgo.com/library/concept/purged-cross-validation/); [Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)
- **CPCV.** Combinatorial purged CV applies purging and embargo across many combinations of train/test blocks. It produces many backtest paths, and so a distribution of performance, instead of one easily overfit walk-forward path. — [fynance docs: combinatorial_purged_cv](https://fynance.readthedocs.io/en/latest/generated/fynance.data.combinatorial_purged_cv.html); [Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)
- **PBO / CSCV.** Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest Overfitting", *Journal of Computational Finance* 20(4), 2017 (SSRN drafts from 2013). The paper argues that standard hold-out is unreliable for investment backtests and proposes combinatorially symmetric cross-validation (CSCV). CSCV splits the performance matrix into blocks and enumerates every half/half train/test assignment. PBO is the share of splits in which the in-sample best configuration ranks below the median out of sample. — [SSRN 2326253](https://papers.ssrn.com/abstract=2326253); [Risk.net / J. Comp. Finance](https://www.risk.net/2471206); [LuxAlgo PBO explainer](https://www.luxalgo.com/library/concept/probability-of-backtest-overfitting/)
- **Deflated Sharpe Ratio.** Bailey & Lopez de Prado, *Journal of Portfolio Management*, 2014. The DSR is the probability that the selected strategy's true Sharpe exceeds a benchmark, after correcting for the number of trials and for non-normal returns (skew, kurtosis). PBO tests rank stability without distributional assumptions, and DSR tests magnitude against a luck-based hurdle, so the two complement each other. — [TradingStrategy.ai backtesting docs](https://tradingstrategy.ai/docs/learn/backtesting.html); [LuxAlgo](https://www.luxalgo.com/library/concept/probability-of-backtest-overfitting/)
- **How fast overfitting appears.** Bailey, Borwein, Lopez de Prado & Zhu, "Pseudo-Mathematics and Financial Charlatanism", *Notices of the AMS*, 2014:
  - After trying only 7 configurations, a researcher is expected to find at least one 2-year backtest with an annualized Sharpe above 1 when the expected out-of-sample Sharpe is 0.
  - Testing more than 45 configurations on 5 years of data is "almost guaranteed" to produce an in-sample annualized Sharpe of 1 with an expected out-of-sample Sharpe of 0.
  - The paper defines a Minimum Backtest Length, approximately MinBTL < 2 ln(N) / E[max_N]^2 years. This is a necessary but not sufficient condition.
  - Most analysts do not report how many configurations they tried.
  — [WMU ScholarWorks record](https://scholarworks.wmich.edu/math_pubs/40/); [IAQF talk abstract](https://iaqf.org/event-813217); [ScienceDaily 2014](https://sciencedaily.com/releases/2014/04/140410103005.htm); [paper PDF](https://carmamaths.org/jon/backtest.pdf)
- **Multiple-testing hurdle.** Harvey, Liu & Zhu, "...and the Cross-Section of Expected Returns" (NBER w20592, 2014; *RFS* 2016) counted at least 316 published factors. They argue a new factor needs a t-statistic above 3.0, not 2.0. — [NBER w20592](https://www.nber.org/papers/w20592); [SSRN 2513152](https://papers.ssrn.com/abstract=2513152)
- **Sharpe haircuts.** Harvey & Liu, "Backtesting" (*JPM* 2015) note that the industry habit of cutting backtested Sharpe ratios by 50% is only a rule of thumb. They derive the haircut instead: convert the Sharpe to a t-stat, adjust the p-value for the number of tests (Bonferroni, Holm, BHY), then convert back. The percentage drop from the original to the adjusted Sharpe is the haircut, and it is non-linear. Marginal strategies get cut far more than 50%, while very strong ones get cut less. — [Harvey & Liu, Backtesting (Duke PDF)](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF); [CME-hosted copy](https://www.cmegroup.com/content/dam/cmegroup/education/files/backtesting.pdf); [Man/Oxford-Man summary](https://www.man.com/insights/backtesting)
- **Effect of backtest count in practice.** In the Quantopian cohort, the more backtests a user ran on a strategy, the larger the gap between backtest and out-of-sample performance. — [Wiecki et al., ResearchGate](https://www.researchgate.net/publication/307553701_All_That_Glitters_Is_Not_Gold_Comparing_Backtest_and_Out-of-Sample_Performance_on_a_Large_Cohort_of_Trading_Algorithms)

### Inferences
- For an ML intraday crypto bot, these are the concrete prevention steps:
  - Use purged and embargoed (ideally combinatorial) CV, with the embargo at least as long as the label horizon plus feature look-back.
  - Fit scalers and feature normalizers on training folds only.
  - Use event timestamps, not bar close, for when a feature is "known".
  - Log every configuration and hyper-parameter tried, and report DSR and PBO for the final choice.
  - Hold out a final untouched period, preferably a different market regime.
  - Model fees, spread, slippage, funding, latency and fill probability for limit orders. At scalping horizons these costs often exceed the gross edge.
- Grid search, Optuna-style tuning, feature selection and architecture changes all count as "trials" N. With hundreds of them, a backtest Sharpe of 1-2 on 1-2 years of data is weak evidence by itself.
- Leakage sources specific to short-horizon crypto that should be checked, from general knowledge rather than a source:
  - Resampling candles with right-closed or right-labelled bars.
  - Using the close price as the fill price.
  - Survivorship in the universe of listed coins, since delisted tokens are missing.
  - Exchange data revisions.
  - Funding rates published after the fact.

### Gaps
- I could not open the full DSR or Harvey-Liu PDFs to quote worked numeric haircut examples, such as the haircut at a specific t-stat and number of tests.
- I found no published study that measures leakage-induced inflation for crypto intraday ML specifically.

## 2. Backtest-to-live degradation (Quantopian study; McLean & Pontiff post-publication decay)

### Takeaway
Backtest Sharpe barely predicts live Sharpe (R² < 0.025 across 888 Quantopian algorithms). Published anomalies lose about 26% of their return out of sample and about 58% after publication. A 50% or larger haircut on backtest performance is a reasonable default prior, and for heavily tuned retail and ML strategies the realistic expectation is often roughly zero live alpha.

### Cited Findings
- **The Quantopian study.** Wiecki, Campbell, Lent & Stauth (Quantopian), "All That Glitters Is Not Gold: Comparing Backtest and Out-of-Sample Performance on a Large Cohort of Trading Algorithms" (SSRN 2745220, 2016; *Journal of Investing*).
  - Sample: 888 algorithms with at least 6 months of out-of-sample performance.
  - Backtest Sharpe ratio "offer[s] little value in predicting out of sample performance (R² < 0.025)". That implies a correlation of about 0.16 or less, which is my arithmetic.
  - IS/OOS correlations were very weak for Sharpe, information ratio and alpha.
  - Volatility, max drawdown and hedging did have significant predictive value.
  - More backtesting meant a larger IS-OOS gap.
  - Non-linear ML classifiers trained on backtest-behaviour features reached R² = 0.17 on hold-out data, and a portfolio built on them beat one built from the top backtest-Sharpe algorithms.
  — [ResearchGate](https://www.researchgate.net/publication/307553701_All_That_Glitters_Is_Not_Gold_Comparing_Backtest_and_Out-of-Sample_Performance_on_a_Large_Cohort_of_Trading_Algorithms); [Semantic Scholar](https://www.semanticscholar.org/paper/All-That-Glitters-Is-Not-Gold:-Comparing-Backtest-a-Wiecki-Campbell/b028c2d0b2145e85d543fb21ad3fcf896c932ab5); [Quantpedia summary](https://quantpedia.com/quantopians-academic-paper-about-in-vs-out-of-sample-performance-of-trading-alg/)
- **Quantopian study periods.** According to a secondary summary, the in-sample period ran from 2010 to deployment in early 2015, and the out-of-sample period ran from June 2015 to February 2016. — [CXO Advisory summary](https://www.cxoadvisory.com/big-ideas/in-sample-vs-out-of-sample-performance-of-888-trading-strategies) (not fetched; snippet only)
- **Post-publication decay.** McLean & Pontiff, "Does Academic Research Destroy Stock Return Predictability?", *Journal of Finance* 71(1), 2016. Across 97 published predictors, portfolio returns are 26% lower out of sample (an upper bound on data-mining effects) and 58% lower after publication. That leaves about 32% attributed to publication-informed arbitrage. The working-paper version reported 95 predictors with 25% and 56%. — [Gwern-hosted PDF](https://Www.Gwern.net/doc/economics/2016-mclean.pdf); [AFA JF vol. 71 issue 1](https://afajof.org/issue/volume-71-issue-1/); [Counterpoint copy](https://counterpointfunds.com/wp-content/uploads/2017/07/PredictabilityMcleanPontiff.pdf)
- **When the decay is largest.** A secondary summary says the largest degradation relative to in-sample comes 3-4 years after publication. — [CXO Advisory: effects of market adaptation](https://www.cxoadvisory.com/big-ideas/effects-of-market-adaptation)
- **Haircut rule of thumb.** Practitioners commonly haircut backtested Sharpe ratios by 50%. Harvey & Liu show the right haircut depends on the number of tests and is often larger for marginal results. — [Harvey & Liu, Backtesting](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF)

### Inferences
- McLean-Pontiff studies academic equity anomalies that were each tested fairly carefully. Retail or self-built ML crypto strategies with heavy tuning and no trial accounting probably degrade more than 26-58%. Wiecki's near-zero R² suggests that for a typical tuned strategy, the backtest Sharpe carries almost no information about live Sharpe.
- Wiecki found volatility and drawdown more predictive than Sharpe. This supports checking whether the live risk profile (volatility, drawdown) matches the backtest as an early warning sign.

### Gaps
- I could not retrieve the exact mean or median backtest versus OOS Sharpe values from Wiecki et al., because the PDF hosts were unreachable.
- I found no large-sample study of crypto bot backtest-vs-live degradation, for example from 3Commas, Hummingbot or Numerai-style data.

## 3. What crypto hedge funds / crypto quant and market-neutral funds actually achieve (2020-2026)

### Takeaway
Crypto hedge fund returns are dominated by crypto beta: index volatility is around 60%, and funds lost about 50% in 2022. Market-neutral crypto funds earned high-single to mid-double-digit returns in strong years (median +26% in 2021; roughly +18% to +35% in 2024 depending on the source) and slightly negative returns in early 2025. I found no reliable published Sharpe figures for market-neutral crypto indices. For comparison, a mature systematic index like SG Trend runs at a Sharpe of about 0.5 over 26 years.

### Cited Findings
- **PwC/AIMA/CoinShares 2022 report (covering 2021).**
  - Median crypto hedge fund return: +63.4% in 2021, versus +127.55% in 2020.
  - Median by strategy in 2021: discretionary long/short +199%, discretionary long-only +176%, quantitative long-only +109%, quantitative long/short +66%, market neutral +26%.
  — [AIMA press release 2022](https://www.aima.org/article/press-release-more-than-a-third-of-traditional-hedge-funds-now-invest-in-digital-assets-nearly-double-a-year-ago-pwc-global-crypto-hedge-fund-report-2022.html); [PwC press release 2022](https://pwc.com/gx/en/news-room/press-releases/2022/pwc-global-crypto-hedge-fund-report-2022.html)
- **PwC/AIMA/CoinShares 2023 report (12 July 2023).** The share of traditional hedge funds investing in crypto fell from 37% to 29%, while average allocation among those invested rose from 4% to 7%. I could not retrieve the strategy-level 2022 returns. — [PwC Türkiye page, 2023 report](https://www.pwc.com.tr/pwc-global-crypto-hedge-fund-report-2023); [report PDF (unfetchable)](https://www.pwc.com/hu/hu/iparagak/assets/5th-annual-global-crypto-hedge-fund-report-july-2023.pdf)
- **VisionTrack 2022-2023.** According to Galaxy VisionTrack data as reported by Benzinga, crypto hedge funds averaged about −52% in 2022 and +44% in 2023. — [Benzinga](https://www.benzinga.com/amp/content/36406473)
- **AIMA/PwC 2025 survey (published November 2025).** 55% of traditional hedge funds held crypto-related assets, up from 47%, based on 122 respondents surveyed in H1 2025. Most invested less than 2% of AUM. — [Hedgeweek](https://www.hedgeweek.com/more-than-half-of-hedge-funds-now-hold-crypto-assets-finds-aima-survey/); [Investment Executive](https://www.investmentexecutive.com/news/research-and-markets/hedge-funds-creep-further-into-crypto-aima/)
- **HFR Cryptocurrency Index.**
  - Five years to February 2025: annualized return 51.4%, annualized volatility 59.5%. Bitcoin rose about 470% over the same period.
  - Implied return/vol ratio is about 0.86 before the risk-free rate (my arithmetic).
  - In March 2025 HFR added 11 sub-indices, including market-neutral, quantitative, arbitrage and volatility.
  — [The Full FX: HFR expands crypto data](https://thefullfx.com/hfr-expands-crypto-fund-performance-data/)
- **Galaxy VisionTrack 2024.** Composite +54.62%, Fundamental +50.86%, Quantitative +65.58%, Market Neutral +34.95%, Bitcoin +121.09%. 2025 YTD through April: Composite −12.85%, Market Neutral −3.55%, BTC +0.74%. — [Galaxy VisionTrack indices](https://visiontrack.galaxy.com/indices/) (via search snippet)
- **Conflicting 2024 market-neutral figure.** Bloomberg reported the VisionTrack Market Neutral index at about +18.5% for 2024 ("crypto hedge funds had great 2024 but failed to top bitcoin", Jan 2025). This conflicts with the +34.95% above, and I could not resolve the difference. — [BNN Bloomberg](https://bnnbloomberg.ca/markets/crypto/2025/01/15/crypto-hedge-funds-had-great-2024-but-failed-to-top-bitcoin)
- **Crypto funds and Bitcoin beta.** Institutional Investor ran the headline that crypto hedge funds deliver bitcoin-like returns with lower volatility. Eurekahedge's crypto index tracked bitcoin almost in lockstep (2022 commentary). — [Institutional Investor](https://www.institutionalinvestor.com/article/2bstm0ttbd36wmby0uy2o/portfolio/crypto-hedge-funds-deliver-bitcoin-like-returns-without-the-volatility)
- **Low-reliability aggregator figures.** SQ Magazine lists 2025 crypto hedge fund figures: quant 48%, market-neutral 13%, average 36% with 46% volatility, average Sharpe 1.6. These are unsourced. — [sqmagazine](https://sqmagazine.co.uk/crypto-hedge-funds-statistics.md) (UNVERIFIED; do not rely on these)
- **Mature systematic benchmark.**
  - SG Trend Index, December 1999 to June 2026: annualized Sharpe 0.47 on monthly net returns.
  - Replicating trend systems reached 0.47-0.55, and the differences were not significant.
  — [arXiv 2607.19497, "The Science and Practice of Trend-Following Systems"](https://arxiv.org/pdf/2607.19497)
- **Flat stretches are normal at that Sharpe.** CFM notes that trailing 8-month returns near 0% are "not atypical" for the SG CTA index, and that flat performance is common for a modest-Sharpe strategy. — [CFM blog](https://www.cfm.com/is-trend-really-having-a-bad-year/)

### Inferences
- Realistic net live Sharpe ratios, with low confidence on exact numbers:
  - Diversified systematic macro or CTA: about 0.4-0.8.
  - Good multi-strategy or market-neutral funds: about 1-2.
  - Elite HFT or market makers: often higher, but capacity-limited and dependent on infrastructure and fee rebates that a retail bot does not have.
- Crypto market-neutral returns in 2021-2024 were inflated by high funding and basis yields, which carry their own tail risks (exchange failure such as FTX in 2022, depegs). They are not evidence that smooth double-digit returns are free.
- At a true Sharpe of about 1, the share of positive months is roughly Φ(1/√12) ≈ 61%. At Sharpe 2 it is about 72%, and at Sharpe 3 about 81%. This assumes normal i.i.d. monthly returns and is my derivation. More than 90% positive months under normality needs an annual Sharpe of about 4.4 or higher, a level essentially never sustained at scale outside HFT and market making.

### Gaps
- I did not find primary 2022 strategy-level medians from the PwC 2023 report, any PwC/AIMA strategy return data for 2023-2025, or Sharpe ratios for HFR or VisionTrack market-neutral sub-indices.
- I found no sourced statistic on the "share of profitable months" for top systematic funds. The SG Trend positive-month share was not found either.

## 4. Retail trader profitability (day trading, CFD/forex, crypto)

### Takeaway
Large-sample evidence agrees on a single figure range. Roughly 70-97% of retail active traders lose money. Fewer than 1-3% are persistently profitable after costs, and there is little evidence that traders learn their way into profitability.

### Cited Findings
- **ESMA CFD analysis (March 2018).** National regulators' account-level analyses found that 74-89% of retail CFD accounts lose money, with average losses per client of €1,600 to €29,000. This led to the August 2018 restrictions: leverage caps from 30:1 to 2:1, margin close-out and negative-balance protection. — [ESMA press release](https://www.esma.europa.eu/node/84933)
- **Broker disclosures.** Brokers must publish the percentage of retail CFD accounts that lose money, under ESMA Decision 2018/796. — [Exante retail risk warning](https://exante.eu/retail_risk_warning/)
- **Taiwan day traders (Barber, Lee, Liu & Odean).**
  - Data: Taiwan Stock Exchange records, about 1992-2006.
  - Fewer than 3% of day traders "predictably earn future profits net of trading costs", and they account for less than 10% of day-trading volume.
  - About 97% can expect to lose money.
  - Aggregate day-trader performance is negative. Unprofitable traders are more likely to quit, yet more than half of day trading comes from experienced traders with a history of losses.
  — [Barber et al., "Do Day Traders Rationally Learn About Their Ability?" (Berkeley)](https://faculty.haas.berkeley.edu/odean/papers/day%20traders/day%20trading%20and%20learning%20110217.pdf); ["The Cross-Section of Speculator Skill" (Berkeley)](https://faculty.haas.berkeley.edu/odean/papers/day%20traders/Day%20Trading%20Skill%20110523.pdf)
- **"Fewer than 1%" figure.** A secondary summary says fewer than 1% (about 1,000 of 360,000) are consistently profitable. I could not confirm this in the primary paper, so the 3% figure is safer to cite. — [CXO Advisory](https://www.cxoadvisory.com/individual-investing/do-day-traders-make-money/)
- **Brazil futures day traders.** Chague, De-Losso & Giovannetti, "Day Trading for a Living?" (FGV working paper TD 525, 2020; SSRN 3423101), using CVM data.
  - Among 19,646 people who began day trading mini-Ibovespa futures in 2013-2015, they looked at those who persisted at least 300 days.
  - 97% of them lost money, and only 0.4% earned more than a bank teller's wage (about US$54 per day).
  - The best trader earned about US$310 per day.
  - They found no evidence of learning.
  — [FGV repository](https://repositorio.fgv.br/items/e87d04fc-4b4c-4a56-ab5a-1c84c1afde1b); [IDEAS/RePEc](https://ideas.repec.org/p/fgv/eesptd/525.html)
- **Crypto retail (BIS Bulletin No. 69).** Cornelli, Doerr, Frost & Gambacorta, "Crypto shocks and retail losses", 20 February 2023.
  - App data from August 2015 to December 2022 show a majority of crypto app users in nearly all economies lost money on bitcoin.
  - After the Terra/Luna and FTX collapses, large investors sold while small retail investors bought.
  - The modelled median investor who invested $100 per month lost about 48% ($431 of $900) by 15 December 2022, according to secondary coverage of the BIS model.
  — [BIS Bulletin 69](https://www.bis.org/publ/bisbull69.htm); [Ecofin summary of model figures](https://www.ecofinagency.com/finance/1104-44420-bitcoin-retail-investors-lost-47-89-of-invested-funds-on-average-bis-bulletin)

### Inferences
- A retail ML scalping bot competes in the same arena as the losing majority, plus professional market makers with lower fees and latency. The base rate for "this bot is net profitable live" is low, so strong evidence is needed before believing a backtest.

### Gaps
- I found no large-sample regulator or academic statistic specifically on retail crypto derivatives or algo-bot profitability, for example from Binance or Bybit futures. Exchanges do not publish CFD-style loss disclosures.
- No 2024-2026 update of the ESMA aggregate figures was found.

## 5. Is "profitable almost always" (>90% positive months) realistic? What hidden risks produce smooth curves?

### Takeaway
More than 90% positive months is a red flag, not a benchmark. Under normal returns it implies an annual Sharpe above roughly 4.4, while good systematic indices run at Sharpe 0.5-2. In practice, very smooth curves usually come from short-volatility or short-tail payoffs (option selling, carry), averaging down (martingale, DCA, grid bots, no stop-loss), illiquid or stale marking, leakage or overfitting in a backtest, or outright fraud. The Madoff feeder fund had only 7 negative months out of 174.

### Cited Findings
- **Madoff.** Markopolos's 2005 red-flag memo noted that a third-party fund run by Madoff's firm had negative returns in only 7 of 174 months analyzed, about 96% positive months. That pattern "simply doesn't exist in finance". — [NBC News](https://www.nbcnews.com/news/amp/wbna28310980); [Wikipedia: Harry Markopolos](https://en.wikipedia.org/wiki/Harry_Markopolos); [SEC OIG exhibit](https://sec.gov/news/studies/2009/oig-509/exhibit-0400.pdf)
- **Lo's Capital Decimation Partners.** Andrew Lo's hypothetical fund sold out-of-the-money S&P 500 puts with leverage. It produced a stellar, smooth record over 1992-1999 (described as close to 4% per month) and carried an unpredictable catastrophic-loss risk. Lo used it to show that a decade of strong returns cannot reveal tail risk. — [Time: "You Too Can Make 40% A Year"](https://business.time.com/?p=1928); [Econbrowser](https://econbrowser.com/?p=8515)
- **Grid bots.**
  - Grid bots lose when price trends out of their range, and fees and slippage reduce returns. — [KuCoin margin grid bot](https://kucoin.com/support/20915718363417)
  - Below the lower limit, the bot "will remain at maximum asset capacity, holding a full basket of losing inventory with mounting unrealized losses", and its stop-loss is optional. — [TradingView grid scripts](https://it.tradingview.com/scripts/bot)
- **Martingale sizing.** Traders add martingale sizing to DCA and grid bots to average down. This raises exposure during drawdowns, and script authors warn it is "a rather rough tool" with increased risk under leverage. — [Manimama: automated strategy types](https://manimamaeu.pp.ua/types-of-automated-strategies-in-crypto-trading); [TradingView martingale grid bot](https://vn.tradingview.com/scripts/page-17)
- **Margin grids and liquidation.** With margin grids, losses beyond principal trigger forced liquidation and the "funds may be reduced to zero". — [XREX margin grid risks](https://support.xrex.io/en/articles/9556690-what-are-the-risks-and-rewards-of-margin-grid)
- **Flat periods at realistic Sharpe.** At realistic Sharpe ratios, long flat or negative stretches are normal. CFM finds trailing 8-month returns near 0% "not atypical" for the SG CTA index. — [CFM](https://www.cfm.com/is-trend-really-having-a-bad-year/)

### Inferences
- Checks for a smooth backtest or live curve:
  - Win rate versus payoff ratio. A high win rate with a large average loss or a fat left tail suggests short-volatility behaviour.
  - Maximum adverse excursion and peak unrealized loss, not only realized P&L.
  - Leverage and position size over time, to spot averaging down.
  - Whether a stop-loss exists, and what happens in a 30-50% gap or a cascade such as March 2020, May 2021, or the FTX collapse in November 2022.
  - Exchange or counterparty risk.
  - Whether the backtest fills limit orders that would really have been filled.
  - Skewness and kurtosis, which the DSR penalizes.
- A plausible honest target for a good retail intraday crypto ML strategy after costs: a live Sharpe around 1-2 if it has a real edge, about 55-70% positive months, and drawdowns that can exceed 10-30%, with the edge likely to decay. A backtest showing a Sharpe above 3-4 or more than 90% positive months should be presumed leaked, overfit, cost-free, or short-tail until proven otherwise.

### Gaps
- I found no academic study quantifying how often live crypto grid or martingale bots blow up.
- Lo's original paper ("Risk Management for Hedge Funds", *Financial Analysts Journal* 2001) was not fetched, so the exact CDP parameters are unverified.
