# time-series-notes

# Regression
For simple linear regression model (popluation) $$E(Y)=\beta_{0}+\beta_{1}X$$ <BR>
We can then estimate $E(Y)$ by $\widehat{Y}$ from sample regression equation <br> $$\widehat{Y}=b_{0}+b_{1}X$$


Using **least squares method (LSM)**, we can estimate $\beta_{0}$ and $\beta_{1}$ by $b_{0}$ and $b_{1}$.<br>
We shall find $b_{0}$ and $b_{1}$ so that the **sum of the squares of the errors (residuals) SSE** is a minimum. <br>

<br>

**Gauss-Markov Theorem** <br>
Under the conditions of the regression model, the least squares estimators $b_{0}$ and $b_{1}$ are unbiased estimators, and have minimum variance among all unbiased estimators. <br>


<br>

**Types of Data** <br>
* time series data
* cross-sectional data
* panel data

<br>

**Classical linear regression model CLRM** <br>

assumptions:
* $Var(u_{t})=\sigma^{2}<\infty$
* $Cov(u_{t}, u_{j})=0$ for $i\neq j$
* $u_{t} \sim N(0, \sigma^{2}$)

$\sigma^{2}$ is constant, this is known as homoscedasticity. <br>
$\sigma^{2}$ is not  constant, this is known as heteroscedastic. <br>
GQ(Goldfeld-Quandt) Test and and White's Test can detect heteroscedasticity. <br>
GQ Test Null hypothesis H0: the error terms are homoscedastic.<br>
If the form of the heteroscedasticity is known, we should use (generalised least squares)GLS ((weighted least squares)WLS) instead of OLS.<br>
Deal with the heteroscedasticity: 1) WLS 2) transforming data into logs 3)use white's correction <br>
For heteroscedastic, OLS estimation still unbaiased, but they are no longer BLUE <br>

<br>

**violating assumption 2 (autocorrelation)** <br>
**Positive Autocorrelation** is indicated by a **cyclical residual plot** over time.<br>
**Negative autocorrelation** is indicated by an alternating pattern where the residuals cross the time axis more frequently than if they were distributed randomly (Graph: PPT1 Page 20, 21)<br>
using **Durbin-Watson (DW) test** to detect autocorrelation. null hypothesis = no autocorrelation. (Graph: PPT1 Page 23).<br>
can also use Breusch-Godfrey Test to detect autocorrelation. null hypothesis = no autocorrelation. <br>
violating assumption 2: still unbiased, but not BLUE. MSE may underestimate.

<br>

**violating assumption 3 (not normal distributed)** <br>
Bera Jarque test can detect whether the data is normal distributed. NULL hypothesis = normal.

<br>


# Univariate Time-Series Modelling and Forecasting

## stochastic processing
* **A Strictly Stationary Process <br>**
the distribution of its values remains the same as time progresses 

* **A Weakly/Covariance Stationary Process <br>**
a stationary process should have a constant
mean, a constant variance and a constant autocovariance structure,
respectively.<BR>

    * $ E(y_{t})=\mu $
    * $ E(y_{t} - \mu)(y_{t} - \mu) = \sigma_{}^{2} <  \infty $
    * $ E(y_{t_{1}} - \mu)(y_{t_{2}} - \mu) = \gamma_{t_{2}-t_{1}} $ <br>
    
    if $\tau_{s} = \gamma_{s} / \gamma_{0}$ is plotted against s=0,1,2..., a graph known as the **autocorrelation function (acf)** or correlogram is obtained.


* **A White Noise Process <br>**
a white noise process is one with no discernible
structure.

    * $ E(y_{t})=\mu $
    * $ var(y_{t}) = \sigma_{}^{2}  $
    * $ \gamma_{t-r} = \sigma_{}^{2}   $ if t=r;  $ \gamma_{t-r} = 0   $ otherwise

    thus a white noise process has constant mean and variance, and zero autocovariances, except at lag zero. <br>
    if we assume $\mu=0$ and $y_{t}$ is normally distributed. Meanwhile,  the sample autocorrelation coefficients follows  $$ \widetilde{\tau_{s}} \sim approx. N(0,1/T)$$ where T is the sample size. <br>
    This result can be used to conduct significance tests for the autocorrelation coefficients by constructing a non-rejection region. <br>
    e.g. a 95% non-rejection region would be given by $ \pm 1.96 \times \frac{1}{\sqrt{T}} $ <br>

    
    Box–Pierce and Ljung–Box test<br>
    null hypothesis: for a given lag, $\tau_{0}=...=\tau_{t}=0$<br>
    Ljung–Box test is better, because Box–Pierce has poor small sample properities. 


<br>

## time series models
### MA
MA is a simple extension of **white noise series**. You can also treat it as **an infinite-order AR model with some parameter constraints**. 


$$ MA(1):  y_{t} =  \mu +  \theta _{1} u_{t-1} + u_{t} $$ 
$ MA(p):  y_{t} =  \mu + \sum_{i=1}^{q} \theta _{i}  u_{t-i} + u_{t}
= \mu + \sum_{i=1}^{q} \theta _{i} L_{}^{i} u_{t} + u_{t} $ or as $$ MA(p): y_{t} = \mu +\theta(L)u_{t} $$ where $ \theta(L)=1+\theta_{1}L+ \theta_{2}L^{2}+...+\theta_{q}L^{q}$.


The property of MA 

* $ E(y_{t})=\mu $
* $ var(y_{t}) = \gamma_{0} = (1+\sum_{i=1}^{q}\theta_{i}^{2})\sigma_{}^{2} $
* covariance $ \gamma_{s} = (\theta_{s}+\theta_{s+1}\theta_{1}+\theta_{s+2}\theta_{2}+...+\theta_{q}\theta_{q-s})\sigma_{}^{2}$ for s=1,2,3,...,q; $ \gamma_{s}=0$ for s>q. (p257)

<br>

**Can caculate $\gamma$ and $\tau$** 

<br>

## AR
AR model predicts future behavior based on past behavior.


$$ AR(1): y_{t} = \mu + \varphi_{1}y_{t-1} + u_{t} $$
$ AR(p):  y_{t} =  \mu + \sum_{i=1}^{q} \varphi _{i}  y_{t-i} + u_{t}
= \mu + \sum_{i=1}^{q} \varphi _{i} L_{}^{i} y_{t-i} + u_{t} $ 

or as $$  AR(p): \varphi (L)y_{t} = \mu + u_{t} $$ where $ \varphi (L)=(1-\varphi_{1}L -\varphi_{2}L^{2} - ... - \varphi_{p}L^{p}  ) $ <BR></BR>

**The stationary condition for an AR(p) model** <br>
For $ AR(p): \varphi (L)y_{t} = \mu + u_{t} $, we assume $\mu=0$, than $ y_{t} = \varphi (L)^{-1}u_{t} $. If $\varphi (L)^{-1}$ converges to zero, 
$$ \varphi (L)^{-1} = 1-\varphi_{1}z - \varphi_{2}z^{2}-...- \varphi_{p}z^{p}=0$$
AR(p) meets stationary condition.



e.g. $y_{t}=3y_{t-1}-2.75y_{t-2}+0.75y_{t-3}+y_{t}$ <br>
$\varphi_{1}=3,\varphi_{2}=-2.75,\varphi_{3}=0.75$<br>
$1-3z+2.75z^{2}-0.75z^{3}=0$ has the roots: 1,2/3,2. <br>
only 2 lies outside the unit circle, so non-stationary.



<br>

**Wold’s Decomposition Theorem**<br>
Any stationary AR(p) with no constant and no other terms can be expressed as MA(∞). <br>
e.g. For a stationary AR(1), $ y_{t} = \varphi_{1}y_{t-1} +u_{t}$, can be expressed as an MA(∞).
$$ y_{t}=(1-\varphi_{1}L)^{-1}u_{t} $$
$$ y_{t}=(1+\varphi_{1}L + +\varphi_{1}^{2}L^{2}+...)u_{t} $$
or
$$ y_{t}=u_{t} +\varphi_{1}u_{t-1} + +\varphi_{1}^{2}u_{t-2}+\varphi_{1}^{3}u_{t-3}+... $$
* covariance $ \gamma_{s} $ (P337, example6.2 Q2) <br>

Using Wold’s Decomposition Theorem to calculate the mean, var, cor, and acf of an stationary AR. (example 6.4)

<br>

The property of AR(1)

* $ E(y_{t})=\frac{\mu}{1-\varphi_{1}-\varphi_{2}-...--\varphi_{p}} $ AR(p)
* $ var(y_{t}) = \gamma_{0}=\frac{\sigma_{}^{2}}{1-\varphi^{2}_{1}} $
* covariance $ \gamma_{s}= \frac{\varphi^{s}_{1}\sigma_{}^{2}}{1-\varphi^{2}_{1}}$

<br>

## The Partial Autocorrelation Function
* acf: the correlation between $y_{t}$ and all the observations at previous time spot.
* pacf: direction connection

$ \tau_{11}=\tau_{1} $ , $ \tau_{22}=(\tau_{2}-\tau_{1}^{2})/(1-\tau_{1}^{2}) $ <br>
For AR(2), $ \tau_{11}\neq 0 $ and $ \tau_{22}\neq 0 $, but $ \tau_{33} = \tau_{44}=...=0 $ <br>
For a Invertibility MA(q), it can be invert to AR(∞), thus all the partial autocorrelation is not equal to zero.

<br>    

## ARMA
$$ ARMA(p,q): \varphi(L)y_{t} = \mu + \theta(L)u_{t} $$
where $ \varphi (L)=(1-\varphi_{1}L -\varphi_{2}L^{2} - ... - \varphi_{p}L^{p}  ) $ and  $ \theta(L)=1+\theta_{1}L+ \theta_{2}L^{2}+...+\theta_{q}L^{q}$

* AR 
    * acf: geometrically decaying 
    * pacf: p significant lags                                                                                                        
* MA
    * acf: q significant lags
    * pacf: geometrically decaying
* ARMA
    * acf: geometrically decaying
    * pacf: geometrically decaying

Identifying AR and MA using ACF and PACF Plots

<br>

## Box–Jenkins Approach
* Identification <BR>
AR or MA or ARMA
ACF and PACF Plots → p, q  <br>
* Estimation <BR> 
AIC: more efficient <BR>
SBIC: parsimony <BR>
HQIC

* Diagnostic checking <BR>
residual analysis?: use Q-stats to check wheather the residual is white noise or not!

## Forecasing
+++++++++++++++++++++++


# Modelling long-run relationship in finance

## two types of non-stationary

* stochastic non-stationarity.<br>
random walk with drift <br>
$y_{t} = \mu + y_{t-1} +u_{t}$<br>
remove non-stationarity: by de-trending
* deterministic non-stationarity<br>
trend stationary  <br>
$y_{t} = \alpha + \beta t +u_{t}$<br>
remove non-stationarity: by diff

## unit roots test

* ADF test.<br>
null pyhothesis: series contains a unit root<br>
AD test is only valid if $u_{t}$ is white noise, so we usually use ADF test. <br>

* PP test <br>
null pyhothesis: series contains a unit root<br>
usually give same conclusion as the ADF test.<br>

* KPSS test <br>
null pyhothesis: series is stationary.<br>

## structural break

* Zivot and Andrews approach

## Cointegration
* I(1)+I(1) -> I(1)
* I(2)+I(1) -> I(2) <br>

+++++++++++++++++++++++

# Modelling Volatility

## ARCH: Autoregressive Conditionally Heteroscedastic
ARCH(q): $u_{t}\sim N(0, \sigma^{2}_{t})$

$$\sigma^{2}_{t} = \alpha_{0}+\alpha_{1}u^{2}_{t-1}+\alpha_{2}u^{2}_{t-2} +...++\alpha_{q}u^{2}_{t-q}$$ 
or, <br>
$u_{t}=v_{t}\sigma_{t}$, $v_{t} \sim N(0,1)$ and  $$\sigma^{2}_{t} = \alpha_{0}+\alpha_{1}u^{2}_{t-1}+\alpha_{2}u^{2}_{t-2} +...++\alpha_{q}u^{2}_{t-q}   $$
$\sigma^{2}_{t}$ usually be called $h_{t}$

## ARCH: testing ARCH effect
null hypothesis: $\gamma_{1}=\gamma_{2}=...=\gamma_{q}=0$ <br>
problems with ARCH model: 
* no way to decide on q
* q might be very large
* Non-negativity constraints might be violated. (on ARCH model, $\alpha_{i}>0$ for all i< q)

<br>

## GARCH
GARCH(1,1), like an ARMA(1,1) for the variance equation. $$\sigma^{2}_{t} = \alpha_{0}+\alpha_{1}u^{2}_{t-1}+\beta \sigma^{2}_{t-1}$$ 

GARCH(p,q) $$\sigma^{2}_{t} = \alpha_{0}+ \sum_{i=1}^{q}\alpha_{i}u^{2}_{t-i}+ \sum_{j=1}^{p} \beta_{j} \sigma^{2}_{t-j}$$ 

usually GARCH(1,1) will sufficient to caputure the violatility of the data. <br>

why GARCH is better than ARCH?

* more parsimonious - avoids overfitting
* less likely to breech non-negativity constraints

+++++++++++++++++++++++
