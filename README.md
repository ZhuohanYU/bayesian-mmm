# bayesian-mmm

Small project to cover how to play with Bayesian MMM Model

Typical Preprocessing Steps for Bayesian MMM

Step	Why It's Needed	How to Do It
1. Feature scaling	Prevents numerical instability in sampling	Standardize media spend (e.g., StandardScaler)
2. Encode categorical variables	Needed for hierarchical effects like region/product	Convert to index (pd.Categorical(...).codes)
3. Time ordering	Adstock relies on sequential consistency	Sort by time before applying adstock
4. Impute missing values	MCMC breaks if you have NaNs	Interpolate, zero-fill, or impute
5. Lagged/spread media	For adstock + carryover	Apply adstock or geometric decay transform
6. Normalize spend or sales	Optional, but helps with priors	Especially for hierarchical shrinkage
7. Binning/labeling	Useful for oversampling or fairness	E.g., high/low_sales, region_type
![image](https://github.com/user-attachments/assets/014a580e-528b-4008-b7ca-5c16752c7962)




you can use your own data for testing or the code will generate small sample for you in the function

the inbalanced data will be upsampled automatically to match with the major group numbers

it will also give you the evaluation report in the output
