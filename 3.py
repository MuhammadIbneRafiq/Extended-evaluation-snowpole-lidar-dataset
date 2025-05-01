#!/usr/bin/env python3
import math
from scipy.stats import norm

def main():
    # Given parameters:
    # Historical (1995) mean and assumed standard deviation
    mu_old = 3.68    # population mean in 1995 (assumed to be the true mean under H0)
    sigma = 13       # population standard deviation (assumed same for both years)
    n = 50           # sample size in 2022
    xbar_2022 = 10.666  # observed sample mean in 2022
    
    # For a 95% probability interval, the critical z-value is:
    alpha = 0.05
    z = norm.ppf(1 - alpha/2)  # approximately 1.96
    
    # Calculate the standard error of the sample mean:
    se = sigma / math.sqrt(n)
    
    # -------------------------------
    # 3a & 3b: Construct the probability interval (a, b) 
    # under the assumption that μ = 3.68.
    a = mu_old - z * se  # lower bound
    b = mu_old + z * se  # upper bound
    
    # -------------------------------
    # 3c: Compute P(X̄ > 10.666) assuming μ = 3.68.
    # Under H0, X̄ ~ N(3.68, (sigma/sqrt(n))^2)
    p_value = 1 - norm.cdf(xbar_2022, loc=mu_old, scale=se)
    
    # -------------------------------
    # 3e & 3f: Construct a 95% confidence interval for μ
    # when μ is unknown, using the 2022 sample mean.
    s = xbar_2022 - z * se  # lower limit of confidence interval
    t = xbar_2022 + z * se  # upper limit of confidence interval
    
    # Print the results:
    print("Assuming μ = 3.68 (historical mean) with σ = 13 and n = 50:")
    print(f"3a. a = {a:.3f}")
    print(f"3b. b = {b:.3f}")
    print(f"--> 95% probability interval for X̄: ({a:.3f}, {b:.3f})")
    print()
    
    print("3c. Probability that the sample mean exceeds 10.666:")
    print(f"P(X̄ > 10.666) = {p_value:.6f}")
    print()
    
    print("Interpretation:")
    print("If the true mean were still 3.68, 95% of samples of size 50 would have a sample mean")
    print(f"between {a:.3f} and {b:.3f}. However, the observed sample mean of 10.666 is far above this range,")
    print("with an extremely low probability (~0.007%) of occurring if μ were truly 3.68. This suggests that")
    print("the mean daily temperature has likely increased.")
    print()
    
    print("Now, constructing a 95% confidence interval for μ using the 2022 sample:")
    print(f"3e. s = {s:.3f}")
    print(f"3f. t = {t:.3f}")
    print(f"--> 95% Confidence Interval for μ: ({s:.3f}, {t:.3f})")
    
if __name__ == "__main__":
    main()
