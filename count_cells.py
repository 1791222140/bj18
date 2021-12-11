from empiricaldist import Cdf
import pymc3 as pm



total_squares = 25
squares_counted = 5
yeast_counted = 49
billion = 1e9


with pm.Model() as model:
    yeast_conc = pm.Normal("yeast conc", 
                           mu=2 * billion, sd=0.4 * billion)

    shaker1_vol = pm.Normal("shaker1 vol", 
                               mu=9.0, sd=0.05)
    shaker2_vol = pm.Normal("shaker2 vol", 
                               mu=9.0, sd=0.05)
    shaker3_vol = pm.Normal("shaker3 vol", 
                               mu=9.0, sd=0.05)

    yeast_slurry_vol = pm.Normal("yeast slurry vol",
                                    mu=1.0, sd=0.01)
    shaker1_to_shaker2_vol = pm.Normal("shaker1 to shaker2",
                                    mu=1.0, sd=0.01)
    shaker2_to_shaker3_vol = pm.Normal("shaker2 to shaker3",
                                    mu=1.0, sd=0.01)

    dilution_shaker1 = (yeast_slurry_vol / 
                        (yeast_slurry_vol + shaker1_vol))
    dilution_shaker2 = (shaker1_to_shaker2_vol / 
                        (shaker1_to_shaker2_vol + shaker2_vol))
    dilution_shaker3 = (shaker2_to_shaker3_vol / 
                        (shaker2_to_shaker3_vol + shaker3_vol))
    
    final_dilution = (dilution_shaker1 * 
                      dilution_shaker2 * 
                      dilution_shaker3)

    chamber_vol = pm.Gamma("chamber_vol", 
                           mu=0.0001, sd=0.0001 / 20)

    yeast_in_chamber = pm.Poisson("yeast in chamber", 
        mu=yeast_conc * final_dilution * chamber_vol)

    count = pm.Binomial("count", 
                        n=yeast_in_chamber, 
                        p=squares_counted/total_squares,
                        observed=yeast_counted)
    options = dict(return_inferencedata=False)

    # trace = pm.sample(1000, **options)
    # posterior_sample = trace['yeast conc'] / billion
    # cdf_pymc = Cdf.from_seq(posterior_sample)
    # print(cdf_pymc.mean(), cdf_pymc.credible_interval(0.9))