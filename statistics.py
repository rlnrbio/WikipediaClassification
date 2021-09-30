# -*- coding: utf-8 -*-
"""
Created on Fri Sep 03 15:15:15 2021

@author: rapha
"""
from dataloading import load_metadata
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
# Script to create some statistics about the dataset and create distribution plots

meta = load_metadata()

# get some border-values for articles (such as longest)
article_length =meta["textlength"]
article_length_good = meta[meta["goodArticle"] == True]["textlength"]
article_length_nogood = meta[meta["goodArticle"] == False]["textlength"]
length_longest_article_good = np.max(article_length_good)
longest_article_good = meta[meta["textlength"] == length_longest_article_good]

# use longest good article as threshold to exclude nogood entries that are
# longer than this article (e.g. lists, logdocuments etc.)
# longest_article_good as of 10.09.2021: Roman Empire: 217366 letters

n_excluded = np.sum(article_length_nogood > length_longest_article_good)

article_length_nogood_upd = article_length_nogood[article_length_nogood <= length_longest_article_good]

# plot histogram of article lengths:
fig, ax = plt.subplots(figsize= (10,8))
ax.hist(article_length_good, color = "green", bins = 80, alpha = 0.6)
ax.axvline(np.mean(article_length_good), color = "green")

# set x-axis label
ax.set_xlabel("Article Size",fontsize=14)
# set y-axis label
ax.set_ylabel("# Good Articles", color = "green", fontsize=14)

ax2=ax.twinx()
ax2.hist(article_length_nogood_upd, color = "red", bins = 80, alpha = 0.6)
ax2.axvline(np.mean(article_length_nogood_upd), color = "red")

ax2.set_ylabel("# NoGood Articles", color = "red", fontsize = 14)


#### Plot using density:
fig, ax = plt.subplots(figsize= (10,8))
ax.hist(article_length_good, color = "green", bins = 80, alpha = 0.6, density = True)
ax.axvline(np.mean(article_length_good), color = "green")

# set x-axis label
ax.set_xlabel("Article Size",fontsize=14)
# set y-axis label
ax.set_ylabel("Density", color = "black", fontsize=14)

#ax=ax.twinx()
ax.hist(article_length_nogood_upd, color = "red", bins = 80, alpha = 0.6, density = True)
ax.axvline(np.mean(article_length_nogood_upd), color = "red")



###### fit gamma distribution to good articles:
fit_alpha, fit_loc, fit_beta=stats.gamma.fit(article_length_good)

x = np.linspace(0, length_longest_article_good, length_longest_article_good)
y = stats.gamma.pdf(x, a=fit_alpha, scale=fit_beta, loc = fit_loc)


# plot good articles with distribution
fig, ax = plt.subplots(figsize= (10,8))
ax.hist(article_length_good, color = "green", bins = 80, alpha = 0.6, density = True)
ax.axvline(np.mean(article_length_good), color = "green")

ax.plot(x, y, color = "black", label = "Gamma distribution,  \nalpha: {}, beta: {}, loc: {}".format(round(fit_alpha, 3), round(fit_beta, 3), round(fit_loc, 3)))

# set x-axis label
ax.set_xlabel("Article Size",fontsize=14)
# set y-axis label
ax.set_ylabel("# Good Articles", color = "green", fontsize=14)
ax.legend()


# plot bad articles with distribution
fig, ax = plt.subplots(figsize= (10,8))
ax.hist(article_length_nogood_upd, color = "red", bins = 80, alpha = 0.6, density = True)
ax.axvline(np.mean(article_length_nogood_upd), color = "red")

ax.plot(x, y, color = "black", label = "Gamma distribution,  \nalpha: {}, beta: {}, loc: {}".format(round(fit_alpha, 3), round(fit_beta, 3), round(fit_loc, 3)))

# set x-axis label
ax.set_xlabel("Article Size",fontsize=14)
# set y-axis label
ax.set_ylabel("# NoGood Articles", color = "red", fontsize=14)
ax.legend()
