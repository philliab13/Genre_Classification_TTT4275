#This can probably be a good way of desciding the last feature:
# You can compute a feature’s ANOVA F-score manually by comparing the variance between classes to the variance within classes. In other words, for each feature you’d do the following:

# Compute the overall mean of the feature.
# For each class:
# Compute the class mean.
# Compute the contribution to the between-class sum of squares: multiply the number of samples in the class by the squared difference between the class mean and the overall mean.
# Compute the contribution to the within-class sum of squares: sum the squared differences between each sample in that class and the class mean.
# Calculate mean squares:
# Mean square between (MSB) = (Between-class sum of squares) / (number of classes – 1)
# Mean square within (MSW) = (Within-class sum of squares) / (total number of samples – number of classes)
# F-score: The ratio 
# 𝐹
# =
# MSB
# /
# MSW
# F=MSB/MSW.