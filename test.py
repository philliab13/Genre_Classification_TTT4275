import pandas as pd

df = pd.DataFrame(data={
    'edu_goal': ['bachelors', 'bachelors', 'bachelors', 'masters', 'masters', 'masters', 'masters', 'phd', 'phd', 'phd'],
    'hours_study': [1, 2, 3, 3, 3, 4, 3, 4, 5, 5],
    'hours_TV': [4, 3, 4, 3, 2, 3, 2, 2, 1, 1],
    'hours_sleep': [10, 10, 8, 8, 6, 6, 8, 8, 10, 10],
    'height_cm': [155, 151, 160, 160, 156, 150, 164, 151, 158, 152],
    'grade_level': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    'exam_score': [71, 72, 78, 79, 85, 86, 92, 93, 99, 100]
})

print(df)

X = df.drop(columns=['exam_score'])

print(X)

y = df['exam_score']

print(y)

X_num = X.drop(columns=['edu_goal'])

print(X_num)

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0)  # 0 is default

print(selector.fit_transform(X_num))
print(selector.get_support(indices=True))

num_cols = X_num.columns[selector.get_support(indices=True)].tolist()
print(num_cols)

X_num = X_num[num_cols]
print(X_num)

X = X[['edu_goal']+num_cols]
print(X)