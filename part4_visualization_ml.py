import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('students.csv')

# ---- Task 1 ----

print(df.head())

print("shape:", df.shape)
print(df.dtypes)

print(df.describe())

print(df['passed'].value_counts())

subject_cols = ['math', 'science', 'english', 'history', 'pe']

# average for students who passed
print("passed students avg:")
print(df[df['passed'] == 1][subject_cols].mean())

# average for students who failed
print("failed students avg:")
print(df[df['passed'] == 0][subject_cols].mean())

# finding top student
temp = df[subject_cols].mean(axis=1)
top = temp.idxmax()
print("best student:", df.loc[top, 'name'], "avg:", round(temp[top], 2))


# ---- Task 2 ----

df['avg_score'] = df[subject_cols].mean(axis=1)

# plot 1 - bar chart
avgs = df[subject_cols].mean()
plt.figure(figsize=(7, 5))
plt.bar(avgs.index, avgs.values, color='steelblue')
plt.title('Average Score per Subject')
plt.xlabel('Subject')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('plot1_bar.png')
plt.show()

# plot 2 - histogram of math scores
plt.figure(figsize=(7, 5))
plt.hist(df['math'], bins=5, color='salmon', edgecolor='black')
m = df['math'].mean()
plt.axvline(m, color='blue', linestyle='--', label=f'mean: {m:.1f}')
plt.title('Math Score Distribution')
plt.xlabel('Score')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plot2_histogram.png')
plt.show()

# plot 3 - scatter study hours vs avg score
p = df[df['passed'] == 1]
f = df[df['passed'] == 0]

plt.figure(figsize=(7, 5))
plt.scatter(p['study_hours_per_day'], p['avg_score'], color='green', label='Pass')
plt.scatter(f['study_hours_per_day'], f['avg_score'], color='red', label='Fail')
plt.title('Study Hours vs Avg Score')
plt.xlabel('Study Hours per Day')
plt.ylabel('Avg Score')
plt.legend()
plt.tight_layout()
plt.savefig('plot3_scatter.png')
plt.show()

# plot 4 - boxplot attendance pass vs fail
pass_att = df[df['passed'] == 1]['attendance_pct'].tolist()
fail_att = df[df['passed'] == 0]['attendance_pct'].tolist()

plt.figure(figsize=(6, 5))
plt.boxplot([pass_att, fail_att], labels=['Pass', 'Fail'])
plt.title('Attendance % - Pass vs Fail')
plt.xlabel('Result')
plt.ylabel('Attendance %')
plt.tight_layout()
plt.savefig('plot4_boxplot.png')
plt.show()

# plot 5 - line plot math and science per student
plt.figure(figsize=(10, 5))
plt.plot(df['name'], df['math'], marker='o', label='Math')
plt.plot(df['name'], df['science'], marker='s', linestyle='--', label='Science')
plt.title('Math and Science Scores per Student')
plt.xlabel('Name')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('plot5_line.png')
plt.show()


# ---- Task 3 ----

# seaborn bar plots for math and science split by passed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.barplot(data=df, x='passed', y='math', ax=ax1)
ax1.set_title('Avg Math by Pass/Fail')

sns.barplot(data=df, x='passed', y='science', ax=ax2)
ax2.set_title('Avg Science by Pass/Fail')

plt.tight_layout()
plt.savefig('plot6_seaborn_bar.png')
plt.show()

# seaborn regplot attendance vs avg score
plt.figure(figsize=(7, 5))
sns.regplot(data=df[df['passed'] == 1], x='attendance_pct', y='avg_score', color='green', label='Pass')
sns.regplot(data=df[df['passed'] == 0], x='attendance_pct', y='avg_score', color='red', label='Fail')
plt.title('Attendance vs Avg Score')
plt.xlabel('Attendance %')
plt.ylabel('Avg Score')
plt.legend()
plt.tight_layout()
plt.savefig('plot7_seaborn_scatter.png')
plt.show()

# seaborn was much easier for the regression plot, regplot handles the line automatically.
# with matplotlib i would have had to calculate the regression manually which takes more code.
# for simple group comparisons seaborn barplot is also quicker than setting up everything by hand.


# ---- Task 4 ----

features = ['math', 'science', 'english', 'history', 'pe', 'attendance_pct', 'study_hours_per_day']

X = df[features]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

print("training accuracy:", model.score(X_train_scaled, y_train))

y_pred = model.predict(X_test_scaled)
print("test accuracy:", accuracy_score(y_test, y_pred))

# show results per student
names = df.loc[X_test.index, 'name']
for n, actual, pred in zip(names, y_test, y_pred):
    a = 'Pass' if actual == 1 else 'Fail'
    p = 'Pass' if pred == 1 else 'Fail'
    status = 'correct' if actual == pred else 'wrong'
    print(f'{n} -> actual: {a}, predicted: {p} ({status})')

# feature coefficients
coefs = list(zip(features, model.coef_[0]))
coefs.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nfeature importance:")
for name, val in coefs:
    print(f"  {name}: {val:.4f}")

# bar chart for coefficients
fnames = [c[0] for c in coefs]
fvals = [c[1] for c in coefs]
colors = ['green' if v > 0 else 'red' for v in fvals]

plt.figure(figsize=(8, 5))
plt.barh(fnames, fvals, color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Coefficients')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.savefig('plot8_feature_importance.png')
plt.show()

