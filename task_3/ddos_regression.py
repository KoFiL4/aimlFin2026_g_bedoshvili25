import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'minute': np.arange(1, 21),
    'requests': [12, 15, 14, 13, 15, 150, 165, 155, 170, 160, 14, 12, 15, 13, 11, 14, 12, 15, 13, 12]
}
df = pd.DataFrame(data)

X = df[['minute']]
y = df['requests']
model = LinearRegression().fit(X, y)
df['predicted'] = model.predict(X)

df['is_ddos'] = df['requests'] > (df['predicted'] * 2)
attack_minutes = df[df['is_ddos'] == True]['minute'].tolist()

print(f"DDoS Attack detected at minutes: {attack_minutes}")

plt.figure(figsize=(10, 6))
plt.scatter(df['minute'], df['requests'], color='blue', label='Actual Traffic')
plt.plot(df['minute'], df['predicted'], color='red', label='Regression Line (Trend)')
plt.fill_between(df['minute'], df['requests'], where=df['is_ddos'], color='red', alpha=0.3, label='Attack Interval')
plt.title('Log Analysis: DDoS Detection via Regression')
plt.xlabel('Minute')
plt.ylabel('Requests count')
plt.legend()
plt.show()