import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("disney_sample_data.csv")

print("task#1")
lanugagesInUse = df['language'].value_counts()
print(lanugagesInUse)

print("task#2")
countries_reach = df.groupby('country')['reach'].sum().sort_values(ascending=False)
print(countries_reach)

print("task#3")
df = df[df['host_traffic'] > 0]
efficiency_by_source = (df['reach'] / df['host_traffic']).groupby(df['host']).mean().sort_values(ascending=False).head(3)
print(efficiency_by_source)


print("task#4")
df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')

sentiment_mapping = {
    'positive': 1,
    'neutral': 0,
    'negative': -1
}

df['sentiment'] = df['sentiment'].map(sentiment_mapping, na_action='ignore')

df = df.dropna(subset=['sentiment'])

df.sort_values('created_date', inplace=True)

df.set_index('created_date', inplace=True)

df_weekly_sentiment = df['sentiment'].resample('W').mean()

plt.figure(figsize=(15, 7))
plt.plot(df_weekly_sentiment, label='Average Sentiment over time', marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.title('Average Sentiment Analysis over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()