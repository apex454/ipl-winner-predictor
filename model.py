import pandas as pd

df = pd.read_csv("ipl_data.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df = df[['team1', 'team2', 'toss_winner', 'venue', 'match_winner']]

df = df.dropna()

df['team1_wins'] = (df['team1'] == df['match_winner']).astype(int)

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

X_train = train_df[['team1', 'team2', 'toss_winner', 'venue']].copy()
y_train = train_df['team1_wins']

X_test = test_df[['team1', 'team2', 'toss_winner', 'venue']].copy()
y_test = test_df['team1_wins']

from sklearn.preprocessing import LabelEncoder

le_team = LabelEncoder()

all_teams = pd.concat([
    X_train['team1'], X_train['team2'],
    X_test['team1'], X_test['team2']
])

le_team.fit(all_teams)

X_train['team1'] = le_team.transform(X_train['team1'])
X_train['team2'] = le_team.transform(X_train['team2'])

X_test['team1'] = le_team.transform(X_test['team1'])
X_test['team2'] = le_team.transform(X_test['team2'])

le_toss = LabelEncoder()
le_toss.fit(pd.concat([X_train['toss_winner'], X_test['toss_winner']]))

X_train['toss_winner'] = le_toss.transform(X_train['toss_winner'])
X_test['toss_winner'] = le_toss.transform(X_test['toss_winner'])

le_venue = LabelEncoder()
le_venue.fit(pd.concat([X_train['venue'], X_test['venue']]))

X_train['venue'] = le_venue.transform(X_train['venue'])
X_test['venue'] = le_venue.transform(X_test['venue'])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print("Predictions:", predictions[:10])
