import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# carregar dados
df = pd.read_csv("data/Potabilidade_da_agua.csv")

# tratar dados
df = df.fillna(df.mean())

X = df.drop("Potability", axis=1)
y = df["Potability"]

# scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# salvar
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("✅ Modelo treinado e salvo com sucesso!")