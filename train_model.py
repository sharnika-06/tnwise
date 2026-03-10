import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate synthetic dataset
data = []

for i in range(600):

    blink_rate = np.random.randint(5, 25)
    face_width = np.random.randint(120, 350)

    # FONT SIZE RULE
    if face_width < 180:
        font_size = 40   # user far
    elif face_width > 280:
        font_size = 18   # user close
    else:
        font_size = 28   # comfortable range (no change)

    # COLOR RULE
    if blink_rate < 8:
        font_color = 1  # red alert
    else:
        font_color = 0  # normal

    data.append([blink_rate, face_width, font_size, font_color])

df = pd.DataFrame(data, columns=[
    "blink_rate",
    "face_width",
    "font_size",
    "font_color"
])

X = df[["blink_rate", "face_width"]]

# Train models
model_size = RandomForestClassifier()
model_size.fit(X, df["font_size"])

model_color = RandomForestClassifier()
model_color.fit(X, df["font_color"])

# Save models
joblib.dump(model_size, "model_size.pkl")
joblib.dump(model_color, "model_color.pkl")

print("ML Models trained and saved!")