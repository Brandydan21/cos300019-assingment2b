from travel_time.travel_time_estimator import estimate_travel_time
from tensorflow.keras.models import load_model

from_node = 970
to_node = 2846
date_time = "2006-11-14 8:30"
model = load_model("training/models/lstm_scat_model.keras")

time = estimate_travel_time(from_node, to_node, date_time, model)
print(f"Travel time from {from_node} to {to_node} at {date_time} is {time}")
