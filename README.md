# PathWise v1 model: street safety zones (k-means clustering)

**Question:** When a PathWise user walks at night, which NYC street segments should the app route them around?

**Data:** `street_segments.csv`, 300 street segments, one row per segment, night-time monthly averages.
**The data is synthetic.** `make_data.py` generates it from three made-up street types with random noise. It is not real NYC data yet.

| Column | Meaning |
|---|---|
| `segment_id` | identifier (not a feature) |
| `night_crimes_per_month` | reported incidents on the segment at night per month |
| `streetlights_per_100m` | streetlights per 100 metres |
| `night_foot_traffic_per_hr` | people passing per hour at night |
| `open_businesses_night` | businesses open after 10pm |
| `dist_to_subway_m` | metres to the nearest subway entrance |

Outcome: none (clustering).

**Run:**
```
pip install pandas scikit-learn matplotlib
python pathwise_model.py
```
The script splits 70/30, scales on the training set, prints the elbow (training and test), fits k = 3, returns a route decision for a new segment, and compares baseline vs v1.

**Assumptions for the value estimate (not real figures):** 10,000 night trips per month, 8 segments per trip, and a 1-in-20,000 chance per monthly crime that someone passing through is affected.
