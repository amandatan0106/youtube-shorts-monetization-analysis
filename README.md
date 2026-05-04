# Makeup Creator Monetization Predictor

A data science project analyzing YouTube Shorts to predict creator monetization
potential in the makeup niche — using engagement signals as a proxy for
short-form video platform monetization strategies.

---

## Project Overview

Short-form video platforms monetize through creator partnerships, brand deals,
and commerce integrations. This project identifies what content and channel
signals predict high monetization potential in makeup YouTube Shorts, using
YouTube as a data-accessible proxy for short-form video platforms.

**Two modeling approaches:**
1. **Regression** — predict a video's engagement rate
2. **Classification** — predict whether a video has high monetization potential

**A/B Test** — formally test whether upload timing significantly affects engagement.

---

## Project Structure

```
CREATOR_MONETIZATION/
│
├── data/
│   ├── makeup_shorts_raw.csv        ← raw scraped data
│   └── makeup_shorts_clean.csv      ← cleaned, feature-engineered data
│
├── notebooks/
│   ├── data_fetching.ipynb          ← YouTube API data collection
│   ├── data_analysis.ipynb          ← EDA, cleaning, feature engineering
│   └── modeling.ipynb               ← regression, classification, A/B test
│
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- YouTube Data API v3 key ([get one here](https://console.cloud.google.com))

### Install dependencies
```bash
pip install google-api-python-client pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels scipy
```

### API Configuration
In `data_fetching.ipynb`, replace:
```python
API_KEY = "YOUR_API_KEY_HERE"
```
with your YouTube Data API v3 key.

### Running the notebooks
Run in this order:
```
1. data_fetching.ipynb      ← collects raw data (~10-15 mins)
2. data_analysis.ipynb      ← EDA and feature engineering
3. modeling.ipynb           ← all models and A/B test
```

---

## Dataset

### Source
- **API:** YouTube Data API v3
- **Query:** 25 makeup-specific search queries with `#shorts` tag
- **Filter:** Videos ≤ 60 seconds (true Shorts)
- **Coverage:** 2022–2025 (post YouTube Shorts launch)
- **Size:** 1,983 videos across 1,170 unique creators

### Raw Data Columns (`makeup_shorts_raw.csv`)

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique YouTube video identifier |
| `channel_id` | str | Unique YouTube channel identifier |
| `search_query` | str | Query used to find the video |
| `title` | str | Video title |
| `description` | str | Video description (truncated to 300 chars) |
| `tags` | str | Pipe-separated tags from creator |
| `published_at` | datetime | Upload timestamp (UTC) |
| `duration_seconds` | int | Video length in seconds (max 60) |
| `views` | int | Total view count |
| `likes` | int | Total like count |
| `comments` | int | Total comment count |
| `engagement_rate` | float | (likes + comments) / views |
| `like_rate` | float | likes / views |
| `comment_rate` | float | comments / views |
| `days_since_upload` | int | Days between upload and data collection |
| `view_velocity` | float | views / days_since_upload |
| `subscriber_count` | int | Channel subscriber count |
| `total_channel_views` | int | Total lifetime channel views |
| `total_videos` | int | Total videos on channel |
| `url` | str | Full YouTube Shorts URL |

### Clean Data Columns (`makeup_shorts_clean.csv`)

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique YouTube video identifier |
| `channel_id` | str | Unique YouTube channel identifier |
| `published_at` | datetime | Upload timestamp (UTC) |
| `year` | int16 | Upload year |
| `quarter` | int16 | Upload quarter (1-4) |
| `month` | category | Upload month name |
| `day_of_week` | category | Upload day name |
| `hour` | int32 | Upload hour (0-23 UTC) |
| `hour_bucket` | category | Night/Morning/Afternoon/Evening |
| `days_since_upload` | int32 | Days between upload and collection |
| `duration_seconds` | int32 | Video length in seconds |
| `duration_bucket` | category | 0-15s / 16-30s / 31-45s / 46-60s |
| `content_type` | category | Classified content type (15 categories) |
| `subscriber_tier` | category | Influencer tier based on subscriber count |
| `log_subscriber_count` | float32 | log1p(subscriber_count) |
| `log_total_channel_views` | float32 | log1p(total_channel_views) |
| `log_total_videos` | float32 | log1p(total_videos) |
| `log_views` | float32 | log1p(views) |
| `log_likes` | float32 | log1p(likes) |
| `log_comments` | float32 | log1p(comments) |
| `log_view_velocity` | float32 | log1p(view_velocity) |
| `engagement_rate` | float32 | (likes + comments) / views |
| `like_rate` | float32 | likes / views |
| `is_viral` | int8 | 1 if views > Q3 + 1.5×IQR else 0 |

### Content Type Categories

| Category | Keywords Used |
|---|---|
| GRWM | grwm, get ready with me, grwu |
| Tutorial | tutorial, how to, step by step |
| Transformation | transformation, before and after, unrecognizable, makeover |
| Dupe/Affordable | dupe, drugstore, affordable, cheap |
| Beginner | beginner, easy, simple, basic |
| Glam | glam, bold, full face, full glam |
| Natural | natural, no makeup, clean girl, minimal |
| Routine | routine, morning routine, glowy |
| Hack/Tip | hack, hacks, tip, trick, method |
| Everyday/School | school, everyday, daily, work, office |
| K-Beauty/KPop | kpop, k-pop, korean, idol, kbeauty |
| Haul/Shopping | sephora, ulta, haul, new in, unboxing |
| Skin/Coverage | acne, texture, redness, coverage, skin, pores |
| Transition/Ranking | power of makeup, transition, ranking, satisfying |
| Other | does not match any above keywords |

### Subscriber Tier Definition

| Tier | Subscriber Range |
|---|---|
| Unknown | < 1,000 |
| Nano | 1,000 – 5,000 |
| Micro | 5,000 – 50,000 |
| Mid-Tier | 50,000 – 300,000 |
| Macro | 300,000 – 1,000,000 |
| Mega | 1,000,000+ |

---

## Key EDA Findings

- **Longer Shorts win** — 46-60s videos have highest engagement across all analyses
- **Evening posts outperform** — 18-24 UTC uploads average 50% higher engagement than morning
- **Engagement declining YoY** — 2022 → 2025 engagement rates trending down as competition increases
- **Routine & Dupe content** drive highest engagement by content type
- **Mega/Macro creators** have higher engagement than smaller tiers in makeup niche
- **289 viral videos** (14.6%) flagged via IQR method

---

## Modeling Results

### Regression — Predicting Engagement Rate

| Model | CV R² | Test R² | RMSE | MAE |
|---|---|---|---|---|
| Linear Regression | — | 0.364 | 0.0181 | 0.0137 |
| Random Forest | 0.3946 | 0.4389 | 0.0170 | 0.0130 |
| **XGBoost** | **0.4270** | **0.4428** | **0.0170** | **0.0128** |

### Classification — Predicting High Monetization Potential

Target: `high_potential = 1` if engagement_rate > median AND log_views > median

| Model | CV F1 | Test F1 | AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | — | 0.6466 | 0.8855 | 0.5245 | 0.8427 |
| Random Forest | 0.7016 | 0.6866 | 0.9125 | 0.6161 | 0.7753 |
| **XGBoost** | **0.7226** | **0.7310** | **0.9403** | **0.6667** | **0.8090** |

**XGBoost selected as final model for both tasks.**

---

## A/B Test — Upload Timing

**Question:** Do evening uploads (18-24 UTC) generate significantly higher engagement than morning uploads (6-12 UTC)?

| Group | N | Avg Engagement Rate |
|---|---|---|
| Morning (6-12 UTC) | 377 | 0.0253 |
| Evening (18-24 UTC) | 467 | 0.0380 |

| Metric | Result |
|---|---|
| Relative Lift | +50.1% |
| Test | Mann-Whitney U |
| P-value | < 0.0001 |
| Cohen's d | 0.5573 (Medium) |
| Decision | Reject H₀ |

Evening uploads generate statistically significant higher engagement (p < 0.0001).

---

## Business Recommendations

1. **Prioritize longer content (46-60s)** — maximum duration Shorts consistently
   outperform shorter videos in both engagement and monetization potential

2. **Evening upload timing drives 50% lift** — platforms should incorporate
   upload timing guidance into creator monetization education programs

3. **Routine & Natural content outperforms Hack/Tip format** — content category
   should be weighted in creator partnership identification

4. **Mid-tier creators are underserved** — Macro/Mega channels dominate but
   Mid-Tier (50K-300K) creators show competitive engagement with higher growth potential

5. **View velocity is a double-edged signal** — viral reach and sustained
   engagement serve different monetization objectives and should be evaluated separately

---

## Limitations

- YouTube used as proxy for short-form platforms — algorithmic differences may affect generalizability
- 76% of creators represented by a single video — per-creator analysis limited
- UTC timestamps may misclassify creator local posting times
- Unobservable content quality factors limit regression R² (~0.44)
- High AUC partly attributable to correlation between view velocity feature and log_views target component
- Observational data — causal claims require true randomized experiments

---

## Tech Stack

- **Data Collection:** YouTube Data API v3, `google-api-python-client`
- **Analysis:** `pandas`, `numpy`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn`, `xgboost`, `statsmodels`

---

## Author

Amanda Tan — [GitHub](https://github.com/amandatan0106)
