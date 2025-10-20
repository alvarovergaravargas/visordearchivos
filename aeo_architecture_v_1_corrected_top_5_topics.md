# AEO “Responsive + Embedding” Architecture (v1) — corrected for **top‑5 topics**

This spec defines an end‑to‑end pipeline that accepts **[brand, product, country]** and returns structured outputs with **top‑5 topics** for both the **product category** and the **product**.

---

## 0) System overview
**Pipeline:**
1. **Normalize Input** → clean brand/product, resolve `country` (ISO‑2), set `time_window = last 30 days` in local time.
2. **Product Category Resolver (PCR)** → return **search‑term‑style product category** popular in the country.
3. **Competitor Resolver (CR)** → return two **mostly compared** competitors (Brand + Product) in the country.
4. **Category Topics Builder (CTB)** → generate **100** topics for the category (country + last month) → **embed → cluster → top‑5** with **Interest Share**.
5. **Category Mention Matrix (CMM)** → expected **mention rate** between the **top‑5 category topics** and **{product, comp1, comp2, Other}** using **semantic affinity + response likelihood (logprob)** fusion.
6. **Product Topics Builder (PTB)** → generate **100** topics for {product or comps} → **embed → cluster → top‑5** with **Interest Share**.
7. **Cross Mention Matrix (XMM)** *(optional)* → re‑estimate mention rates using product‑side prompts; **5 rows** (one per category top topic) for stability/triangulation.
8. **Assembler** → final JSON.

Caching: PCR, CR, and the 100‑topic lists cache for 24h per `[country, category/product]`.

---

## 1) Product Category Resolver (PCR)
**Goal:** most common **consumer search term** (1–3 words) for the product in the country.

**Method (ensemble; choose consensus or tie‑break with LLM):**
- **Rules + Alias Map** (e.g., “Galaxy S25 Ultra” → “smartphone”).
- **Embedding KNN** vs country‑localized category dictionary.
- **Responsive AI tie‑breaker** *(temperature=0)*:
  > “You are a market taxonomy assistant for **[country]**. Return the most common consumer search term (1–3 words) for **[product]**. Output only the term.”

**Output:** `"product_category": "smartphone"`

---

## 2) Competitor Resolver (CR)
**Goal:** two **mostly compared** competitors (Brand + Product) for the country.

**Steps:**
- Build a **seed set** by family/price tier.
- **Embedding similarity** rank.
- **Responsive AI voting** *(temperature=0)* for “mostly compared in [country], last month” (top‑5 JSON).
- **Rank fusion** (Borda or mean rank) → pick **top‑2**.

**Output:** `{ "first_competitor": "BrandA ModelX", "second_competitor": "BrandB ModelY" }`

---

## 3) Category Topics Builder (CTB)
**3.1 Generate 100 raw topics (country + last month)**
Prompt *(temperature=0; deterministic)*:
> “List exactly 100 distinct, consumer‑phrased questions or intents people in **[country]** asked in the last 30 days about **[product_category]**. ≤12 words. No duplicates. Output a JSON array of strings.”

**3.2 Embed + Cluster → Top‑5 topics**
- **Embedding:** production‑grade embeddings (e.g., 3k‑dim).
- **Clustering:** HDBSCAN (preferred)
  - `min_cluster_size = max(5, floor(100*0.06))`
  - `min_samples = 1`
  - *Fallback:* KMeans with heuristic K.
- **Labeling:** take 5 nearest samples to each cluster centroid and ask LLM:
  > “Name this topic in ≤4 words (market language) that best covers: [5 sample items]. Return only the label.”
- **Interest Share (IS):** `IS_k = size(cluster_k) / 100`. Keep **top‑5** by size (ties → density).

**Outputs:**
- `category_topics_100` (100 strings)
- `category_top5` (5 labeled clusters with `interest_share` and `members`)

---

## 4) Category Mention Matrix (CMM)
**Goal:** expected **mention rate** `P(item | topic_k)` over items `{product, comp1, comp2, Other}` for the **top‑5 category topics**.

**Signals:**
### A) Semantic Affinity (SA)
- Compute **topic centroid embeddings** (mean over members).
- Compute cosine similarity to item embeddings (Brand + Product strings).
- Convert per‑topic to probabilities via **temperature‑scaled softmax**:  
  `p_sem(i|k) = softmax_i( sim(topic_k, item_i) / τ )`, recommend `τ = 0.15`.
- Add an **Other** small mass then renormalize.

### B) Response Likelihood (RL) via logprobs
Per (topic_k, country):
- **Cloze single‑choice** prompt *(temperature=0, logprobs on)*:
  > “In **[country]**, when people discuss **[topic_label]** about **[product_category]**, which product is most likely to be mentioned first?  
  > A) **[product]**  
  > B) **[comp1]**  
  > C) **[comp2]**  
  > D) Other  
  > **Answer with exactly one letter.**”
- Read token logprobs for A/B/C/D; normalize; apply mild **Dirichlet smoothing** favoring Other.

### C) Calibrated Fusion
`p(i|k) = α · p_sem(i|k) + (1−α) · p_rl(i|k)` with `α ≈ 0.6`. Clip to `[0.01, 0.97]` and renormalize.

**Output:** 5 rows × 4 probs each.

---

## 5) Product Topics Builder (PTB) — **fixed to top‑5**
**5.1 Generate 100 raw topics** focused on **{product or comps}** in **[country]**, **last 30 days**:
> “List exactly 100 distinct questions/intents people in **[country]** asked in the last 30 days about **[product]**, **[comp1]**, or **[comp2]**. ≤12 words. No duplicates. Output JSON array.”

**5.2 Embed + Cluster → **keep top‑5** clusters** by size. Label and compute `interest_share` like CTB.

**Outputs:**
- `product_topics_100` (100 strings)
- `product_top5` (5 labeled clusters)

---

## 6) Cross Mention Matrix (XMM) *(optional)*
Re‑estimate `P(item | topic_k)` across the **same 5 category topics** using **product‑topic prompts** to triangulate CMM. Report 5 rows and an optional **consistency score** (mean absolute diff; ≤7pp = stable).

---

## 7) Output JSON (canonical)
```json
{
  "input": { "brand": "", "product": "", "country": "" },
  "product_category": "string",
  "competitors": { "first": "BrandA ModelX", "second": "BrandB ModelY" },
  "category_topics": {
    "raw_100": ["... x100"],
    "top5": [
      { "label": "Battery life", "interest_share": 0.22, "members": [0,7,...] },
      { "label": "Camera quality", "interest_share": 0.20, "members": [...] },
      { "label": "Display", "interest_share": 0.19, "members": [...] },
      { "label": "Performance", "interest_share": 0.20, "members": [...] },
      { "label": "Price & deals", "interest_share": 0.19, "members": [...] }
    ],
    "mention_rate_matrix": [
      { "topic_label": "Battery life", "product": 0.41, "first_competitor": 0.33, "second_competitor": 0.18, "other": 0.08 },
      { "topic_label": "Camera quality", "product": 0.37, "first_competitor": 0.39, "second_competitor": 0.16, "other": 0.08 },
      { "topic_label": "Display", "product": 0.35, "first_competitor": 0.34, "second_competitor": 0.21, "other": 0.10 },
      { "topic_label": "Performance", "product": 0.36, "first_competitor": 0.33, "second_competitor": 0.22, "other": 0.09 },
      { "topic_label": "Price & deals", "product": 0.32, "first_competitor": 0.38, "second_competitor": 0.20, "other": 0.10 }
    ]
  },
  "product_topics": {
    "raw_100": ["... x100"],
    "top5": [
      { "label": "Thermal throttling", "interest_share": 0.22, "members": [...] },
      { "label": "Night photos", "interest_share": 0.21, "members": [...] },
      { "label": "Battery endurance", "interest_share": 0.20, "members": [...] },
      { "label": "Video stabilization", "interest_share": 0.19, "members": [...] },
      { "label": "Trade-in & price", "interest_share": 0.18, "members": [...] }
    ]
  },
  "cross_mention_rate_matrix": [
    { "topic_label": "Battery life", "product": 0.40, "first_competitor": 0.34, "second_competitor": 0.18, "other": 0.08 },
    { "topic_label": "Camera quality", "product": 0.36, "first_competitor": 0.40, "second_competitor": 0.16, "other": 0.08 },
    { "topic_label": "Display", "product": 0.35, "first_competitor": 0.35, "second_competitor": 0.20, "other": 0.10 },
    { "topic_label": "Performance", "product": 0.37, "first_competitor": 0.33, "second_competitor": 0.21, "other": 0.09 },
    { "topic_label": "Price & deals", "product": 0.31, "first_competitor": 0.39, "second_competitor": 0.20, "other": 0.10 }
  ],
  "meta": {
    "time_window": "last_30_days",
    "country": "US",
    "model_temp": 0,
    "fusion_alpha": 0.6,
    "embedding_model": "text-embedding-3-large",
    "clustering": "HDBSCAN|KMeans-fallback",
    "version": "aeo-rae-v1"
  }
}
```

---

## 8) Ready‑to‑paste prompts
- **PCR (tie‑breaker):** return the **single** search term (1–3 words), no explanation.
- **CR:** “top 5 competing products (brand + product) for last month in [country]” → JSON array; then take top‑2.
- **CTB 100:** exactly 100 topics, ≤12 words, last 30 days, no dups → JSON array.
- **Cluster Label:** “≤4 words” covering 5 sample items; return only the label.
- **RL single‑choice:** A/B/C/D with **exactly one letter** as the answer.

---

## 9) Scoring details
- **Semantic Affinity:** cosine → per‑topic softmax (τ≈0.15). Optionally z‑score sims per topic before softmax.
- **Response Likelihood:** sum token logprobs per choice; smooth towards **Other**; normalize.
- **Fusion:** `p = α p_sem + (1−α) p_rl` with `α=0.6` default.
- **Interest Share:** cluster_size / 100; report to two decimals.

---

## 10) Determinism & quality controls
- `temperature=0`, `top_p=1`, fixed prompts.
- De‑dup (Jaccard ≥0.8) before embedding.
- Country awareness: bilingual synonyms if market is multilingual.

---

## 11) Evaluation & monitoring
- **Stability score:** re‑run generation twice; Jaccard(Top‑K labels) ≥ 0.7.
- **Geo relevance:** spot‑check 10 random topics per country.
- **Mention sanity:** ensure `Other` ≥ 5% avg.
- **Drift watch:** alert if Interest Share entropy collapses week‑over‑week.

---

## 12) Runtime deliverables
For a specific **[brand, product, country]**, return exactly:
1) **product category** (search‑term style)
2) **1st competitor, 2nd competitor** (Brand + Product)
3) **top ChatGPT topics in product category**
   - 100 topics (list)
   - **top‑5 clusters** with **Interest Share**
   - **mention‑rate matrix** across `{product, comp1, comp2, Other}`
4) **top ChatGPT topics in product**
   - 100 topics (list)
   - **top‑5 clusters** with **Interest Share**
   - *(Optional)* cross‑mention re‑score for triangulation

