import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
# Deepak - Check exact file names below.
df_j = pd.read_excel("Bo'27.xlsx")
df_s = pd.read_excel("Bo'26.xlsx")

# =========================
# 2. CREATE UNIQUE IDS
# =========================
df_j["J_ID"] = ["J" + str(i).zfill(3) for i in range(1, len(df_j) + 1)]
df_s["S_ID"] = ["S" + str(i).zfill(3) for i in range(1, len(df_s) + 1)]

# =========================
# 3. CLEAN IMPORTANT FIELDS
# =========================

# Standardize CA column (handle Yes/No variations)
def clean_ca(x):
    if pd.isna(x):
        return "No"
    x = str(x).strip().lower()
    return "Yes" if x in ["yes", "y"] else "No"

df_j["Is CA"] = df_j["Is CA"].apply(clean_ca)
df_s["Is CA"] = df_s["Is CA"].apply(clean_ca)

# Clean Undergrad Category
df_j["Undergrad Category"] = df_j["Undergrad Category"].astype(str).str.strip()
df_s["Undergrad Category"] = df_s["Undergrad Category"].astype(str).str.strip()

# =========================
# 4. EXPERIENCE TYPE (F / E)
# =========================

def get_exp_type(x):
    if pd.isna(x) or x == 0:
        return "F"   # Fresher
    return "E"       # Experienced

df_j["ExpType"] = df_j["Total number of months of work experience"].apply(get_exp_type)
df_s["ExpType"] = df_s["Total number of months of work experience"].apply(get_exp_type)

# =========================
# 5. CREATE SEGMENT KEY
# =========================
# Format: CA | UG Category | ExpType

def create_segment(df):
    return (
        df["Is CA"].astype(str) + "|" +
        df["Undergrad Category"].astype(str) + "|" +
        df["ExpType"]
    )

df_j["Segment"] = create_segment(df_j)
df_s["Segment"] = create_segment(df_s)

# =========================
# DEBUG: SEGMENT ANALYSIS
# =========================

print("\n========== SEGMENT DEBUG START ==========\n")

j_seg_counts = df_j["Segment"].value_counts()
s_seg_counts = df_s["Segment"].value_counts()

all_segments = sorted(set(j_seg_counts.index).union(set(s_seg_counts.index)))

for seg in all_segments:

    j_count = j_seg_counts.get(seg, 0)
    s_count = s_seg_counts.get(seg, 0)

    print(f"\n-------------------------------")
    print(f"SEGMENT: {seg}")
    print(f"Juniors: {j_count}")
    print(f"Seniors: {s_count}")

    # 🚨 Warnings
    if s_count == 0:
        print("❌ NO SENIORS AVAILABLE")
    elif s_count < j_count:
        print("⚠️ POSSIBLE OVERLOAD (More juniors than seniors)")
    elif s_count > j_count * 2:
        print("⚠️ UNDERUTILIZATION RISK (Too many seniors)")

    # =========================
    # LIST SENIORS IN SEGMENT
    # =========================
    seg_seniors = df_s[df_s["Segment"] == seg]

    if len(seg_seniors) > 0:
        print("\nSENIORS IN THIS SEGMENT:")

        for _, row in seg_seniors.iterrows():
            print(f"{row['S_ID']} | {row['Student Name (FN LN)']}")

    # =========================
    # LIST JUNIORS IN SEGMENT
    # =========================
    seg_juniors = df_j[df_j["Segment"] == seg]

    if len(seg_juniors) > 0:
        print("\nJUNIORS IN THIS SEGMENT:")

        for _, row in seg_juniors.iterrows():
            print(f"{row['J_ID']} | {row['Student Name (FN LN)']}")

print("\n========== SEGMENT DEBUG END ==========\n")

# =========================
# 6. OPTIONAL: VIEW SEGMENT DISTRIBUTION
# =========================

print("Junior Segment Distribution:")
print(df_j["Segment"].value_counts())

print("\nSenior Segment Distribution:")
print(df_s["Segment"].value_counts())

# =========================
# 7. CLEAN EXPERIENCE MONTHS
# =========================

print("\n========== STEP 7: CLEAN EXPERIENCE ==========")

def clean_exp_months(x):
    if pd.isna(x):
        return 0
    try:
        return int(x)
    except:
        return 0

df_j["ExpMonths"] = df_j["Total number of months of work experience"].apply(clean_exp_months)
df_s["ExpMonths"] = df_s["Total number of months of work experience"].apply(clean_exp_months)

print("\nJunior Experience Stats:")
print(df_j["ExpMonths"].describe())

print("\nSenior Experience Stats:")
print(df_s["ExpMonths"].describe())


# =========================
# 8. INDUSTRY EXTRACTION
# =========================

print("\n========== STEP 8: INDUSTRY EXTRACTION ==========")

def extract_industries(row):
    industries = []

    for i in range(1, 4):
        vals = [
            row.get(f"Choose Industry {i}, if Role {i} is Manufacturing related (otherwise choose N/A)"),
            row.get(f"Choose industry {i}, if Role {i} is Services related (Otherwise choose N/A)"),
            row.get(f"Other industry (Role {i})"),
            row.get(f"Other industry  - {i}")
        ]

        val = next((v for v in vals if pd.notna(v) and str(v).strip() != "N/A"), None)

        if val:
            industries.append(str(val).strip().lower())

    return industries

df_j["Industries"] = df_j.apply(extract_industries, axis=1)
df_s["Industries"] = df_s.apply(extract_industries, axis=1)

# Debug prints
print("\nSample Extracted Junior Industries:")
print(df_j[["J_ID", "Industries"]].head(10))

print("\nSample Extracted Senior Industries:")
print(df_s[["S_ID", "Industries"]].head(10))

# Unique industries
all_j_ind = set(i for sub in df_j["Industries"] for i in sub)
all_s_ind = set(i for sub in df_s["Industries"] for i in sub)

print("\nTotal Unique Junior Industries:", len(all_j_ind))
print("Total Unique Senior Industries:", len(all_s_ind))

print("\nJunior Industries List:", sorted(all_j_ind))
print("\nSenior Industries List:", sorted(all_s_ind))

print("\nEmpty Industry Rows (Juniors):", sum(df_j["Industries"].apply(len) == 0))
print("Empty Industry Rows (Seniors):", sum(df_s["Industries"].apply(len) == 0))


# =========================
# 9. CLEAN ACADEMIC FIELDS
# =========================

print("\n========== STEP 9: ACADEMIC CLEANING ==========")

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

# Juniors
df_j["UG_Degree_Clean"] = df_j["Undergraduate Degree"].apply(clean_text)
df_j["UG_Spec_Clean"] = df_j["Area of Specialization"].apply(clean_text)
df_j["Masters_Clean"] = df_j["Masters Degree, if any (e.g. M.Tech)"].apply(clean_text)
df_j["Masters_Spec_Clean"] = df_j["Area of Specialization, if any (e.g. M.Tech Computer Science)"].apply(clean_text)

# Seniors
df_s["UG_Degree_Clean"] = df_s["Undergraduate Degree"].apply(clean_text)
df_s["UG_Spec_Clean"] = df_s["Area of Specialization"].apply(clean_text)
df_s["Masters_Clean"] = df_s["Masters Degree (if any) e.g. M.Tech"].apply(clean_text)
df_s["Masters_Spec_Clean"] = df_s["Masters Specialization (if any) e.g. M.Tech Computer Science"].apply(clean_text)

print("\nSample Academic Fields (Juniors):")
print(df_j[["J_ID", "UG_Degree_Clean", "UG_Spec_Clean", "Masters_Clean"]].head(10))

print("\nSample Academic Fields (Seniors):")
print(df_s[["S_ID", "UG_Degree_Clean", "UG_Spec_Clean", "Masters_Clean"]].head(10))


# =========================
# 10. FINAL SUMMARY SNAPSHOT
# =========================

print("\n========== FINAL FEATURE SUMMARY ==========")

print("\nJuniors:")
print(df_j[["J_ID", "Segment", "ExpMonths", "Industries"]].head())

print("\nSeniors:")
print(df_s[["S_ID", "Segment", "ExpMonths", "Industries"]].head())


# =========================
# 13. HELPER FUNCTIONS
# =========================

print("\n========== STEP 13: SCORING SETUP (UPDATED) ==========")
# Deepak - Update scores accordingly
def exp_gap_score(gap, max_score):
    if gap <= 6:
        return max_score
    elif gap <= 12:
        return max_score * 0.75
    elif gap <= 24:
        return max_score * 0.5
    else:
        return max_score * 0.25


def industry_overlap_score(j_inds, s_inds, weight, per_match):
    overlap = len(set(j_inds) & set(s_inds))
    return min(overlap * per_match, weight)


# =========================
# 14. ROUND 1 SCORE (ACADEMIC HEAVY)
# =========================
# Deepak - Update round 1 score accordingly
# UnderGrad Degree - 25
# Specilization    - 25
# PG Degree        - 10
# Specialization   - 10
# Workex Industry  - 15(5 per each company*3)
# No of months     - 10(max)
def score_r1(j, s):
    breakdown = {}

    # UG Degree
    ug = 25 if j["UG_Degree_Clean"] == s["UG_Degree_Clean"] else 0

    # UG Specialization
    spec = 25 if j["UG_Spec_Clean"] == s["UG_Spec_Clean"] else 0

    # Masters
    masters = 10 if ( j["Masters_Clean"] != "" and s["Masters_Clean"] != "" and j["Masters_Clean"] == s["Masters_Clean"]) else 0

    # Masters Spec
    mspec = 10 if (
    j["Masters_Spec_Clean"] != "" and s["Masters_Spec_Clean"] != "" and j["Masters_Spec_Clean"] == s["Masters_Spec_Clean"]) else 0

    # Industry (low weight here)
    ind = industry_overlap_score(
        j["Industries"],
        s["Industries"],
        weight=15,
        per_match=5
    )

    # Experience (low weight here)
    if j["ExpMonths"] == 0 and s["ExpMonths"] == 0:
        exp = 0
    else:
        gap = abs(j["ExpMonths"] - s["ExpMonths"])
        exp = exp_gap_score(gap, 10)   # use 20 in R2

    total = ug + spec + masters + mspec + ind + exp

    breakdown.update({
        "UG": ug,
        "Spec": spec,
        "Masters": masters,
        "M_Spec": mspec,
        "Industry": ind,
        "Exp": exp,
        "Total": total
    })

    return total, breakdown


# =========================
# 15. ROUND 2 SCORE (WORKEX HEAVY)
# =========================
# Deepak - Update round 2 score accordingly
# UnderGrad Degree - 8
# Specilization    - 7
# PG Degree        - 5
# Specialization   - 0
# Workex Industry  - 35(12 per each company*3)
# No of months     - 20(max)
def score_r2(j, s):
    breakdown = {}

    # Industry (dominant)
    ind = industry_overlap_score(
        j["Industries"],
        s["Industries"],
        weight=35,
        per_match=12
    )

   # Experience (high weight)
    if j["ExpMonths"] == 0 and s["ExpMonths"] == 0:
        exp = 0
    else:
        gap = abs(j["ExpMonths"] - s["ExpMonths"])
        exp = exp_gap_score(gap, 20)

    # UG + Spec (supporting signal)
    ug = 8 if j["UG_Degree_Clean"] == s["UG_Degree_Clean"] else 0
    spec = 7 if j["UG_Spec_Clean"] == s["UG_Spec_Clean"] else 0

    # Masters (minor)
    masters = 5 if ( j["Masters_Clean"] != "" and s["Masters_Clean"] != "" and j["Masters_Clean"] == s["Masters_Clean"] ) else 0

    total = ind + exp + ug + spec + masters

    breakdown.update({
        "Industry": ind,
        "Exp": exp,
        "UG": ug,
        "Spec": spec,
        "Masters": masters,
        "Total": total
    })

    return total, breakdown


# =========================
# 16. QUICK TEST (UPDATED)
# =========================

print("\n========== SAMPLE SCORING CHECK (UPDATED) ==========")

sample_j = df_j.iloc[0]
sample_seg = sample_j["Segment"]

sample_seniors = df_s[df_s["Segment"] == sample_seg].head(3)

print(f"\nTesting for Junior: {sample_j['J_ID']} | Segment: {sample_seg}")

for _, s in sample_seniors.iterrows():
    r1, b1 = score_r1(sample_j, s)
    r2, b2 = score_r2(sample_j, s)

    print("\n----------------------------")
    print(f"Senior: {s['S_ID']}")

    print("\nR1 Breakdown:", b1)
    print("R2 Breakdown:", b2)

# =========================
# =========================
# =========================

debug_file = open("matching_debug2.txt", "w", encoding="utf-8")

def log(msg):
    print(msg)
    debug_file.write(str(msg) + "\n")

# =========================
# 17. MATCHING ENGINE (FINAL CLEAN VERSION)
# =========================

log("\n========== STEP 17: MATCHING ENGINE ==========")
# Deepak - Limit of 3 per round per senior
MAX_CAP = 3
EXTENDED_CAP = 6

# Load tracking
senior_load_r1 = {sid: 0 for sid in df_s["S_ID"]}
senior_load_r2 = {sid: 0 for sid in df_s["S_ID"]}

# Lookup dictionaries
j_lookup = df_j.set_index("J_ID").to_dict("index")
s_lookup = df_s.set_index("S_ID").to_dict("index")

final_matches = []
r1_debug = []
r2_debug = []

assignment_tracker = []

for _, j in df_j.iterrows():

    seg = j["Segment"]
    candidates = df_s[df_s["Segment"] == seg]

    log("\n====================================================")
    log(f"Processing Junior: {j['J_ID']} | Segment: {seg}")
    log(f"Candidates Available: {len(candidates)}")
    log("\nCurrent Senior Load Snapshot (Segment Only):")

    seg_seniors = candidates["S_ID"].tolist()

    if len(candidates) == 0:
        log("⚠️ No seniors available in this segment")

        final_matches.append([
            j["J_ID"],
            j_lookup[j["J_ID"]]["Student Name (FN LN)"],
            j_lookup[j["J_ID"]]["Phone Number"],
            None, None, None, None, 0, "NO SENIOR",
            None, None, None, None, 0, "NO SENIOR"
        ])
        continue

    # =========================
    # SCORE ALL CANDIDATES
    # =========================

    scores = []

    for _, s in candidates.iterrows():

        r1, b1 = score_r1(j, s)
        r2, b2 = score_r2(j, s)

        scores.append({
            "S_ID": s["S_ID"],
            "R1": float(r1),   # 🔥 FIX: force numeric
            "R2": float(r2),
            "R1_break": b1,
            "R2_break": b2
        })

        r1_debug.append([j["J_ID"], s["S_ID"], *b1.values()])
        r2_debug.append([j["J_ID"], s["S_ID"], *b2.values()])

    scores_df = pd.DataFrame(scores)

    scores_df = scores_df.sample(frac=1).reset_index(drop=True)

    # =========================
    # ROUND 1
    # =========================
    scores_df["Load_R1"] = scores_df["S_ID"].apply(lambda sid: senior_load_r1[sid])
    scores_df_r1 = scores_df.sort_values(by=["Load_R1", "R1"], ascending=[True, False])

    log("\n----- R1 START -----")

    r1_senior = None
    r1_score = 0
    flag_r1 = "NORMAL"

    for _, row in scores_df_r1.iterrows():
        sid = row["S_ID"]
        load = senior_load_r1[sid]

        log(f"Trying {sid} | Score: {row['R1']} | R1 Load: {senior_load_r1[sid]}")

        if load < MAX_CAP:
            r1_senior = sid
            r1_score = row["R1"]
            log(f"✅ Selected {sid}")
            break
        else:
            log(f"❌ Skipped {sid} (MAX_CAP reached)")

    if r1_senior is None:
        best_row = scores_df_r1.iloc[0]
        r1_senior = best_row["S_ID"]
        r1_score = best_row["R1"]
        flag_r1 = "FORCED OVERLOAD"
        log(f"🚨 Forced assign {r1_senior}")

    log("----- R1 END -----")

    # =========================
    # ROUND 2
    # =========================
    scores_df["Load_R2"] = scores_df["S_ID"].apply(lambda sid: senior_load_r2[sid])
    scores_df_r2 = scores_df.sort_values(by=["Load_R2", "R2"], ascending=[True, False])

    log("\n----- R2 START -----")

    r2_senior = None
    r2_score = 0
    flag_r2 = "NORMAL"

    for _, row in scores_df_r2.iterrows():
        sid = row["S_ID"]
        load = senior_load_r2[sid]

        log(f"Trying {sid} | Score: {row['R2']} | R2 Load: {senior_load_r2[sid]}")

        if sid != r1_senior and load < MAX_CAP:
            r2_senior = sid
            r2_score = row["R2"]
            log(f"✅ Selected {sid}")
            break
        else:
            log(f"❌ Skipped {sid}")

    if r2_senior is None:
        for _, row in scores_df_r2.iterrows():
            sid = row["S_ID"]
            if sid != r1_senior:
                r2_senior = sid
                r2_score = row["R2"]
                flag_r2 = "FORCED OVERLOAD"
                log(f"🚨 Forced assign {sid}")
                break

    log("----- R2 END -----")

    # =========================
    # LOAD UPDATE
    # =========================
    if r1_senior:
        senior_load_r1[r1_senior] += 1

    if r2_senior:
        senior_load_r2[r2_senior] += 1
    log("\nUpdated Loads:")
    if r1_senior:
        log(f"{r1_senior} -> R1 Load: {senior_load_r1[r1_senior]}")
    if r2_senior:
        log(f"{r2_senior} -> R2 Load: {senior_load_r2[r2_senior]}")

    # =========================
    # FETCH DETAILS
    # =========================

    j_name = j_lookup[j["J_ID"]]["Student Name (FN LN)"]
    j_phone = j_lookup[j["J_ID"]]["Phone Number"]

    if r1_senior:
        s1 = s_lookup[r1_senior]
        s1_name = s1["Student Name (FN LN)"]
        s1_phone = s1["WhatsApp Number"]
        s1_email = s1["Email Address"]
    else:
        s1_name = s1_phone = s1_email = None

    if r2_senior:
        s2 = s_lookup[r2_senior]
        s2_name = s2["Student Name (FN LN)"]
        s2_phone = s2["WhatsApp Number"]
        s2_email = s2["Email Address"]
    else:
        s2_name = s2_phone = s2_email = None

    # =========================
    # FINAL STORE
    # =========================

    final_matches.append([
        j["J_ID"],
        j_name,
        j_phone,

        r1_senior,
        s1_name,
        s1_phone,
        s1_email,
        r1_score,
        flag_r1,

        r2_senior,
        s2_name,
        s2_phone,
        s2_email,
        r2_score,
        flag_r2
    ])

# =========================
# CLOSE DEBUG FILE
# =========================
# FINAL LOAD SNAPSHOT (END OF MATCHING)
# =========================

log("\n\n========== FINAL SENIOR LOAD SNAPSHOT ==========\n")

for sid in senior_load_r1.keys():
    r1 = senior_load_r1[sid]
    r2 = senior_load_r2[sid]
    total = r1 + r2

    log(f"{sid} | R1 Load: {r1} | R2 Load: {r2} | Total: {total}")

log("\n========== END OF LOAD SNAPSHOT ==========\n")
debug_file.close()

# =========================
# 18. SAVE OUTPUT (3 SHEET STRUCTURE)
# =========================

print("\n========== SAVING OUTPUT ==========")

master_rows = []
r1_rows = []
r2_rows = []

# =========================
# MASTER SHEET
# =========================

for match in final_matches:

    j_id = match[0]
    r1_id = match[3]
    r2_id = match[9]

    j = j_lookup[j_id]

    def get_senior(sid):
        return s_lookup[sid] if sid else None

    s1 = get_senior(r1_id)
    s2 = get_senior(r2_id)

    master_rows.append([

        # Junior
        j_id,
        r1_id,
        r2_id,
        j["Student Name (FN LN)"],
        s1["Student Name (FN LN)"] if s1 else None, 
        s2["Student Name (FN LN)"] if s2 else None,
        j["Undergrad Category"],
        s1["Undergrad Category"] if s1 else None,
        s2["Undergrad Category"] if s2 else None,
        j["Is CA"],
        s1["Is CA"] if s1 else None,
        s2["Is CA"] if s2 else None,
        j["UG_Degree_Clean"],
        s1["UG_Degree_Clean"] if s1 else None,
        s2["UG_Degree_Clean"] if s2 else None,
        j["UG_Spec_Clean"],
        s1["UG_Spec_Clean"] if s1 else None,
        s2["UG_Spec_Clean"] if s2 else None,
        j["Masters_Clean"],
        s1["Masters_Clean"] if s1 else None,
        s2["Masters_Clean"] if s2 else None,
        j["Masters_Spec_Clean"],
        s1["Masters_Spec_Clean"] if s1 else None,
        s2["Masters_Spec_Clean"] if s2 else None,
        j["ExpMonths"],
        s1["ExpMonths"] if s1 else None,
        s2["ExpMonths"] if s2 else None,
        ", ".join(j["Industries"]),
        ", ".join(s1["Industries"]) if s1 else None,
        ", ".join(s2["Industries"]) if s2 else None,
        match[7],   # R1 Score
        match[8],   # R1 Flag
        match[13],  # R2 Score
        match[14]   # R2 Flag
    ])

master_cols = [

    # Junior
    "J_ID","R1_ID","R2_ID",
    "J_Name","R1_Name","R2_Name",
    "J_Undergrad_Category","R1_Undergrad_Category","R2_Undergrad_Category",
    "J_CA","R1_CA","R2_CA",
    "J_UG","R1_UG","R2_UG",
    "J_Spec","R1_Spec","R2_Spec",
    "J_Masters","R1_Masters","R2_Masters",
    "J_MSpec","R1_MSpec","R2_MSpec",
    "J_Exp","R1_Exp","R2_Exp",
    "J_Industries","R1_Industries","R2_Industries",
    # R1
    "R1_Score","R1_Flag",
    # R2
    "R2_Score","R2_Flag"
]

master_df = pd.DataFrame(master_rows, columns=master_cols)


# =========================
# R1 DEBUG SHEET
# =========================

for row in r1_debug:

    j_id, s_id = row[0], row[1]
    j = j_lookup[j_id]
    s = s_lookup[s_id]

    r1_rows.append([

        j_id,
        s_id,
        j["Student Name (FN LN)"],
        s["Student Name (FN LN)"],
        j["Undergrad Category"],
        s["Undergrad Category"],
        j["Is CA"],
        s["Is CA"],
        j["UG_Degree_Clean"],
        s["UG_Degree_Clean"],
        j["UG_Spec_Clean"],
        s["UG_Spec_Clean"],
        j["Masters_Clean"],
        s["Masters_Clean"],
        j["Masters_Spec_Clean"],
        s["Masters_Spec_Clean"],
        j["ExpMonths"],
        s["ExpMonths"],
        ", ".join(j["Industries"]),
        ", ".join(s["Industries"]),

        # Score breakdown
        row[2],  # UG
        row[3],  # Spec
        row[4],  # Masters
        row[5],  # M Spec
        row[6],  # Industry
        row[7],  # Exp
        row[8]   # Total
    ])

r1_cols = [

    "J_ID","S_ID",
    "J_Name","S_Name",
    "J_Undergrad_Category","S_Undergrad_Category",
    "J_CA","S_CA",
    "J_UG","S_UG",
    "J_Spec","S_Spec",
    "J_Masters","S_Masters",
    "J_MSpec","S_MSpec",
    "J_Exp","S_Exp",
    "J_Industries","S_Industries",
    "Score_UG","Score_Spec","Score_Masters",
    "Score_MSpec","Score_Industry","Score_Exp","Total"
]

r1_df = pd.DataFrame(r1_rows, columns=r1_cols)


# # =========================
# # R2 DEBUG SHEET
# # =========================

for row in r2_debug:

    j_id, s_id = row[0], row[1]
    j = j_lookup[j_id]
    s = s_lookup[s_id]

    r2_rows.append([

        j_id,
        j["Student Name (FN LN)"],
        j["Undergrad Category"],
        j["Is CA"],
        j["UG_Degree_Clean"],
        j["UG_Spec_Clean"],
        j["Masters_Clean"],
        j["Masters_Spec_Clean"],
        j["ExpMonths"],
        ", ".join(j["Industries"]),

        s_id,
        s["Student Name (FN LN)"],
        s["Undergrad Category"],
        s["Is CA"],
        s["UG_Degree_Clean"],
        s["UG_Spec_Clean"],
        s["Masters_Clean"],
        s["Masters_Spec_Clean"],
        s["ExpMonths"],
        ", ".join(s["Industries"]),

        # Score breakdown
        row[2],  # Industry
        row[3],  # Exp
        row[4],  # UG
        row[5],  # Spec
        row[6],  # Masters
        row[7]   # Total
    ])

r2_cols = [

    "J_ID","J_Name","J_Undergrad_Category","J_CA",
    "J_UG","J_Spec","J_Masters","J_MSpec",
    "J_Exp","J_Industries",

    "S_ID","S_Name","S_Undergrad_Category","S_CA",
    "S_UG","S_Spec","S_Masters","S_MSpec",
    "S_Exp","S_Industries",

    "Score_Industry","Score_Exp","Score_UG",
    "Score_Spec","Score_Masters","Total"
]

r2_df = pd.DataFrame(r2_rows, columns=r2_cols)


# =========================
# WRITE TO EXCEL
# =========================

with pd.ExcelWriter("output15.xlsx") as writer:
    master_df.to_excel(writer, sheet_name="Master", index=False)
    r1_df.to_excel(writer, sheet_name="R1_Debug", index=False)
    r2_df.to_excel(writer, sheet_name="R2_Debug", index=False)

print("✅ Output saved with full debug structure")