import duckdb
import os
import time

# ============================================================
# AMAZON ML CHALLENGE 2026
# PRODUCTION CANDIDATE GENERATION
#
# Based exactly on the validated:
# V3 + ADDRESS TOKENS blocking strategy
# ============================================================

SOURCE1 = "dataset/test/test_source1.tsv"
SOURCE2 = "dataset/test/test_source2.tsv"
SOURCE3 = "dataset/test/test_source3.tsv"

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

CHUNK_SIZE = 50000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("==============================================")
print("PRODUCTION CANDIDATE GENERATION")
print("V3 + ADDRESS TOKENS")
print("==============================================")
print()

start_time = time.time()

con = duckdb.connect()
con.execute("PRAGMA threads=2")

# ============================================================
# STEP 1: LOAD SOURCE 1
# ============================================================

print("Loading test Source 1...")

con.execute("""
CREATE OR REPLACE TEMP TABLE source1_all AS
SELECT
    row_number() OVER () AS rn,
    entity_id,
    country,

    trim(
        regexp_replace(
            lower(coalesce(business_name, '')),
            '[^[:alnum:]]+',
            ' ',
            'g'
        )
    ) AS norm_name,

    trim(
        regexp_replace(
            lower(coalesce(business_address, '')),
            '[^[:alnum:]]+',
            ' ',
            'g'
        )
    ) AS norm_address

FROM read_csv(
    'dataset/test/test_source1.tsv',
    delim='\t',
    header=true
)
""")

source1_count = con.execute(
    "SELECT COUNT(*) FROM source1_all"
).fetchone()[0]

print("Source 1 records:", source1_count)
print()

# ============================================================
# STEP 2: LOAD SOURCE 2 + SOURCE 3
# ============================================================

print("Loading test Source 2 and Source 3...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates AS

SELECT
    entity_id,
    country,

    trim(
        regexp_replace(
            lower(coalesce(business_name, '')),
            '[^[:alnum:]]+',
            ' ',
            'g'
        )
    ) AS norm_name,

    trim(
        regexp_replace(
            lower(coalesce(business_address, '')),
            '[^[:alnum:]]+',
            ' ',
            'g'
        )
    ) AS norm_address

FROM read_csv(
    'dataset/test/test_source2.tsv',
    delim='\t',
    header=true
)

UNION ALL

SELECT
    entity_id,
    country,

    trim(
        regexp_replace(
            lower(coalesce(business_name, '')),
            '[^[:alnum:]]+',
            ' ',
            'g'
        )
    ) AS norm_name,

    trim(
        regexp_replace(
            lower(coalesce(business_address, '')),
            '[^[:alnum:]]+',
            ' ',
            'g'
        )
    ) AS norm_address

FROM read_csv(
    'dataset/test/test_source3.tsv',
    delim='\t',
    header=true
)
""")

candidate_count = con.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

print("Total candidate records:", candidate_count)
print()

# ============================================================
# STEP 3: BUILD CANDIDATE-SIDE NAME TOKEN TABLE
# EXACT VALIDATED SETTING: token length >= 4
# ============================================================

print("Building candidate name-token index...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidate_name_tokens AS
SELECT DISTINCT
    c.entity_id,
    c.country,
    trim(t.token) AS token
FROM candidates c,
LATERAL UNNEST(
    string_split(c.norm_name, ' ')
) AS t(token)
WHERE length(trim(t.token)) >= 4
  AND trim(t.token) <> ''
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE token_frequency AS
SELECT
    country,
    token,
    COUNT(*) AS token_count
FROM candidate_name_tokens
GROUP BY country, token
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE useful_tokens AS
SELECT
    country,
    token,
    token_count
FROM token_frequency
WHERE token_count <= 1000
""")

print("Candidate name-token index ready.")
print()

# ============================================================
# STEP 4: BUILD CANDIDATE-SIDE NAME PREFIX INDEX
# EXACT VALIDATED SETTING: 5 chars, frequency <= 5000
# ============================================================

print("Building name-prefix index...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidate_name_prefixes AS
SELECT
    entity_id,
    country,
    substr(norm_name, 1, 5) AS name_prefix
FROM candidates
WHERE length(norm_name) >= 5
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE name_prefix_frequency AS
SELECT
    country,
    name_prefix,
    COUNT(*) AS prefix_count
FROM candidate_name_prefixes
GROUP BY country, name_prefix
""")

print("Name-prefix index ready.")
print()

# ============================================================
# STEP 5: BUILD CANDIDATE-SIDE ADDRESS NUMBER INDEX
# EXACT VALIDATED SETTING:
# 4+ digits, frequency <= 500
# ============================================================

print("Building address-number index...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidate_address_numbers AS
SELECT DISTINCT
    c.entity_id,
    c.country,
    trim(x.num) AS address_number
FROM candidates c,
LATERAL UNNEST(
    regexp_extract_all(c.norm_address, '[0-9]{4,}')
) AS x(num)
WHERE length(trim(x.num)) >= 4
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE address_number_frequency AS
SELECT
    country,
    address_number,
    COUNT(*) AS cnt
FROM candidate_address_numbers
GROUP BY country, address_number
""")

print("Address-number index ready.")
print()

# ============================================================
# STEP 6: BUILD CANDIDATE-SIDE ADDRESS TOKEN INDEX
# EXACT VALIDATED SETTING:
# token length >= 4, frequency <= 1000
# ============================================================

print("Building address-token index...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidate_address_tokens AS
SELECT DISTINCT
    c.entity_id,
    c.country,
    trim(t.token) AS token
FROM candidates c,
LATERAL UNNEST(
    string_split(c.norm_address, ' ')
) AS t(token)
WHERE length(trim(t.token)) >= 4
  AND trim(t.token) <> ''
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE address_token_frequency AS
SELECT
    country,
    token,
    COUNT(*) AS token_count
FROM candidate_address_tokens
GROUP BY country, token
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE useful_address_tokens AS
SELECT
    country,
    token,
    token_count
FROM address_token_frequency
WHERE token_count <= 1000
""")

print("Address-token index ready.")
print()

# ============================================================
# STEP 7: CREATE OUTPUT FILE
# ============================================================

print("Preparing output file...")

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")

# ============================================================
# STEP 8: PROCESS SOURCE 1 IN CHUNKS
# ============================================================

total_chunks = (source1_count + CHUNK_SIZE - 1) // CHUNK_SIZE

print()
print("Generating candidates...")
print("Chunk size:", CHUNK_SIZE)
print("Total chunks:", total_chunks)
print()

for chunk_start in range(0, source1_count, CHUNK_SIZE):

    chunk_number = (chunk_start // CHUNK_SIZE) + 1
    chunk_end = min(chunk_start + CHUNK_SIZE, source1_count)

    print(
        f"[Chunk {chunk_number}/{total_chunks}] "
        f"Processing Source 1 rows {chunk_start + 1}-{chunk_end}..."
    )

    # --------------------------------------------------------
    # Source 1 chunk
    # --------------------------------------------------------

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE source1 AS
    SELECT
        entity_id,
        country,
        norm_name,
        norm_address
    FROM source1_all
    WHERE rn > {chunk_start}
      AND rn <= {chunk_end}
    """)

    # --------------------------------------------------------
    # BLOCK 1: EXACT NAME
    # --------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE block_exact_name AS
    SELECT
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id
    FROM source1 s
    JOIN candidates c
        ON s.country = c.country
       AND s.norm_name = c.norm_name
    WHERE s.norm_name <> ''
    """)

    # --------------------------------------------------------
    # BLOCK 2: RARE NAME TOKENS
    # EXACT VALIDATED SETTING
    # --------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE source1_tokens AS
    SELECT DISTINCT
        s.entity_id,
        s.country,
        trim(t.token) AS token
    FROM source1 s,
    LATERAL UNNEST(
        string_split(s.norm_name, ' ')
    ) AS t(token)
    WHERE length(trim(t.token)) >= 4
      AND trim(t.token) <> ''
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE selected_tokens AS
    SELECT
        s.entity_id,
        s.country,
        s.token
    FROM source1_tokens s
    JOIN useful_tokens u
        ON s.country = u.country
       AND s.token = u.token
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY s.entity_id
        ORDER BY u.token_count ASC, s.token
    ) <= 3
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE block_rare_tokens AS
    SELECT DISTINCT
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id
    FROM selected_tokens s
    JOIN candidate_name_tokens c
        ON s.country = c.country
       AND s.token = c.token
    """)

    # --------------------------------------------------------
    # BLOCK 3: 5-CHAR NAME PREFIX
    # EXACT VALIDATED SETTING
    # --------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE source1_name_prefixes AS
    SELECT
        entity_id,
        country,
        substr(norm_name, 1, 5) AS name_prefix
    FROM source1
    WHERE length(norm_name) >= 5
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE block_name_prefix AS
    SELECT DISTINCT
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id
    FROM source1_name_prefixes s
    JOIN name_prefix_frequency f
        ON s.country = f.country
       AND s.name_prefix = f.name_prefix
    JOIN candidate_name_prefixes c
        ON c.country = s.country
       AND c.name_prefix = s.name_prefix
    WHERE f.prefix_count <= 5000
    """)

    # --------------------------------------------------------
    # BLOCK 4: EXACT ADDRESS
    # --------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE block_exact_address AS
    SELECT
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id
    FROM source1 s
    JOIN candidates c
        ON s.country = c.country
       AND s.norm_address = c.norm_address
    WHERE length(s.norm_address) >= 5
    """)

    # --------------------------------------------------------
    # BLOCK 5: ADDRESS NUMBER
    # EXACT VALIDATED SETTING
    # --------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE source1_address_numbers AS
    SELECT DISTINCT
        s.entity_id,
        s.country,
        trim(x.num) AS address_number
    FROM source1 s,
    LATERAL UNNEST(
        regexp_extract_all(s.norm_address, '[0-9]{4,}')
    ) AS x(num)
    WHERE length(trim(x.num)) >= 4
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE block_address_number AS
    SELECT DISTINCT
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id
    FROM source1_address_numbers s
    JOIN address_number_frequency f
        ON s.country = f.country
       AND s.address_number = f.address_number
    JOIN candidate_address_numbers c
        ON c.country = s.country
       AND c.address_number = s.address_number
    WHERE f.cnt <= 500
    """)

    # --------------------------------------------------------
    # BLOCK 6: ADDRESS TOKENS
    # EXACT VALIDATED SETTING
    # --------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE source1_address_tokens AS
    SELECT DISTINCT
        s.entity_id,
        s.country,
        trim(t.token) AS token
    FROM source1 s,
    LATERAL UNNEST(
        string_split(s.norm_address, ' ')
    ) AS t(token)
    WHERE length(trim(t.token)) >= 4
      AND trim(t.token) <> ''
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE selected_address_tokens AS
    SELECT
        s.entity_id,
        s.country,
        s.token
    FROM source1_address_tokens s
    JOIN useful_address_tokens u
        ON s.country = u.country
       AND s.token = u.token
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY s.entity_id
        ORDER BY u.token_count ASC, s.token
    ) <= 2
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE block_address_tokens AS
    SELECT DISTINCT
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id
    FROM selected_address_tokens s
    JOIN candidate_address_tokens c
        ON s.country = c.country
       AND s.token = c.token
    """)

    # --------------------------------------------------------
    # COMBINE ALL VALIDATED BLOCKS
    # --------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE final_candidates AS

    SELECT source1_entity_id, candidate_entity_id
    FROM block_exact_name

    UNION

    SELECT source1_entity_id, candidate_entity_id
    FROM block_rare_tokens

    UNION

    SELECT source1_entity_id, candidate_entity_id
    FROM block_name_prefix

    UNION

    SELECT source1_entity_id, candidate_entity_id
    FROM block_exact_address

    UNION

    SELECT source1_entity_id, candidate_entity_id
    FROM block_address_number

    UNION

    SELECT source1_entity_id, candidate_entity_id
    FROM block_address_tokens
    """)

    # --------------------------------------------------------
    # WRITE THIS CHUNK
    # --------------------------------------------------------

    rows = con.execute("""
    SELECT
        s.entity_id AS source1_entity_id,
        COALESCE(
            string_agg(
                f.candidate_entity_id,
                ',' ORDER BY f.candidate_entity_id
            ),
            ''
        ) AS candidate_entity_ids
    FROM source1 s
    LEFT JOIN final_candidates f
        ON s.entity_id = f.source1_entity_id
    GROUP BY s.entity_id
    ORDER BY s.entity_id
    """).fetchall()

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
        newline=""
    ) as f:

        for source1_id, candidate_ids in rows:
            f.write(
                f"{source1_id}\t{candidate_ids or ''}\n"
            )

    elapsed = (time.time() - start_time) / 60

    print(
        f"[Chunk {chunk_number}/{total_chunks}] DONE "
        f"| elapsed: {elapsed:.1f} min"
    )
    print()

# ============================================================
# FINAL CHECK
# ============================================================

print("==============================================")
print("CANDIDATE GENERATION COMPLETE")
print("==============================================")

print("Output:", OUTPUT_FILE)

with open(
    OUTPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    output_lines = sum(1 for _ in f) - 1

print("Source 1 records:", source1_count)
print("Output rows:", output_lines)

if output_lines == source1_count:
    print("ROW COUNT CHECK: PASS")
else:
    print("ROW COUNT CHECK: FAILED")

elapsed = (time.time() - start_time) / 60

print("Total time:", round(elapsed, 2), "minutes")
print()
print("YOUR BLOCKING PART IS DONE.")
print("==============================================")