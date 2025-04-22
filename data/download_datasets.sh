#!/usr/bin/env bash
# download_datasets.sh
# ├── data/
# │   ├── swords/   → swords-v1.1_{dev,test}.json.gz
# │   ├── prolex/   → ProLex_v1.0_{dev,test}.csv
# │   └── tsar22/   → tsar2022_{en,es,pt}_{test_none,test_gold}.tsv

set -euo pipefail

ROOT="$(dirname "$(realpath "$0")")/"
mkdir -p "${ROOT}"/{swords,prolex,tsar22}

echo "==> SWORDS"
curl -L -o "${ROOT}/swords/swords-v1.1_dev.json.gz"  \
  "https://github.com/p-lambda/swords/blob/main/assets/parsed/swords-v1.1_dev.json.gz?raw=1"
curl -L -o "${ROOT}/swords/swords-v1.1_test.json.gz" \
  "https://github.com/p-lambda/swords/blob/main/assets/parsed/swords-v1.1_test.json.gz?raw=1"  # :contentReference[oaicite:0]{index=0}

echo "==> ProLex"
curl -L -o "${ROOT}/prolex/ProLex_v1.0_dev.csv" \
  "https://raw.githubusercontent.com/BillyZhang24kobe/LS_Proficiency/main/data/dev/ProLex_v1.0_dev.csv"
curl -L -o "${ROOT}/prolex/ProLex_v1.0_test.csv" \
  "https://raw.githubusercontent.com/BillyZhang24kobe/LS_Proficiency/main/data/test/ProLex_v1.0_test.csv"  # :contentReference[oaicite:1]{index=1}

echo "==> TSAR‑2022 (en / es / pt)"
for lang in en es pt; do
  for split in test_none test_gold; do
    file="tsar2022_${lang}_${split}.tsv"
    url="https://raw.githubusercontent.com/LaSTUS-TALN-UPF/TSAR-2022-Shared-Task/main/datasets/test/${file}"
    curl -L -o "${ROOT}/tsar22/${file}" "${url}"
  done
done                                                                         # :contentReference[oaicite:2]{index=2}

echo "All datasets downloaded into: ${ROOT}"
