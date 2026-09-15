#!/usr/bin/env bash
# Entry 17: first 250 generations of the 18 seed-extension arms (E_SEEDEXT s3-s5, E_SEEDEXT2 s6-s11).
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
while read -r arm fid; do
  python -u "$SRC"/drive_fetch.py "$fid" "data/gens_ext/$arm" --limit 250
done <<'LIST'
A_raw_s3_r16 12YMsUHH7_OoX0Y7c2SQGHb3gHJOljKZ5
A_raw_s4_r16 1TfQ-MKpKqKYMWHbkYhP9QfeiYzngA-ri
A_raw_s5_r16 1vQ6fLrbdrsRxmdWEzavoEZxyIpxVi7wM
B_raw_s3_r16 1197pe4B1wDxDzdmSXwYD2Mf8kJ0_OyAU
B_raw_s4_r16 1ymWHCC0DUv-eXKwCP6KF9UodNNKkKUDl
B_raw_s5_r16 1b-7iCU2H52-ou7OQhmRQ-qrNixuAxEhH
A_raw_s6_r16 1J7fdw7D1fg3rrrvl8FtkFI82q-KAWg17
A_raw_s7_r16 12p9hj4o2vnPdJiY-Skja60IMCEOISlPZ
A_raw_s8_r16 1CK4rwL2TIe8uIyUMgmsepwGZuyeljGpT
A_raw_s9_r16 1VftXa50Hgqvt3eVdQZhRwb6pjsC-Je1Q
A_raw_s10_r16 1yPCfcIa-IpR_8--P4GQIjcuPhYXV_5bH
A_raw_s11_r16 1YrnyeaHAHA9WjFxPziiV3PfWIW7dYIE6
B_raw_s6_r16 1TlLvYRiT1CI_rlFQYjJQvvW1kU6O78S6
B_raw_s7_r16 1BDg4eX2GkgOIJn1AVMJAgmQvheLkLz1O
B_raw_s8_r16 158KnetD3mFxzWbR18LlnVYKOQBWsi7kM
B_raw_s9_r16 1Do7OePwuKFZIz_9wYfDh7OObSWX37lVv
B_raw_s10_r16 1ZeQ-emn8Z73jnc3HaLR9Kv1x7-khYED0
B_raw_s11_r16 1wwekkvKQuWEoVG8z6B4neUN9_-jHdl5P
LIST
echo "[fetch_ext] done $(date)"
