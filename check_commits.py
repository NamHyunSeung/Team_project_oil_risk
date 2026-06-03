import re, subprocess, sys

# 전체 커밋 해시 목록 (7자리)
result = subprocess.run(
    ['git', 'log', '--oneline', '--no-merges'],
    capture_output=True, encoding='utf-8', errors='replace', cwd='.'
)
all_commits = {}
for line in result.stdout.strip().split('\n'):
    parts = line.split(' ', 1)
    if len(parts) == 2:
        all_commits[parts[0]] = parts[1]

# DEVELOPMENT_HISTORY.md에서 해시 추출
with open('docs/DEVELOPMENT_HISTORY.md', encoding='utf-8') as f:
    content = f.read()
doc_hashes = set(re.findall(r'`([0-9a-f]{7})`', content))

# 미기재 커밋
missing = [(h, msg) for h, msg in all_commits.items() if h not in doc_hashes]

print(f'전체 커밋: {len(all_commits)}')
print(f'문서 기재 해시: {len(doc_hashes)}')
print(f'미기재 커밋: {len(missing)}')
print()

skip_keywords = ['docs:', 'chore:']
real_missing = [(h, m) for h, m in missing if not any(k in m for k in skip_keywords)]
print(f'실질 미기재 (docs/chore 제외): {len(real_missing)}')
print()
with open('missing_commits.txt', 'w', encoding='utf-8') as out:
    for h, m in real_missing:
        out.write(f'{h}  {m}\n')
print('missing_commits.txt 저장 완료')
