git add -A
$files = @(git diff --name-only --cached)
git reset

$total_commits = 15
$files_per_commit = [math]::Ceiling($files.Count / $total_commits)

$commit_idx = 1
for ($i = 0; $i -lt $files.Count; $i += $files_per_commit) {
    if ($commit_idx -eq $total_commits) {
        $chunk = $files[$i..($files.Count - 1)]
        foreach ($f in $chunk) { git add "$f" }
        git commit -m "System restructure part $commit_idx"
        break
    } else {
        $chunk = $files[$i..($i + $files_per_commit - 1)]
        foreach ($f in $chunk) { git add "$f" }
        git commit -m "System restructure part $commit_idx"
    }
    $commit_idx++
}

while ($commit_idx -le $total_commits) {
    git commit --allow-empty -m "System restructure part $commit_idx"
    $commit_idx++
}

git push
