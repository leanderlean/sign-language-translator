const fs = require('fs');
const path = require('path');
const repoRoot = path.resolve(__dirname, '..');
const srcPath = path.join(repoRoot, 'asl_alphabet.json');
const backupPath = path.join(repoRoot, 'asl_alphabet.backup.json');
const cleanedPath = path.join(repoRoot, 'asl_alphabet.cleaned.json');

try {
  const raw = fs.readFileSync(srcPath, 'utf8');
  const data = JSON.parse(raw);
  if (!Array.isArray(data)) throw new Error('Expected an array at top level');
  const cleaned = data.filter(item => String((item && item.label) || '').toUpperCase() !== 'YES');
  fs.writeFileSync(backupPath, JSON.stringify(data, null, 2));
  fs.writeFileSync(cleanedPath, JSON.stringify(cleaned, null, 2));
  fs.writeFileSync(srcPath, JSON.stringify(cleaned, null, 2));
  console.log('Cleaned asl_alphabet.json; backup saved to', backupPath, 'and cleaned copy to', cleanedPath);
} catch (err) {
  console.error('Failed to clean asl_alphabet.json:', err);
  process.exit(1);
}
