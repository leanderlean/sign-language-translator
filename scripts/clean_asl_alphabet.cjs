// const fs = require('fs');
// const path = require('path');
// const repoRoot = path.resolve(__dirname, '..');
// const srcPath = path.join(repoRoot, 'asl_alphabet.json');
// const backupPath = path.join(repoRoot, 'asl_alphabet.backup.json');
// const cleanedPath = path.join(repoRoot, 'asl_alphabet.cleaned.json');

// try {
//   const raw = fs.readFileSync(srcPath, 'utf8');
//   const data = JSON.parse(raw);
//   if (!Array.isArray(data)) throw new Error('Expected an array at top level');
//   const cleaned = data.filter(item => String((item && item.label) || '').toUpperCase() !== 'YES');
//   fs.writeFileSync(backupPath, JSON.stringify(data, null, 2));
//   fs.writeFileSync(cleanedPath, JSON.stringify(cleaned, null, 2));
//   fs.writeFileSync(srcPath, JSON.stringify(cleaned, null, 2));
//   console.log('Cleaned asl_alphabet.json; backup saved to', backupPath, 'and cleaned copy to', cleanedPath);
// } catch (err) {
//   console.error('Failed to clean asl_alphabet.json:', err);
//   process.exit(1);
// }

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const srcPath = path.join(repoRoot, 'asl_alphabet.json');
const backupPath = path.join(repoRoot, 'asl_alphabet.backup.json');
const tightenedPath = path.join(repoRoot, 'asl_alphabet.tightened.json');

// The 0-based landmark indices for the fingertips of the 4 fingers
const FINGER_TIP_LANDMARKS = [8, 12, 16, 20]; 

// Scaling factor: 0.85 reduces the distance from the wrist by 15%
// Make this lower (e.g., 0.75) for an even tighter fist
const TIGHTEN_FACTOR = 0.85; 

try {
  const raw = fs.readFileSync(srcPath, 'utf8');
  const data = JSON.parse(raw);
  
  if (!Array.isArray(data)) throw new Error('Expected an array at top level');

  // Create a deep copy and modify only the "A" labels
  const modifiedData = data.map(item => {
    if (item && item.label === 'A' && Array.isArray(item.features)) {
      // Clone features array to safely edit
      const features = [...item.features]; 

      FINGER_TIP_LANDMARKS.forEach(landmarkIndex => {
        // In a flat array, each landmark starts at index: landmarkIndex * 3
        const baseIdx = landmarkIndex * 3;

        // Multiply x, y, and z by the tightening factor to pull them toward (0,0,0)
        features[baseIdx]     = features[baseIdx] * TIGHTEN_FACTOR;     // X
        features[baseIdx + 1] = features[baseIdx + 1] * TIGHTEN_FACTOR; // Y
        features[baseIdx + 2] = features[baseIdx + 2] * TIGHTEN_FACTOR; // Z
      });

      return { ...item, features };
    }
    return item; // Leave non-"A" labels completely untouched
  });

  // Write files
  fs.writeFileSync(backupPath, JSON.stringify(data, null, 2));
  fs.writeFileSync(tightenedPath, JSON.stringify(modifiedData, null, 2));
  fs.writeFileSync(srcPath, JSON.stringify(modifiedData, null, 2)); // Overwrites main file

  console.log('Successfully tightened "A" fingertips!');
  console.log('Backup saved to:', backupPath);
  console.log('Tightened copy saved to:', tightenedPath);

} catch (err) {
  console.error('Failed to process coordinates:', err);
  process.exit(1);
}