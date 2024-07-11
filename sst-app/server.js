const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(express.static('public'));

app.get('/api/images', (req, res) => {
  const folder = req.query.folder;
  console.log(folder)
  const imagesDir = path.join(__dirname, `public/images/${folder}/original`);
  const boxesDir = path.join(__dirname,`public/images/${folder}/test_classification/box`);
  const heatmapsDir = path.join(__dirname,`public/images/${folder}/test_classification/overlays`);
  fs.readdir(imagesDir, (err, originalfiles) => {
    if (err) {
      return res.status(500).json({ error: 'Failed to read images directory' });
    }
  fs.readdir(boxesDir, (err, boxfiles) => {
    if (err) {
      return res.status(500).json({ error: 'Failed to read images directory' });
    }
  fs.readdir(heatmapsDir, (err, heatmapfiles) => {
    if (err) {
      return res.status(500).json({ error: 'Failed to read images directory' });
    }

    const images = originalfiles.map((file, index) => {
      const originalPath = path.join(`images/${folder}/original`, file);
      const boxesPath = boxfiles[index] ? path.join(`images/${folder}/test_classification/box`, boxfiles[index]) : null;
      const heatmapPath = heatmapfiles[index] ? path.join(`images/${folder}/test_classification/overlays`, heatmapfiles[index]) : null;
      return {
        original: originalPath,
        box: boxesPath, 
        heatmap: heatmapPath,
        alt: path.parse(file).name
      };
    });

    res.json(images);
      });
    });
  });
});


const folderPath = path.join(__dirname, 'data');
app.get('/api/folders', (req, res) => {
  console.log('request begin')
  fs.readdir(folderPath, (err, files) => {
      if (err) {
          console.log('error reading folder',err)
          return res.status(500).json({ error: 'Unable to read folder' });
      }

      const folders = files.filter(file => fs.statSync(path.join(folderPath, file)).isDirectory());
      console.log('res: ',folders)
      res.json(folders);
  });
});


app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
